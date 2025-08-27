#!/usr/bin/env python3
"""
Threshold Sensitivity Analyzer for Semantic Search

Analyzes how the number of retrieved snippets changes with different similarity thresholds.
Helps determine optimal thresholds for different analysis types and queries.
Now includes ROC curve validation using BART-large-MNLI for ground truth classification.

Author: Marleen de Jonge
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import torch
from transformers import pipeline
from tqdm import tqdm

from roc_validator import ROCValidationAnalyzer


# Import base searcher
try:
    from .base_searcher import BaseSemanticSearcher
    
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from base_searcher import BaseSemanticSearcher

logger = logging.getLogger(__name__)


class SensitivityAnalyzer(BaseSemanticSearcher):
    """
    Analyzer for threshold sensitivity analysis with ROC curve validation.
    
    Tests how snippet retrieval counts change across different similarity thresholds
    and validates results using BART-large-MNLI for ground truth classification.
    """
    
    def __init__(self, index_path: Path, config_path: Optional[str] = None):
        """Initialize the sensitivity analyzer."""
        super().__init__(index_path, config_path)
        
        # ROC validation components
        self.bart_classifier = None
        self.validation_cache = {}
        self.device = self._setup_device()
        
        logger.info("✅ Sensitivity Analyzer initialized")
    
    def _setup_device(self) -> str:
        """Setup optimal device for BART model."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_bart_classifier(self) -> None:
        """Load BART-large-MNLI classifier for ROC validation."""
        if self.bart_classifier is not None:
            return
            
        logger.info("🤖 Loading BART-large-MNLI classifier for ROC validation...")
        
        try:
            self.bart_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
            logger.info("✅ BART-large-MNLI loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load BART classifier: {e}")
            raise
    
    def threshold_sensitivity_analysis(self,
                                     policy_name: str,
                                     threshold_range: Tuple[float, float] = (0.3, 0.7),
                                     num_points: int = 10,
                                     companies: Optional[List[str]] = None,
                                     plot: bool = True,
                                     log_scale: bool = True,
                                     run_roc_validation: bool = False,
                                     roc_sample_size: int = 500,
                                     output_dir: Optional[Path] = None,
                                     **kwargs) -> Dict[str, Any]:
        """
        Perform threshold sensitivity analysis for a specific policy.
        
        Args:
            policy_name: Name of policy to analyze
            threshold_range: (min_threshold, max_threshold)
            num_points: Number of threshold points to test
            companies: Optional company filter
            plot: Whether to create visualization
            log_scale: Whether to use log scale for snippet count plot
            run_roc_validation: Whether to run ROC curve validation with BART-MNLI
            roc_sample_size: Number of snippets to use for ROC validation
            output_dir: Directory to save plots
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        logger.info(f"Running threshold sensitivity for: {policy_name}")
        
        # Get queries for this policy
        if self.config:
            try:
                queries = self.config.get_query_list('policies', policy_name)
            except:
                queries = [f"{policy_name} climate policy"]
        else:
            # Fallback queries
            policy_queries = {
                'paris_agreement': ['Paris Agreement COP21 international climate accord'],
                'eu_green_deal': ['European Green Deal climate policy'],
                'us_ira': ['Inflation Reduction Act Biden 2022 climate legislation'],
                'cop_meetings': ['COP climate conference international negotiations']
            }
            queries = policy_queries.get(policy_name, [f"{policy_name} climate policy"])
        
        # Create threshold range
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        
        # Test each threshold
        results = []
        
        for threshold in thresholds:
            logger.info(f"Testing threshold: {threshold:.3f}")
            
            # Search with this threshold
            snippets = self.multi_query_search(
                queries=queries,
                min_score=threshold,
                aggregation='max'
            )
            
            # Apply filters if specified
            if companies:
                snippets = self.filter_snippets(snippets, companies=companies)
            
            # Count results
            count = len(snippets)
            avg_score = np.mean([s.similarity_score for s in snippets]) if snippets else 0
            
            results.append({
                'threshold': threshold,
                'snippet_count': count,
                'avg_score': avg_score
            })
            
            logger.debug(f"Threshold {threshold:.3f}: {count} snippets, avg score {avg_score:.3f}")
        
        # Convert to DataFrame
        sensitivity_df = pd.DataFrame(results)
        
        # Calculate metrics
        analysis_result = {
            'policy_name': policy_name,
            'sensitivity_data': sensitivity_df,
            'threshold_range': threshold_range,
            'queries_used': queries,
            'optimal_thresholds': self._find_optimal_thresholds(sensitivity_df),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Run ROC validation if requested
        if run_roc_validation:
            logger.info(f"🎯 Running ROC validation for {policy_name}...")
            roc_results = self._run_roc_validation(
                policy_name=policy_name,
                queries=queries,
                sample_size=roc_sample_size,
                threshold_range=threshold_range,
                companies=companies,
                output_dir=output_dir
            )
            analysis_result['roc_validation'] = roc_results
        
        # Create plot if requested
        if plot:
            plot_path = self._create_sensitivity_plot(
                policy_name=policy_name,
                sensitivity_df=sensitivity_df,
                output_dir=output_dir,
                log_scale=log_scale
            )
            analysis_result['plot_path'] = plot_path
        
        logger.info(f"✅ Sensitivity analysis complete for {policy_name}")
        return analysis_result
    
    def _find_optimal_thresholds(self, sensitivity_df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """
        Find optimal thresholds using different methods.
        
        Args:
            sensitivity_df: DataFrame with threshold sensitivity data
            
        Returns:
            Dictionary with optimal thresholds from different methods
        """
        if sensitivity_df.empty:
            return {}
        
        optimal_thresholds = {}
        
        # Method 1: Elbow method (maximum curvature)
        try:
            counts = sensitivity_df['snippet_count'].values
            thresholds = sensitivity_df['threshold'].values
            
            # Calculate second derivative (curvature)
            if len(counts) >= 3:
                second_deriv = np.gradient(np.gradient(counts))
                elbow_idx = np.argmax(np.abs(second_deriv))
                optimal_thresholds['elbow_method'] = thresholds[elbow_idx]
        except:
            optimal_thresholds['elbow_method'] = None
        
        # Method 2: Steepest decline (largest negative slope)
        try:
            slopes = np.diff(sensitivity_df['snippet_count'].values)
            if len(slopes) > 0:
                steepest_idx = np.argmin(slopes)  # Most negative slope
                optimal_thresholds['steepest_decline'] = sensitivity_df.iloc[steepest_idx]['threshold']
        except:
            optimal_thresholds['steepest_decline'] = None
        
        # Method 3: Quality-quantity balance (highest score with reasonable count)
        try:
            # Find threshold where avg_score * log(count) is maximized
            df_positive = sensitivity_df[sensitivity_df['snippet_count'] > 0].copy()
            if not df_positive.empty:
                df_positive['quality_quantity_score'] = (
                    df_positive['avg_score'] * np.log(df_positive['snippet_count'] + 1)
                )
                best_idx = df_positive['quality_quantity_score'].idxmax()
                optimal_thresholds['quality_quantity_balance'] = df_positive.loc[best_idx, 'threshold']
        except:
            optimal_thresholds['quality_quantity_balance'] = None
        
        return optimal_thresholds
    
    def _create_sensitivity_plot(self,
                               policy_name: str,
                               sensitivity_df: pd.DataFrame,
                               output_dir: Optional[Path] = None,
                               log_scale: bool = True) -> Optional[Path]:
        """
        Create sensitivity analysis visualization.
        
        Args:
            policy_name: Name of policy
            sensitivity_df: DataFrame with sensitivity data
            output_dir: Output directory for plot
            log_scale: Whether to use log scale for snippet count plot
            
        Returns:
            Path to saved plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            thresholds = sensitivity_df['threshold']
            counts = sensitivity_df['snippet_count']
            scores = sensitivity_df['avg_score']
            
            # Plot 1: Snippet count vs threshold (with optional log scale)
            ax1.plot(thresholds, counts, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.set_xlabel('Similarity Threshold')
            ax1.set_ylabel('Number of Snippets Retrieved', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'Threshold Sensitivity Analysis: {policy_name.replace("_", " ").title()}')
            
            # Apply log scale if requested
            if log_scale:
                ax1.set_yscale('log')
                ax1.set_ylabel('Number of Snippets Retrieved (Log Scale)', color='b')
            
            # Plot 2: Average score vs threshold
            ax2.plot(thresholds, scores, 'r-', linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('Similarity Threshold')
            ax2.set_ylabel('Average Similarity Score', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Average Quality vs Threshold')
            
            plt.tight_layout()
            
            # Save plot
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_path = output_dir / f"{policy_name}_threshold_sensitivity.png"
            else:
                plot_path = Path(f"{policy_name}_threshold_sensitivity.png")
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Sensitivity plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Failed to create sensitivity plot: {e}")
            return None
    
    def multi_policy_sensitivity(self,
                                policy_names: List[str],
                                threshold_range: Tuple[float, float] = (0.3, 0.7),
                                num_points: int = 10,
                                output_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run sensitivity analysis for multiple policies.
        
        Args:
            policy_names: List of policy names
            threshold_range: Threshold range to test
            num_points: Number of points to test
            output_dir: Output directory
            
        Returns:
            Dictionary with results for each policy
        """
        results = {}
        
        for policy in policy_names:
            try:
                logger.info(f"Running sensitivity for: {policy}")
                policy_result = self.threshold_sensitivity_analysis(
                    policy_name=policy,
                    threshold_range=threshold_range,
                    num_points=num_points,
                    plot=True,
                    output_dir=output_dir / policy if output_dir else None
                )
                results[policy] = policy_result
                
            except Exception as e:
                logger.error(f"Failed sensitivity for {policy}: {e}")
                continue
        
        # Create comparison plot
        if len(results) > 1 and output_dir:
            self._create_comparison_plot(results, output_dir)
        
        return results
    
    def _create_comparison_plot(self, 
                              results: Dict[str, Dict[str, Any]], 
                              output_dir: Path) -> None:
        """Create comparison plot across multiple policies."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for policy, result in results.items():
                df = result['sensitivity_data']
                ax.plot(df['threshold'], df['snippet_count'], 
                       label=policy.replace('_', ' ').title(), 
                       linewidth=2, marker='o', markersize=4)
            
            ax.set_xlabel('Similarity Threshold')
            ax.set_ylabel('Number of Snippets Retrieved')
            ax.set_title('Threshold Sensitivity Comparison Across Policies')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            comparison_path = output_dir / "threshold_sensitivity_comparison.png"
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Comparison plot saved: {comparison_path}")
            
        except Exception as e:
            logger.error(f"Failed to create comparison plot: {e}")
    
    def _run_roc_validation(self,
                       policy_name: str,
                       queries: List[str],
                       sample_size: int,
                       threshold_range: Tuple[float, float],
                       companies: Optional[List[str]] = None,
                       output_dir: Optional[Path] = None,
                       **kwargs) -> Dict[str, Any]:  # <-- Add **kwargs here
        """
        Run ROC curve validation using BART-large-MNLI.
        
        Args:
            policy_name: Name of policy being validated
            queries: Search queries for this policy
            sample_size: Number of snippets to validate
            threshold_range: Range of thresholds to test
            companies: Optional company filter
            output_dir: Output directory for ROC plots
            
        Returns:
            Dictionary with ROC validation results
        """
        logger.info(f"🎯 Running improved ROC validation for {policy_name}...")
    
        # Initialize the improved ROC analyzer
        roc_analyzer = ROCValidationAnalyzer()
        
        # Define policy description for BART classification
        policy_descriptions = {
            'paris_agreement': 'climate policy, Paris Agreement, international climate accord, global warming targets',
            'eu_green_deal': 'European Union climate policy, Green Deal, EU environmental regulation',
            'us_ira': 'United States climate legislation, Inflation Reduction Act, Biden climate policy',
            'cop_meetings': 'climate conference, international climate negotiations, COP summit, UNFCCC'
        }
        
        policy_description = policy_descriptions.get(
            policy_name, 
            f"{policy_name.replace('_', ' ')} climate policy"
        )
        
        # Use boundary_samples from kwargs, with sample_size as fallback
        boundary_samples = kwargs.get('boundary_samples', min(sample_size // 5, 100))
        num_points = kwargs.get('num_points', 10)
        
        # Generate ROC curve with improved boundary sampling
        roc_data = roc_analyzer.generate_roc_curve_with_boundary_sampling(
            searcher=self,  # Pass self as the searcher
            queries=queries,
            policy_description=policy_description,
            threshold_range=threshold_range,
            num_points=num_points,
            boundary_samples=boundary_samples,
            companies=companies
        )
        
        # Create ROC plots if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            roc_plot_path = roc_analyzer.plot_roc_curves(
                roc_data=roc_data,
                output_path=output_dir / f"{policy_name}_improved_roc_validation.png",
                title_suffix=f" ({policy_name})"
            )
            roc_data['plot_path'] = roc_plot_path
            
            # Export validation results
            export_paths = roc_analyzer.export_validation_results(
                roc_data=roc_data,
                output_dir=output_dir
            )
            roc_data['export_paths'] = export_paths
        
        logger.info(f"✅ Improved ROC validation complete for {policy_name}")
        logger.info(f"   AUC: {roc_data['auc']:.3f}")
        logger.info(f"   Optimal threshold: {roc_data['optimal_threshold']:.3f}")
        
        return roc_data
    
    def _validate_snippets_with_bart(self,
                                   snippets: List,
                                   policy_description: str,
                                   batch_size: int = 8,
                                   confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Validate snippets using BART-large-MNLI classification."""
        texts = [snippet.text for snippet in snippets]
        semantic_scores = [snippet.similarity_score for snippet in snippets]
        
        bart_predictions = []
        bart_confidences = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="BART Validation"):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                # Check cache first
                cache_key = hash(text + policy_description)
                if cache_key in self.validation_cache:
                    prediction, confidence = self.validation_cache[cache_key]
                else:
                    # Run BART classification
                    result = self.bart_classifier(
                        text,
                        candidate_labels=["relevant to " + policy_description, "not relevant"],
                        hypothesis_template="This text is {}."
                    )
                    
                    # Extract prediction and confidence
                    relevant_score = result['scores'][0] if result['labels'][0].startswith('relevant') else result['scores'][1]
                    prediction = relevant_score > confidence_threshold
                    confidence = relevant_score
                    
                    # Cache result
                    self.validation_cache[cache_key] = (prediction, confidence)
                
                bart_predictions.append(prediction)
                bart_confidences.append(confidence)
        
        return {
            'semantic_scores': semantic_scores,
            'bart_predictions': bart_predictions,
            'bart_confidences': bart_confidences,
            'policy_description': policy_description,
            'total_snippets': len(snippets),
            'bart_positive': sum(bart_predictions),
            'bart_negative': len(bart_predictions) - sum(bart_predictions)
        }
    
    def _generate_roc_curve_data(self,
                               validation_results: Dict[str, Any],
                               threshold_range: Tuple[float, float],
                               num_points: int = 10) -> Dict[str, Any]:
        """Generate ROC curve data across different similarity thresholds."""
        semantic_scores = np.array(validation_results['semantic_scores'])
        bart_labels = np.array(validation_results['bart_predictions'])
        
        # Test different similarity thresholds
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        
        tpr_scores = []  # True Positive Rate
        fpr_scores = []  # False Positive Rate
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            # Semantic search predictions at this threshold
            semantic_predictions = semantic_scores >= threshold
            
            # Calculate confusion matrix elements
            tp = np.sum((semantic_predictions == True) & (bart_labels == True))
            fp = np.sum((semantic_predictions == True) & (bart_labels == False))
            tn = np.sum((semantic_predictions == False) & (bart_labels == False))
            fn = np.sum((semantic_predictions == False) & (bart_labels == True))
            
            # Calculate metrics (with zero division protection)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1 - Specificity
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tpr
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            tpr_scores.append(tpr)
            fpr_scores.append(fpr)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Calculate AUC
        auc_score = auc(fpr_scores, tpr_scores)
        
        # Find optimal threshold (closest to top-left corner)
        distances = np.sqrt((np.array(fpr_scores) - 0)**2 + (np.array(tpr_scores) - 1)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
        
        # Find threshold with best F1 score
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds[best_f1_idx]
        
        roc_data = {
            'thresholds': thresholds.tolist(),
            'fpr': fpr_scores,
            'tpr': tpr_scores,
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'auc': auc_score,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr_scores[optimal_idx],
            'optimal_fpr': fpr_scores[optimal_idx],
            'best_f1_threshold': best_f1_threshold,
            'best_f1_score': f1_scores[best_f1_idx],
            'validation_summary': validation_results,
            'bart_positive_rate': validation_results['bart_positive'] / validation_results['total_snippets']
        }
        
        logger.info(f"✅ ROC analysis complete:")
        logger.info(f"   AUC: {auc_score:.3f}")
        logger.info(f"   Optimal threshold: {optimal_threshold:.3f} (TPR: {tpr_scores[optimal_idx]:.3f}, FPR: {fpr_scores[optimal_idx]:.3f})")
        logger.info(f"   Best F1 threshold: {best_f1_threshold:.3f} (F1: {f1_scores[best_f1_idx]:.3f})")
        
        return roc_data
    
    def _create_roc_plots(self,
                         roc_data: Dict[str, Any],
                         policy_name: str,
                         output_dir: Path) -> Path:
        """Create ROC curve and precision-recall curve plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        ax1.plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=2, 
                label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random Classifier')
        ax1.scatter(roc_data['optimal_fpr'], roc_data['optimal_tpr'], 
                   color='red', s=100, zorder=5, label=f'Optimal (τ={roc_data["optimal_threshold"]:.3f})')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve: {policy_name.replace("_", " ").title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax2.plot(roc_data['recall'], roc_data['precision'], 'g-', linewidth=2, label='Precision-Recall')
        best_f1_idx = np.argmax(roc_data['f1'])
        ax2.scatter(roc_data['recall'][best_f1_idx], roc_data['precision'][best_f1_idx], 
                   color='red', s=100, zorder=5, 
                   label=f'Best F1 (τ={roc_data["best_f1_threshold"]:.3f}, F1={roc_data["best_f1_score"]:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Threshold vs Metrics
        ax3.plot(roc_data['thresholds'], roc_data['precision'], 'b-', label='Precision', linewidth=2)
        ax3.plot(roc_data['thresholds'], roc_data['recall'], 'r-', label='Recall', linewidth=2)
        ax3.plot(roc_data['thresholds'], roc_data['f1'], 'g-', label='F1 Score', linewidth=2)
        ax3.axvline(roc_data['optimal_threshold'], color='orange', linestyle='--', 
                   label=f'Optimal ROC (τ={roc_data["optimal_threshold"]:.3f})')
        ax3.axvline(roc_data['best_f1_threshold'], color='purple', linestyle='--', 
                   label=f'Best F1 (τ={roc_data["best_f1_threshold"]:.3f})')
        ax3.set_xlabel('Similarity Threshold')
        ax3.set_ylabel('Score')
        ax3.set_title('Metrics vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # BART validation summary
        ax4.bar(['BART Positive', 'BART Negative'], 
               [roc_data['validation_summary']['bart_positive'], 
                roc_data['validation_summary']['bart_negative']], 
               color=['green', 'red'], alpha=0.7)
        ax4.set_ylabel('Number of Snippets')
        ax4.set_title(f'BART-MNLI Classification Results\n({roc_data["validation_summary"]["total_snippets"]} total snippets)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'ROC Validation Analysis: {policy_name.replace("_", " ").title()}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{policy_name}_roc_validation.png"
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 ROC validation plots saved: {plot_path}")
        return plot_path


def run_sensitivity_analysis(index_path: Path, 
                           policy_name: str,
                           output_dir: Optional[Path] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function for running sensitivity analysis.
    
    Args:
        index_path: Path to semantic index
        policy_name: Policy to analyze
        output_dir: Output directory
        **kwargs: Additional arguments
        
    Returns:
        Sensitivity analysis results
    """
    analyzer = SensitivityAnalyzer(index_path)
    return analyzer.threshold_sensitivity_analysis(
        policy_name=policy_name,
        output_dir=output_dir,
        **kwargs
    )