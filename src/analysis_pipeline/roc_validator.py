import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
from typing import List, Tuple, Optional
from pathlib import Path
import anthropic


class SimpleROCValidator:
    """ROC validation using Claude API and stratified sampling."""
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        print("Claude API validator initialized")
    
    def validate_and_plot_roc(self, 
                             searcher,
                             queries: List[str],
                             policy_description: str,
                             sample_size: int = 500,
                             companies: Optional[List[str]] = None,
                             output_path: str = "roc_curve.png",
                             save_validation_data: bool = True) -> dict:
        """
        Generate ROC curve using stratified sampling and BART validation.
        
        Args:
            searcher: Semantic searcher instance
            queries: List of search queries
            policy_description: Description for BART classification
            sample_size: Number of snippets to sample and validate
            companies: Optional company filter
            output_path: Path to save ROC curve plot
            
        Returns:
            Dict with ROC metrics and optimal threshold
        """
        print("🔍 Getting search results...")
        
        # Get all search results with low threshold
        all_snippets = searcher.multi_query_search(
            queries=queries,
            min_score=0.1,
            aggregation='max'
        )
        
        if companies:
            all_snippets = searcher.filter_snippets(all_snippets, companies=companies)
        
        print(f"📊 Retrieved {len(all_snippets)} total snippets")
        
        # Stratified sampling across score range
        sample_snippets = self._stratified_sample(all_snippets, sample_size)
        print(f"🎯 Sampled {len(sample_snippets)} snippets for validation")
        
        # BART validation
        print("🤖 Running CLAUDE validation...")
        scores = [s.similarity_score for s in sample_snippets]
        labels, validation_details = self._validate_with_claude(sample_snippets, policy_description)

        # Save validation data if requested
        if save_validation_data:
            import pandas as pd
            validation_df = pd.DataFrame({
                'text': [s.text[:200] + "..." if len(s.text) > 200 else s.text for s in sample_snippets],
                'full_text': [s.text for s in sample_snippets],
                'similarity_score': scores,
                'claude_label': labels,
                'claude_confidence': [d.get('confidence', 0.0) for d in validation_details],
                'claude_decision': [d.get('decision', 'UNKNOWN') for d in validation_details],
                'ticker': [s.ticker for s in sample_snippets],
                'year': [s.year for s in sample_snippets],
                'quarter': [s.quarter for s in sample_snippets]
            })
            
            # Sort by similarity score for easier inspection
            validation_df = validation_df.sort_values('similarity_score', ascending=False)
            
            # Save validation DataFrame
            output_dir = Path(output_path).parent
            validation_csv_path = output_dir / "validation_data.csv"
            validation_df.to_csv(validation_csv_path, index=False)
            print(f"Validation data saved: {validation_csv_path}")
            
            # Quick summary
            print(f"Claude validation summary:")
            print(f"  Total snippets: {len(validation_df)}")
            print(f"  Claude labeled as relevant: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
            print(f"  High similarity (>0.4): {sum(validation_df['similarity_score'] > 0.4)} snippets")
            print(f"  High sim + Claude relevant: {sum((validation_df['similarity_score'] > 0.4) & (validation_df['claude_label']))} snippets")

        # Now labels is List[bool], which is valid input for roc_curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Generate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        
        # Find optimal threshold (closest to top-left corner)
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
        
        # Plot ROC curve
        self._plot_roc(
            fpr, tpr, auc_score,
            optimal_threshold=optimal_threshold,
            optimal_idx=optimal_idx,
            output_path=output_path,
            policy_description=policy_description   # <-- add this
        )

        
        return {
            'auc': auc_score,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx],
            'total_snippets': len(all_snippets),
            'validated_snippets': len(sample_snippets),
            'validation_data_saved': save_validation_data
        }
    
    def _stratified_sample(self, snippets: List, sample_size: int) -> List:
        """Stratified sampling across similarity score range."""
        if len(snippets) <= sample_size:
            return snippets
        
        # Sort by similarity score
        sorted_snippets = sorted(snippets, key=lambda x: x.similarity_score)
        
        # Create bins
        n_bins = min(20, len(sorted_snippets) // 10)
        bin_size = len(sorted_snippets) // n_bins
        samples_per_bin = sample_size // n_bins
        
        sampled = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_snippets)
            bin_snippets = sorted_snippets[start_idx:end_idx]
            
            # Sample from this bin
            n_samples = min(samples_per_bin, len(bin_snippets))
            indices = np.random.choice(len(bin_snippets), n_samples, replace=False)
            sampled.extend([bin_snippets[idx] for idx in indices])
        
        return sampled
    
    def _validate_with_claude(self, snippets: List, policy_description: str) -> Tuple[List[bool], List[dict]]:
        """Validate snippets with Claude API and return detailed results."""
        labels = []
        validation_details = []
        
        for i, snippet in enumerate(snippets):
            print(f"  Processing snippet {i+1}/{len(snippets)}...", end='\r')
            
            prompt = f"""Is this text from an earnings call specifically discussing climate disclosure, ESG reporting, or sustainability strategy transparency?

Text: "{snippet.text[:1000]}..."

Format your response as:
Decision: [YES/NO]
Confidence: [1-10]"""

            try:
                response = self._call_claude_api(prompt)
                
                # Parse Claude's response
                lines = response.split('\n')
                decision = "NO"  # default
                confidence = 5    # default
                
                for line in lines:
                    if line.startswith("Decision:"):
                        decision = line.split(":", 1)[1].strip().upper()
                    elif line.startswith("Confidence:"):
                        try:
                            confidence = int(line.split(":", 1)[1].strip())
                        except:
                            confidence = 5
                
                is_relevant = decision == "YES"
                labels.append(is_relevant)
                validation_details.append({
                    'confidence': confidence / 10.0,  # normalize to 0-1
                    'decision': decision
                })
                
            except Exception as e:
                print(f"Error with snippet {i}: {e}")
                # Default to not relevant on error
                labels.append(False)
                validation_details.append({
                    'confidence': 0.0,
                    'decision': "ERROR"
                })
        
        print()  # Clear the progress line
        return labels, validation_details
    
    def _call_claude_api(self, prompt: str) -> str:
        """Make API call to Claude using official client."""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            raise Exception(f"Claude API call failed: {str(e)}")
    
    def _plot_roc(self, fpr, tpr, auc_score, optimal_threshold, optimal_idx, output_path: str, policy_description: str):
        """Plot and save ROC curve."""
        plt.figure(figsize=(8, 6))
        
        # ROC curve
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random Classifier')
        
        # Fix optimal point plotting
        optimal_fpr_actual = fpr[optimal_idx]
        optimal_tpr_actual = tpr[optimal_idx]
        plt.scatter(optimal_fpr_actual, optimal_tpr_actual, color='red', s=100, zorder=5, 
                   label=f'Optimal (τ={optimal_threshold:.3f})')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {policy_description}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add AUC score as text
        plt.text(0.6, 0.2, f'AUC = {auc_score:.3f}\nOptimal τ = {optimal_threshold:.3f}', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 ROC curve saved: {output_path}")
        print(f"✅ AUC: {auc_score:.3f}, Optimal threshold: {optimal_threshold:.3f}")


# Usage example:
def validate_policy_roc(searcher, queries, policy_description, **kwargs):
    """Convenience function for ROC validation."""
    validator = SimpleROCValidator()
    return validator.validate_and_plot_roc(
        searcher, queries, policy_description, **kwargs
    )