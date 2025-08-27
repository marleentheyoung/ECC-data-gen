#!/usr/bin/env python3
"""
Master Climate Analysis Script - Run All Analyses

Comprehensive script that runs semantic search analysis for all climate aspects:
- Policy analysis (existing)
- Risk analysis (new)
- Opportunity analysis (new)

This script:
1. Loads the semantic index
2. Runs policy analysis for all major climate policies
3. Runs risk analysis for climate risks
4. Runs opportunity analysis for climate opportunities
5. Generates time series data and exports to CSV
6. Creates individual visualizations
7. Creates comparative analysis plots
8. Organizes all outputs

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --index-path custom/path --output-dir custom/outputs
    python scripts/run_all.py --policies paris_agreement,eu_green_deal --threshold 0.40
    python scripts/run_all.py --analysis-types policy,risk,opportunity

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

try:
    from analysis_pipeline.policy_analysis import PolicyAnalyzer
    from visualization.policy_plots import PolicyVisualizer
    from analysis_pipeline.sensitivity_analyzer import SensitivityAnalyzer
    from analysis_pipeline.risk_analyzer import RiskAnalyzer
    from analysis_pipeline.opportunity_analyzer import OpportunityAnalyzer
    from  visualization.risk_opportunity_visualizer import RiskOpportunityVisualizer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root directory")
    print("And that all analyzer and visualizer files exist")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/run_all_analysis.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class MasterClimateRunner:
    """
    Master orchestrator for complete climate analysis pipeline.
    
    Handles policy, risk, and opportunity analysis, data export, and visualization.
    """
    
    def __init__(self, 
                 index_path: Path,
                 output_dir: Path,
                 config_path: str = None):
        """
        Initialize the master runner.
        
        Args:
            index_path: Path to semantic index
            output_dir: Base output directory
            config_path: Optional configuration file path
        """
        self.index_path = Path(index_path)
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # Create output directory structure
        self.setup_output_directories()
        
        # Add sensitivity analyzer
        logger.info("🔍 Initializing sensitivity analyzer...")
        self.sensitivity_analyzer = SensitivityAnalyzer(self.index_path, self.config_path)
        
        # Initialize components
        logger.info("🚀 Initializing analyzers...")
        self.policy_analyzer = PolicyAnalyzer(self.index_path, self.config_path)
        self.risk_analyzer = RiskAnalyzer(self.index_path, self.config_path)
        self.opportunity_analyzer = OpportunityAnalyzer(self.index_path, self.config_path)
        
        logger.info("🎨 Initializing visualizers...")
        self.policy_visualizer = PolicyVisualizer(
            jof_style=True,
            output_dir=self.output_dir / "visualizations" / "policies"
        )
        self.risk_opp_visualizer = RiskOpportunityVisualizer(
            jof_style=True,
            output_dir=self.output_dir / "visualizations" / "risk_opportunity"
        )
        
        # Define all policies to analyze
        self.all_policies = [
            'paris_agreement',
            'eu_green_deal', 
            'us_ira',
            'cop_meetings'
        ]
        
        # Storage for results
        self.policy_results = {}
        self.risk_results = {}
        self.opportunity_results = {}
        self.csv_files = {
            'policies': {},
            'risk': None,
            'opportunity': None
        }
        
        logger.info("✅ Master Climate Runner initialized")
    
    def run_sensitivity_analysis(self, 
                           policies: List[str] = None,
                           threshold_range: Tuple[float, float] = (0.3, 0.7),
                           num_points: int = 20,
                           **kwargs) -> Dict[str, Any]:
        """Run threshold sensitivity analysis for policies."""
        if policies is None:
            policies = self.all_policies
        
        logger.info(f"🔍 Starting sensitivity analysis for {len(policies)} policies...")
        
        sensitivity_results = {}
        
        for policy in policies:
            try:
                logger.info(f"📊 Running sensitivity analysis for: {policy}...")
                
                # Run sensitivity analysis
                analysis_result = self.sensitivity_analyzer.threshold_sensitivity_analysis(
                    policy_name=policy,
                    threshold_range=threshold_range,
                    num_points=num_points,
                    plot=True,
                    run_roc_validation=True,  # Enable ROC validation
                    output_dir=self.output_dir / "sensitivity_analysis" / policy,
                    **kwargs
                )
                
                sensitivity_results[policy] = analysis_result
                
                # Log key findings
                optimal_thresholds = analysis_result['optimal_thresholds']
                logger.info(f"✅ {policy} sensitivity complete:")
                for method, threshold in optimal_thresholds.items():
                    if threshold:
                        logger.info(f"   {method}: {threshold:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Failed sensitivity analysis for {policy}: {str(e)}")
                continue
        
        # Create comparison plot across policies
        try:
            self._create_sensitivity_comparison_plot(sensitivity_results)
        except Exception as e:
            logger.error(f"❌ Failed to create sensitivity comparison: {str(e)}")
        
        logger.info(f"📈 Sensitivity analysis complete for {len(sensitivity_results)} policies")
        return sensitivity_results

    def _create_sensitivity_comparison_plot(self, sensitivity_results: Dict[str, Any]) -> None:
        """Create comparison plot of optimal thresholds across policies."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Extract optimal thresholds for comparison
            comparison_data = []
            for policy, results in sensitivity_results.items():
                optimal_thresholds = results['optimal_thresholds']
                for method, threshold in optimal_thresholds.items():
                    if threshold is not None:
                        comparison_data.append({
                            'policy': policy.replace('_', ' ').title(),
                            'method': method,
                            'threshold': threshold
                        })
            
            if not comparison_data:
                logger.warning("No optimal thresholds found for comparison plot")
                return
            
            df = pd.DataFrame(comparison_data)
            
            # Create comparison plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot grouped bar chart
            methods = df['method'].unique()
            policies = df['policy'].unique()
            
            x = np.arange(len(policies))
            width = 0.8 / len(methods)
            
            for i, method in enumerate(methods):
                method_data = df[df['method'] == method]
                thresholds = [method_data[method_data['policy'] == p]['threshold'].iloc[0] 
                            if len(method_data[method_data['policy'] == p]) > 0 else 0 
                            for p in policies]
                
                ax.bar(x + i * width, thresholds, width, 
                    label=method.replace('_', ' ').title(), alpha=0.8)
            
            ax.set_xlabel('Policy')
            ax.set_ylabel('Optimal Threshold')
            ax.set_title('Optimal Thresholds Comparison Across Policies')
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(policies, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            comparison_plot_path = self.output_dir / "sensitivity_analysis" / "threshold_comparison.png"
            comparison_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Sensitivity comparison plot saved: {comparison_plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating sensitivity comparison plot: {e}")
        
    def setup_output_directories(self) -> None:
        """Create organized output directory structure."""
        directories = [
            self.output_dir / "analysis_results" / "policies",
            self.output_dir / "analysis_results" / "risk",
            self.output_dir / "analysis_results" / "opportunity",
            self.output_dir / "csv_exports", 
            self.output_dir / "visualizations" / "policies",
            self.output_dir / "visualizations" / "risk_opportunity",
            self.output_dir / "visualizations" / "comparisons",
            self.output_dir / "reports",
            Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Output directories created in: {self.output_dir}")
    
    def run_policy_analysis(self, 
                           policies: List[str] = None,
                           frequency: str = 'quarter',
                           start_date: str = None,
                           end_date: str = None,
                           min_score: float = None) -> Dict[str, Any]:
        """Run policy analysis (existing functionality)."""
        if policies is None:
            policies = self.all_policies
        
        logger.info(f"🔍 Starting policy analysis for {len(policies)} policies...")
        
        successful_analyses = 0
        failed_analyses = 0
        
        for policy in policies:
            try:
                logger.info(f"📊 Analyzing policy: {policy}...")
                
                analysis_result = self.policy_analyzer.analyze_policy_attention(
                    policy_name=policy,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    min_score=min_score
                )
                
                # Store results
                self.policy_results[policy] = analysis_result
                
                # Export time series to CSV
                csv_path = self.export_policy_csv(policy, analysis_result['timeseries'])
                self.csv_files['policies'][policy] = csv_path
                
                # Export complete analysis to JSON
                self.export_policy_json(policy, analysis_result)
                
                total_mentions = analysis_result['summary_statistics']['total_mentions']
                avg_attention = analysis_result['summary_statistics']['attention_statistics']['mean']
                
                logger.info(f"✅ {policy}: {total_mentions} mentions, avg score: {avg_attention:.3f}")
                successful_analyses += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze policy {policy}: {str(e)}")
                failed_analyses += 1
                continue
        
        logger.info(f"📈 Policy analysis complete: {successful_analyses} successful, {failed_analyses} failed")
        return self.policy_results
    
    def run_risk_analysis(self,
                         frequency: str = 'quarter',
                         start_date: str = None,
                         end_date: str = None,
                         min_score: float = None) -> Dict[str, Any]:
        """Run climate risk analysis."""
        logger.info("🔍 Starting climate risk analysis...")
        
        try:
            # Create risk time series
            timeseries = self.risk_analyzer.create_risk_timeseries(
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                min_score=min_score
            )
            
            if timeseries.empty:
                logger.warning("No risk data found")
                return {}
            
            # Store results
            self.risk_results = {
                'timeseries': timeseries,
                'analysis_type': 'climate_risks',
                'total_mentions': timeseries['similarity_score_count'].sum(),
                'avg_score': timeseries['similarity_score_mean'].mean(),
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Export to CSV
            csv_path = self.export_risk_csv(timeseries)
            self.csv_files['risk'] = csv_path
            
            # Export to JSON
            self.export_risk_json(self.risk_results)
            
            logger.info(f"✅ Risk analysis: {self.risk_results['total_mentions']} mentions, "
                       f"avg score: {self.risk_results['avg_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed risk analysis: {str(e)}")
            logger.error(traceback.format_exc())
        
        return self.risk_results
    
    def run_opportunity_analysis(self,
                                frequency: str = 'quarter',
                                start_date: str = None,
                                end_date: str = None,
                                min_score: float = None) -> Dict[str, Any]:
        """Run climate opportunity analysis."""
        logger.info("🔍 Starting climate opportunity analysis...")
        
        try:
            # Create opportunity time series
            timeseries = self.opportunity_analyzer.create_opportunity_timeseries(
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                min_score=min_score
            )
            
            if timeseries.empty:
                logger.warning("No opportunity data found")
                return {}
            
            # Store results
            self.opportunity_results = {
                'timeseries': timeseries,
                'analysis_type': 'climate_opportunities',
                'total_mentions': timeseries['similarity_score_count'].sum(),
                'avg_score': timeseries['similarity_score_mean'].mean(),
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Export to CSV
            csv_path = self.export_opportunity_csv(timeseries)
            self.csv_files['opportunity'] = csv_path
            
            # Export to JSON
            self.export_opportunity_json(self.opportunity_results)
            
            logger.info(f"✅ Opportunity analysis: {self.opportunity_results['total_mentions']} mentions, "
                       f"avg score: {self.opportunity_results['avg_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed opportunity analysis: {str(e)}")
            logger.error(traceback.format_exc())
        
        return self.opportunity_results
    
    # Export methods
    def export_policy_csv(self, policy: str, timeseries: pd.DataFrame) -> Path:
        """Export policy time series to CSV file."""
        csv_path = self.output_dir / "csv_exports" / f"{policy}_timeseries.csv"
        timeseries.to_csv(csv_path, index=False)
        logger.info(f"💾 Exported {policy} time series: {csv_path}")
        return csv_path
    
    def export_risk_csv(self, timeseries: pd.DataFrame) -> Path:
        """Export risk time series to CSV file."""
        csv_path = self.output_dir / "csv_exports" / "climate_risks_timeseries.csv"
        timeseries.to_csv(csv_path, index=False)
        logger.info(f"💾 Exported risk time series: {csv_path}")
        return csv_path
    
    def export_opportunity_csv(self, timeseries: pd.DataFrame) -> Path:
        """Export opportunity time series to CSV file."""
        csv_path = self.output_dir / "csv_exports" / "climate_opportunities_timeseries.csv"
        timeseries.to_csv(csv_path, index=False)
        logger.info(f"💾 Exported opportunity time series: {csv_path}")
        return csv_path
    
    def export_policy_json(self, policy: str, analysis_result: Dict[str, Any]) -> Path:
        """Export complete policy analysis to JSON file."""
        import json
        
        json_path = self.output_dir / "analysis_results" / "policies" / f"{policy}_complete_analysis.json"
        
        # Convert DataFrames to records for JSON serialization
        json_data = analysis_result.copy()
        json_data['timeseries'] = analysis_result['timeseries'].to_dict('records')
        json_data['firm_analysis'] = analysis_result['firm_analysis'].to_dict('records')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"💾 Exported {policy} complete analysis: {json_path}")
        return json_path
    
    def export_risk_json(self, analysis_result: Dict[str, Any]) -> Path:
        """Export risk analysis to JSON file."""
        import json
        
        json_path = self.output_dir / "analysis_results" / "risk" / "climate_risks_analysis.json"
        
        json_data = analysis_result.copy()
        json_data['timeseries'] = analysis_result['timeseries'].to_dict('records')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"💾 Exported risk analysis: {json_path}")
        return json_path
    
    def export_opportunity_json(self, analysis_result: Dict[str, Any]) -> Path:
        """Export opportunity analysis to JSON file."""
        import json
        
        json_path = self.output_dir / "analysis_results" / "opportunity" / "climate_opportunities_analysis.json"
        
        json_data = analysis_result.copy()
        json_data['timeseries'] = analysis_result['timeseries'].to_dict('records')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"💾 Exported opportunity analysis: {json_path}")
        return json_path
    
    def create_policy_visualizations(self, **kwargs) -> Dict[str, List[Path]]:
        """Create policy visualizations (existing functionality)."""
        logger.info("🎨 Creating policy visualizations...")
        
        viz_frequency = kwargs.get('frequency', 'quarter')
        viz_paths = {}
        
        for policy, csv_path in self.csv_files['policies'].items():
            try:
                logger.info(f"🖼️ Creating plots for {policy}...")
                
                policy_viz_paths = []
                
                # Main time series plot
                ts_path = self.policy_visualizer.plot_policy_timeseries(
                    data_input=csv_path,
                    policy_name=policy,
                    freq=viz_frequency,
                    save_path=self.output_dir / "visualizations" / "policies" / f"{policy}_timeseries.png"
                )
                policy_viz_paths.append(ts_path)
                
                # Annual summary plot
                annual_path = self.policy_visualizer.plot_annual_summary(
                    data_input=csv_path,
                    policy_name=policy,
                    save_path=self.output_dir / "visualizations" / "policies" / f"{policy}_annual.png"
                )
                if annual_path:
                    policy_viz_paths.append(annual_path)
                
                viz_paths[policy] = policy_viz_paths
                logger.info(f"✅ {policy}: {len(policy_viz_paths)} plots created")
                
            except Exception as e:
                logger.error(f"❌ Failed to create visualizations for {policy}: {str(e)}")
                continue
        
        return viz_paths
    
    def create_risk_opportunity_visualizations(self, **kwargs) -> List[Path]:
        """Create risk and opportunity visualizations."""
        logger.info("🎨 Creating risk and opportunity visualizations...")
        
        viz_paths = []
        viz_frequency = kwargs.get('frequency', 'day')
        
        try:
            # Individual plots
            if self.csv_files['risk']:
                risk_path = self.risk_opp_visualizer.plot_risk_timeseries(
                    data_input=self.csv_files['risk'],
                    freq=viz_frequency,
                    save_path=self.output_dir / "visualizations" / "risk_opportunity" / "climate_risks_timeseries.png"
                )
                viz_paths.append(risk_path)
            
            if self.csv_files['opportunity']:
                opp_path = self.risk_opp_visualizer.plot_opportunity_timeseries(
                    data_input=self.csv_files['opportunity'],
                    freq=viz_frequency,
                    save_path=self.output_dir / "visualizations" / "risk_opportunity" / "climate_opportunities_timeseries.png"
                )
                viz_paths.append(opp_path)
            
            # Comparative plots if both exist
            if self.csv_files['risk'] and self.csv_files['opportunity']:
                # Line comparison
                compare_path = self.risk_opp_visualizer.plot_risk_opportunity_comparison(
                    risk_data=self.csv_files['risk'],
                    opportunity_data=self.csv_files['opportunity'],
                    freq=viz_frequency,
                    save_path=self.output_dir / "visualizations" / "risk_opportunity" / "risk_opportunity_comparison.png"
                )
                viz_paths.append(compare_path)
                
                # Stacked plot
                stacked_path = self.risk_opp_visualizer.plot_stacked_risk_opportunity(
                    risk_data=self.csv_files['risk'],
                    opportunity_data=self.csv_files['opportunity'],
                    freq=viz_frequency,
                    save_path=self.output_dir / "visualizations" / "risk_opportunity" / "risk_opportunity_stacked.png"
                )
                viz_paths.append(stacked_path)
                
                # Annual summary
                annual_path = self.risk_opp_visualizer.plot_annual_summary(
                    risk_data=self.csv_files['risk'],
                    opportunity_data=self.csv_files['opportunity'],
                    save_path=self.output_dir / "visualizations" / "risk_opportunity" / "risk_opportunity_annual.png"
                )
                if annual_path:
                    viz_paths.append(annual_path)
            
            logger.info(f"✅ Created {len(viz_paths)} risk/opportunity plots")
            
        except Exception as e:
            logger.error(f"❌ Failed to create risk/opportunity visualizations: {str(e)}")
            logger.error(traceback.format_exc())
        
        return viz_paths
    
    def create_comparative_visualizations(self, **kwargs) -> List[Path]:
        """Create comparative visualizations across all analyses."""
        logger.info("🎨 Creating comparative visualizations...")
        
        comparative_paths = []
        
        # Policy comparisons (existing)
        if len(self.csv_files['policies']) >= 2:
            try:
                comparison_path = self.policy_visualizer.plot_policy_comparison(
                    data_inputs=list(self.csv_files['policies'].values()),
                    policy_names=list(self.csv_files['policies'].keys()),
                    save_path=self.output_dir / "visualizations" / "comparisons" / "all_policies_comparison.png"
                )
                comparative_paths.append(comparison_path)
                
                summary_path = self.policy_visualizer.plot_summary_comparison(
                    data_inputs=list(self.csv_files['policies'].values()),
                    policy_names=list(self.csv_files['policies'].keys()),
                    save_path=self.output_dir / "visualizations" / "comparisons" / "policies_summary.png"
                )
                comparative_paths.append(summary_path)
                
            except Exception as e:
                logger.error(f"❌ Failed to create policy comparisons: {str(e)}")
        
        logger.info(f"✅ Created {len(comparative_paths)} comparison plots")
        return comparative_paths
    
    def generate_summary_report(self) -> Path:
        """Generate a comprehensive summary report of all analyses."""
        logger.info("📋 Generating comprehensive summary report...")
        
        report_path = self.output_dir / "reports" / f"climate_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Climate Analysis Summary Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Index Path:** {self.index_path}\n\n")
            
            # Policy Analysis Section
            if self.policy_results:
                f.write("## Policy Analysis Results\n\n")
                f.write(f"**Total Policies Analyzed:** {len(self.policy_results)}\n\n")
                
                for policy, results in self.policy_results.items():
                    stats = results['summary_statistics']
                    policy_name = results['policy_name'].replace('_', ' ').title()
                    
                    f.write(f"### {policy_name}\n\n")
                    f.write(f"- **Total Mentions:** {stats['total_mentions']}\n")
                    f.write(f"- **Average Attention Score:** {stats['attention_statistics']['mean']:.4f}\n")
                    f.write(f"- **Periods with Mentions:** {stats['mention_statistics']['periods_with_mentions']}\n")
                    f.write(f"- **Peak Mentions:** {stats['mention_statistics']['max_mentions_period']}\n\n")
            
            # Risk Analysis Section
            if self.risk_results:
                f.write("## Risk Analysis Results\n\n")
                f.write(f"- **Total Risk Mentions:** {self.risk_results['total_mentions']}\n")
                f.write(f"- **Average Risk Score:** {self.risk_results['avg_score']:.4f}\n\n")
            
            # Opportunity Analysis Section
            if self.opportunity_results:
                f.write("## Opportunity Analysis Results\n\n")
                f.write(f"- **Total Opportunity Mentions:** {self.opportunity_results['total_mentions']}\n")
                f.write(f"- **Average Opportunity Score:** {self.opportunity_results['avg_score']:.4f}\n\n")
            
            # Summary Statistics
            total_mentions = 0
            if self.policy_results:
                total_mentions += sum(r['summary_statistics']['total_mentions'] for r in self.policy_results.values())
            if self.risk_results:
                total_mentions += self.risk_results['total_mentions']
            if self.opportunity_results:
                total_mentions += self.opportunity_results['total_mentions']
            
            f.write("## Overall Summary\n\n")
            f.write(f"- **Total Climate Mentions Across All Analyses:** {total_mentions:,}\n")
            f.write(f"- **Analysis Types Completed:** ")
            completed = []
            if self.policy_results: completed.append("Policy")
            if self.risk_results: completed.append("Risk")
            if self.opportunity_results: completed.append("Opportunity")
            f.write(", ".join(completed) + "\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("### CSV Exports\n")
            for analysis_type, files in self.csv_files.items():
                if analysis_type == 'policies' and files:
                    for policy, csv_path in files.items():
                        f.write(f"- [{policy}_timeseries.csv]({csv_path.relative_to(self.output_dir)})\n")
                elif files:
                    f.write(f"- [{analysis_type}_timeseries.csv]({files.relative_to(self.output_dir)})\n")
            
            f.write("\n### Visualizations\n")
            f.write("- Policy plots in `visualizations/policies/`\n")
            f.write("- Risk & opportunity plots in `visualizations/risk_opportunity/`\n")
            f.write("- Comparative plots in `visualizations/comparisons/`\n")
            f.write("- Analysis results in `analysis_results/`\n")
        
        logger.info(f"📋 Summary report saved: {report_path}")
        return report_path
     
    def run_complete_pipeline(self, 
                         analysis_types: List[str] = None,
                         policies: List[str] = None,
                         run_sensitivity: bool = False,
                         sensitivity_config: Dict[str, Any] = None,
                         **analysis_kwargs) -> Dict[str, Any]:
        """
        Run the complete climate analysis pipeline.
        
        Args:
            analysis_types: Types of analysis to run ['policy', 'risk', 'opportunity']
            policies: List of policies to analyze (for policy analysis)
            run_sensitivity: Whether to run threshold sensitivity analysis
            sensitivity_config: Configuration for sensitivity analysis
            **analysis_kwargs: Additional arguments for analyses
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        logger.info("🚀 Starting complete climate analysis pipeline...")
        
        if analysis_types is None:
            analysis_types = ['policy', 'risk', 'opportunity']
        
        if sensitivity_config is None:
            sensitivity_config = {
                'threshold_range': (0.3, 0.6),
                'num_points': 20,
                'boundary_samples': 100,
                'roc_sample_size': 500
            }
        
        try:
            # Step 1: Run analyses (existing code)
            logger.info("📊 Step 1: Running analyses...")
            
            if 'policy' in analysis_types:
                logger.info("Running policy analysis...")
                self.run_policy_analysis(policies, **analysis_kwargs)
            
            if 'risk' in analysis_types:
                logger.info("Running risk analysis...")
                self.run_risk_analysis(**analysis_kwargs)
            
            if 'opportunity' in analysis_types:
                logger.info("Running opportunity analysis...")
                self.run_opportunity_analysis(**analysis_kwargs)
            
            # Step 1.5: Run sensitivity analysis if requested
            sensitivity_results = {}
            if run_sensitivity and 'policy' in analysis_types:
                logger.info("🔍 Step 1.5: Running threshold sensitivity analysis...")
                sensitivity_results = self.run_sensitivity_analysis(
                    policies=policies,
                    **sensitivity_config,
                    **{k: v for k, v in analysis_kwargs.items() 
                    if k in ['start_date', 'end_date', 'companies']}
                )
            
            # Step 2: Create visualizations (existing code)
            logger.info("🎨 Step 2: Creating visualizations...")
            
            policy_viz = []
            if 'policy' in analysis_types and self.csv_files['policies']:
                policy_viz = self.create_policy_visualizations(**analysis_kwargs)
            
            risk_opp_viz = []
            if ('risk' in analysis_types or 'opportunity' in analysis_types):
                risk_opp_viz = self.create_risk_opportunity_visualizations(**analysis_kwargs)
            
            comparative_viz = self.create_comparative_visualizations(**analysis_kwargs)
            
            # Step 3: Generate summary report (existing code)
            logger.info("📋 Step 3: Generating summary report...")
            report_path = self.generate_summary_report()
            
            # Calculate execution time
            execution_time = datetime.now() - start_time
            
            # Calculate totals (existing code)
            total_mentions = 0
            if self.policy_results:
                total_mentions += sum(r['summary_statistics']['total_mentions'] for r in self.policy_results.values())
            if self.risk_results:
                total_mentions += self.risk_results['total_mentions']
            if self.opportunity_results:
                total_mentions += self.opportunity_results['total_mentions']
            
            pipeline_results = {
                'policy_results': self.policy_results,
                'risk_results': self.risk_results,
                'opportunity_results': self.opportunity_results,
                'sensitivity_results': sensitivity_results,  # Add this line
                'csv_files': self.csv_files,
                'policy_visualizations': policy_viz,
                'risk_opportunity_visualizations': risk_opp_viz,
                'comparative_visualizations': comparative_viz,
                'summary_report': report_path,
                'execution_time': execution_time,
                'analysis_types_completed': analysis_types,
                'total_mentions': total_mentions
            }
            
            logger.info("🎉 PIPELINE COMPLETE!")
            logger.info(f"⏱️ Execution time: {execution_time}")
            logger.info(f"📊 Analysis types: {', '.join(analysis_types)}")
            if sensitivity_results:
                logger.info(f"🔍 Sensitivity analysis completed for {len(sensitivity_results)} policies")
            logger.info(f"💬 Total mentions found: {total_mentions:,}")
            logger.info(f"📁 All outputs saved to: {self.output_dir}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Run complete climate analysis pipeline (Policy + Risk + Opportunity)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all analyses with sensitivity analysis
    python scripts/run_all.py --run-sensitivity
    
    # Run sensitivity analysis with custom range
    python scripts/run_all.py --run-sensitivity --sensitivity-range 0.2,0.8 --sensitivity-points 30
    
    # Run only policy analysis with sensitivity
    python scripts/run_all.py --analysis-types policy --run-sensitivity
        """
    )
    # Add these new arguments
    parser.add_argument(
        '--run-sensitivity',
        action='store_true',
        help='Run threshold sensitivity analysis for policies'
    )
    
    parser.add_argument(
    '--roc-boundary-samples',
    type=int,
    default=100,
    help='Number of boundary samples per threshold for ROC validation (default: 100)'
)

    parser.add_argument(
        '--roc-sample-size',
        type=int,
        default=500,
        help='Total sample size for ROC validation (default: 500)'
    )

    parser.add_argument(
        '--sensitivity-range',
        type=str,
        default='0.3,0.7',
        help='Threshold range for sensitivity analysis as min,max (default: 0.3,0.7)'
    )
    
    parser.add_argument(
        '--sensitivity-points',
        type=int,
        default=20,
        help='Number of threshold points to test (default: 20)'
    )

    parser.add_argument(
        '--index-path',
        type=str,
        default='data/semantic_indexes/combined',
        help='Path to semantic index directory (default: data/semantic_indexes/combined)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/complete_climate_analysis',
        help='Output directory for all results (default: outputs/complete_climate_analysis)'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        help='Path to configuration file (optional)'
    )
    
    parser.add_argument(
        '--analysis-types',
        type=str,
        default='policy,risk,opportunity',
        help='Comma-separated list of analysis types: policy,risk,opportunity (default: all)'
    )
    
    parser.add_argument(
        '--policies',
        type=str,
        help='Comma-separated list of policies for policy analysis (default: all policies)'
    )
    
    parser.add_argument(
        '--frequency',
        choices=['day', 'quarter', 'year', 'month'],
        default='day',
        help='Time frequency for visualization (default: day)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='Minimum similarity threshold (default: from config or analyzer defaults)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for analysis (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    sensitivity_config = None
    if args.run_sensitivity:
        try:
            range_parts = args.sensitivity_range.split(',')
            sensitivity_config = {
                'threshold_range': (float(range_parts[0]), float(range_parts[1])),
                'num_points': args.sensitivity_points,
                'boundary_samples': args.roc_boundary_samples,
                'roc_sample_size': args.roc_sample_size
            }
        except (ValueError, IndexError):
            logger.error("❌ Invalid sensitivity range format. Use: min,max (e.g., 0.3,0.7)")
            sys.exit(1)
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse analysis types
    analysis_types = [t.strip() for t in args.analysis_types.split(',')]
    valid_types = ['policy', 'risk', 'opportunity']
    
    for analysis_type in analysis_types:
        if analysis_type not in valid_types:
            logger.error(f"❌ Invalid analysis type: {analysis_type}. Valid types: {valid_types}")
            sys.exit(1)
    
    # Parse policies list
    policies = None
    if args.policies:
        policies = [p.strip() for p in args.policies.split(',')]
    
    # Validate index path
    index_path = Path(args.index_path)
    if not index_path.exists():
        logger.error(f"❌ Index path does not exist: {index_path}")
        sys.exit(1)
    
    try:
        logger.info("🚀 Starting master climate analysis script...")
        logger.info(f"Analysis types: {', '.join(analysis_types)}")
        
        # Initialize runner
        runner = MasterClimateRunner(
            index_path=index_path,
            output_dir=args.output_dir,
            config_path=args.config_path
        )
        
        # Prepare analysis arguments
        analysis_kwargs = {
            'frequency': args.frequency,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'min_score': args.threshold
        }
        
        # Remove None values
        analysis_kwargs = {k: v for k, v in analysis_kwargs.items() if v is not None}
        
        # Run complete pipeline
        # Run complete pipeline with sensitivity analysis
        results = runner.run_complete_pipeline(
            analysis_types=analysis_types,
            policies=policies,
            run_sensitivity=args.run_sensitivity,
            sensitivity_config=sensitivity_config,
            **analysis_kwargs
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE CLIMATE ANALYSIS COMPLETE!")
        print("="*80)
        print(f"📊 Analysis types completed: {', '.join(results['analysis_types_completed'])}")
        
        if results['policy_results']:
            print(f"📋 Policies analyzed: {len(results['policy_results'])}")
        if results['risk_results']:
            print(f"⚠️  Risk analysis: {results['risk_results']['total_mentions']:,} mentions")
        if results['opportunity_results']:
            print(f"🌱 Opportunity analysis: {results['opportunity_results']['total_mentions']:,} mentions")
        
        print(f"💬 Total climate mentions: {results['total_mentions']:,}")
        print(f"⏱️  Execution time: {results['execution_time']}")
        print(f"📁 Results saved to: {Path(args.output_dir).absolute()}")
        print(f"📋 Summary report: {results['summary_report'].name}")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()