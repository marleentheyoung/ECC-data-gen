#!/usr/bin/env python3
"""
Script to run event study analyses on climate attention variables.

This script analyzes how firm-level climate attention changes around major
climate policy events, natural disasters, and other external shocks.

Usage:
    # Run all predefined events
    python scripts/5_run_event_studies.py --all
    
    # Run specific event
    python scripts/5_run_event_studies.py --event "Paris Agreement Adoption"
    
    # Run custom event
    python scripts/5_run_event_studies.py --custom-event "Custom Event" --event-date "2020-03-15"
    
    # Focus on specific variables
    python scripts/5_run_event_studies.py --all --variables normalized_climate_attention semantic_opportunities_exposure

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import LOGS_DIR
from src.analysis.climate_variables.event_analyzer import ClimateEventStudyAnalyzer


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / 'event_studies.log', mode='a')
        ]
    )


def check_panel_data(panel_path: Path) -> bool:
    """Check if climate panel data exists and is valid."""
    if not panel_path.exists():
        print(f"❌ Climate panel file not found: {panel_path}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(panel_path, nrows=10)  # Just check structure
        
        required_cols = ['ticker', 'year', 'quarter', 'date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns in panel data: {missing_cols}")
            return False
        
        print(f"✅ Panel data validated: {len(df.columns)} columns found")
        return True
        
    except Exception as e:
        print(f"❌ Error reading panel data: {e}")
        return False


def preview_available_variables(panel_path: Path) -> List[str]:
    """Preview available variables in the climate panel."""
    try:
        import pandas as pd
        df = pd.read_csv(panel_path, nrows=1)
        
        # Find climate-related variables
        climate_vars = []
        prefixes = ['semantic_', 'policy_', 'normalized_climate', 'climate_sentence']
        
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in prefixes):
                climate_vars.append(col)
        
        return sorted(climate_vars)
        
    except Exception as e:
        print(f"⚠️ Could not preview variables: {e}")
        return []


def print_predefined_events(analyzer: ClimateEventStudyAnalyzer):
    """Print available predefined events."""
    print("\n📅 Available Predefined Events:")
    print("=" * 50)
    
    events_by_category = {
        "International Agreements": [],
        "Policy Announcements": [],
        "COP Meetings": [],
        "Climate Reports/Disasters": [],
        "Corporate/Financial": []
    }
    
    for event_name, event_date in analyzer.predefined_events.items():
        if "Paris" in event_name or "Agreement" in event_name:
            events_by_category["International Agreements"].append((event_name, event_date))
        elif "EU" in event_name or "IRA" in event_name or "Taxonomy" in event_name:
            events_by_category["Policy Announcements"].append((event_name, event_date))
        elif "COP" in event_name:
            events_by_category["COP Meetings"].append((event_name, event_date))
        elif "IPCC" in event_name or "Hurricane" in event_name or "Bushfires" in event_name or "Heatwave" in event_name:
            events_by_category["Climate Reports/Disasters"].append((event_name, event_date))
        else:
            events_by_category["Corporate/Financial"].append((event_name, event_date))
    
    for category, events in events_by_category.items():
        if events:
            print(f"\n{category}:")
            for event_name, event_date in sorted(events, key=lambda x: x[1]):
                print(f"  • {event_name}: {event_date}")


def main():
    parser = argparse.ArgumentParser(
        description='Run event study analyses on climate attention variables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all predefined events
    python scripts/5_run_event_studies.py --all
    
    # Run specific predefined event
    python scripts/5_run_event_studies.py --event "Paris Agreement Adoption"
    
    # Run custom event
    python scripts/5_run_event_studies.py --custom-event "Fed Climate Stress Test" --event-date "2021-06-15"
    
    # Focus on specific variables
    python scripts/5_run_event_studies.py --all --variables normalized_climate_attention semantic_regulation_exposure
    
    # Use custom event window
    python scripts/5_run_event_studies.py --event "EU Green Deal Announcement" --event-window -2 2 --estimation-window -12 -3
    
    # List available events
    python scripts/5_run_event_studies.py --list-events
        """
    )
    
    # Event selection arguments
    event_group = parser.add_mutually_exclusive_group(required=False)
    event_group.add_argument(
        '--all',
        action='store_true',
        help='Run event studies for all predefined events'
    )
    
    event_group.add_argument(
        '--event',
        type=str,
        help='Run event study for specific predefined event (use --list-events to see options)'
    )
    
    event_group.add_argument(
        '--custom-event',
        type=str,
        help='Name for custom event (requires --event-date)'
    )
    
    parser.add_argument(
        '--event-date',
        type=str,
        help='Date for custom event in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--list-events',
        action='store_true',
        help='List all available predefined events and exit'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--variables',
        nargs='+',
        help='Specific variables to analyze (default: all climate variables)'
    )
    
    parser.add_argument(
        '--event-window',
        nargs=2,
        type=int,
        default=[-4, 4],
        metavar=('BEFORE', 'AFTER'),
        help='Event window in quarters (default: -4 4)'
    )
    
    parser.add_argument(
        '--estimation-window',
        nargs=2,
        type=int,
        default=[-20, -5],
        metavar=('START', 'END'),
        help='Estimation window in quarters before event (default: -20 -5)'
    )
    
    parser.add_argument(
        '--min-observations',
        type=int,
        default=5,
        help='Minimum observations required per firm (default: 5)'
    )
    
    # Data and output paths
    parser.add_argument(
        '--panel-data',
        type=Path,
        default=Path("outputs/climate_variables/semantic_climate_panel.csv"),
        help='Path to climate panel CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs/event_studies"),
        help='Output directory for event study results'
    )
    
    # Options
    parser.add_argument(
        '--no-regional-comparison',
        action='store_true',
        help='Skip regional (EU vs US) comparison'
    )
    
    parser.add_argument(
        '--create-plots',
        action='store_true',
        help='Create timeline plots for significant results'
    )
    
    parser.add_argument(
        '--preview-only',
        action='store_true',
        help='Only preview data and available variables'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate custom event arguments
    if args.custom_event and not args.event_date:
        parser.error("--custom-event requires --event-date")
    
    if args.event_date and not args.custom_event:
        parser.error("--event-date requires --custom-event")
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("📊 Climate Event Study Analysis")
    print("=" * 50)
    
    # Check if panel data exists
    if not check_panel_data(args.panel_data):
        return
    
    try:
        # Initialize analyzer
        print(f"🔧 Initializing event study analyzer...")
        analyzer = ClimateEventStudyAnalyzer(str(args.panel_data))
        
        # List events if requested
        if args.list_events:
            print_predefined_events(analyzer)
            return
        
        # Preview variables if requested
        if args.preview_only:
            available_vars = preview_available_variables(args.panel_data)
            
            print(f"\n📋 Available Climate Variables ({len(available_vars)}):")
            print("=" * 50)
            
            categories = {
                "Semantic Exposure": [v for v in available_vars if v.startswith('semantic_') and v.endswith('_exposure')],
                "Policy Attention": [v for v in available_vars if v.startswith('policy_')],
                "Sentiment": [v for v in available_vars if 'sentiment' in v],
                "Normalized Attention": [v for v in available_vars if 'normalized' in v or 'climate_sentence' in v],
                "Other": [v for v in available_vars if not any(cat in v for cat in ['semantic_', 'policy_', 'sentiment', 'normalized', 'climate_sentence'])]
            }
            
            for category, vars_list in categories.items():
                if vars_list:
                    print(f"\n{category}:")
                    for var in vars_list:
                        print(f"  • {var}")
            
            print(f"\n💡 Use --variables to specify which variables to analyze")
            return
        
        # Determine variables to analyze
        if args.variables:
            outcome_vars = args.variables
            print(f"🎯 Analyzing variables: {outcome_vars}")
        else:
            # Default variables
            outcome_vars = [
                'normalized_climate_attention',
                'semantic_opportunities_exposure',
                'semantic_regulation_exposure', 
                'semantic_physical_risk_exposure',
                'semantic_transition_risk_exposure',
                'semantic_disclosure_exposure'
            ]
            print(f"🎯 Using default variables ({len(outcome_vars)} variables)")
        
        # Validate variables exist
        available_vars = preview_available_variables(args.panel_data)
        missing_vars = [v for v in outcome_vars if v not in available_vars]
        if missing_vars:
            print(f"⚠️ Warning: Variables not found in data: {missing_vars}")
            outcome_vars = [v for v in outcome_vars if v in available_vars]
            print(f"🎯 Proceeding with {len(outcome_vars)} available variables")
        
        if not outcome_vars:
            print("❌ No valid variables to analyze")
            return
        
        # Configure analysis parameters
        event_window = tuple(args.event_window)
        estimation_window = tuple(args.estimation_window)
        compare_regions = not args.no_regional_comparison
        
        print(f"📅 Event window: {event_window} quarters")
        print(f"📈 Estimation window: {estimation_window} quarters")
        print(f"🌍 Regional comparison: {'Yes' if compare_regions else 'No'}")
        
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run event studies
        if args.all:
            # Run all predefined events
            print(f"\n🚀 Running event studies for all predefined events...")
            
            results = analyzer.run_multiple_events(
                outcome_vars=outcome_vars,
                event_window=event_window,
                estimation_window=estimation_window,
                min_observations=args.min_observations,
                compare_regions=compare_regions
            )
            
            # Save combined results
            output_file = args.output_dir / "all_events_results.json"
            analyzer.save_results(results, str(output_file))
            
            # Print summary
            print(f"\n📊 Event Study Summary:")
            print("=" * 50)
            
            cross_summary = results.get('cross_event_summary', {})
            if 'error' not in cross_summary:
                print(f"Total events analyzed: {cross_summary.get('total_events_analyzed', 0)}")
                
                most_responsive = cross_summary.get('most_responsive_variable')
                if most_responsive:
                    print(f"Most responsive variable: {most_responsive}")
                
                # Show significance by variable
                sig_by_var = cross_summary.get('variables_by_significance', {})
                if sig_by_var:
                    print(f"\nSignificant effects by variable:")
                    for var, count in list(sig_by_var.items())[:5]:  # Top 5
                        print(f"  • {var}: {count} significant events")
            
        elif args.event:
            # Run specific predefined event
            if args.event not in analyzer.predefined_events:
                print(f"❌ Event '{args.event}' not found in predefined events")
                print("Use --list-events to see available options")
                return
            
            event_date = analyzer.predefined_events[args.event]
            print(f"\n🚀 Running event study for: {args.event} ({event_date})")
            
            results = analyzer.run_event_study(
                event_date=event_date,
                event_name=args.event,
                outcome_vars=outcome_vars,
                event_window=event_window,
                estimation_window=estimation_window,
                min_observations=args.min_observations,
                compare_regions=compare_regions
            )
            
            # Save results
            safe_name = args.event.replace(" ", "_").replace("/", "_")
            output_file = args.output_dir / f"{safe_name}_results.json"
            analyzer.save_results(results, str(output_file))
            
            # Print results summary
            print_event_summary(results)
            
            # Create plots if requested
            if args.create_plots and 'error' not in results:
                create_event_plots(analyzer, results, args.output_dir, outcome_vars)
        
        elif args.custom_event:
            # Run custom event
            print(f"\n🚀 Running event study for: {args.custom_event} ({args.event_date})")
            
            results = analyzer.run_event_study(
                event_date=args.event_date,
                event_name=args.custom_event,
                outcome_vars=outcome_vars,
                event_window=event_window,
                estimation_window=estimation_window,
                min_observations=args.min_observations,
                compare_regions=compare_regions
            )
            
            # Save results
            safe_name = args.custom_event.replace(" ", "_").replace("/", "_")
            output_file = args.output_dir / f"{safe_name}_results.json"
            analyzer.save_results(results, str(output_file))
            
            # Print results summary
            print_event_summary(results)
            
            # Create plots if requested
            if args.create_plots and 'error' not in results:
                create_event_plots(analyzer, results, args.output_dir, outcome_vars)
        
        else:
            # No event specified
            print("❌ No event specified. Use --all, --event, or --custom-event")
            print("Use --list-events to see available predefined events")
            return
        
        print(f"\n✅ Event study analysis completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return


def print_event_summary(results: Dict):
    """Print summary of event study results."""
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n📊 Event Study Results: {results['event_name']}")
    print("=" * 60)
    print(f"Event date: {results['event_date']}")
    print(f"Total observations: {results['total_observations']:,}")
    print(f"Unique firms: {results['unique_firms']}")
    
    # Variable results
    var_results = results.get('variable_results', {})
    significant_vars = []
    
    print(f"\n🎯 Variable Results:")
    for var, var_data in var_results.items():
        if 'error' in var_data:
            print(f"  ❌ {var}: {var_data['error']}")
            continue
        
        mean_effect = var_data.get('mean_abnormal_level', 0)
        p_value = var_data.get('p_value', 1.0)
        is_significant = var_data.get('significant', False)
        n_firms = var_data.get('n_firms', 0)
        
        status = "✅ SIGNIFICANT" if is_significant else "  Not significant"
        print(f"  {status} | {var}")
        print(f"    Effect size: {mean_effect:+.4f}, p-value: {p_value:.3f}, firms: {n_firms}")
        
        if is_significant:
            significant_vars.append(var)
    
    # Regional comparison
    if 'regional_comparison' in results:
        print(f"\n🌍 Regional Comparison:")
        regional = results['regional_comparison']
        
        for var, reg_data in regional.items():
            if var in significant_vars:  # Only show for significant variables
                us_effect = reg_data.get('us_abnormal_level')
                eu_effect = reg_data.get('eu_abnormal_level')
                
                if us_effect is not None and eu_effect is not None:
                    print(f"  {var}:")
                    print(f"    US effect: {us_effect:+.4f}")
                    print(f"    EU effect: {eu_effect:+.4f}")
                    print(f"    Difference: {us_effect - eu_effect:+.4f}")
    
    # Summary
    summary = results.get('summary', {})
    if summary:
        print(f"\n📋 Summary:")
        print(f"  Variables tested: {summary.get('total_variables_tested', 0)}")
        print(f"  Significant variables: {summary.get('num_significant', 0)}")
        
        strongest = summary.get('strongest_effect')
        if strongest:
            print(f"  Strongest effect: {strongest['variable']} ({strongest['effect_size']:+.4f})")


def create_event_plots(analyzer, results: Dict, output_dir: Path, variables: List[str]):
    """Create plots for significant event study results."""
    print(f"\n📊 Creating plots for significant results...")
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    var_results = results.get('variable_results', {})
    event_name = results.get('event_name', 'event').replace(" ", "_")
    
    created_plots = 0
    
    for var in variables:
        if var in var_results and var_results[var].get('significant', False):
            try:
                plot_path = plots_dir / f"{event_name}_{var}_timeline.png"
                analyzer.plot_event_timeline(results, var, str(plot_path))
                created_plots += 1
            except Exception as e:
                print(f"⚠️ Could not create plot for {var}: {e}")
    
    if created_plots > 0:
        print(f"✅ Created {created_plots} timeline plots in: {plots_dir}")
    else:
        print("ℹ️ No significant results to plot")


if __name__ == "__main__":
    main()