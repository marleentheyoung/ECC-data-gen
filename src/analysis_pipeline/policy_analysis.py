#!/usr/bin/env python3
"""
Climate Policy Attention Analysis

Creates unnormalized time series of climate policy attention that can be
normalized later using total sentences per period data.

Usage:
    python policy_analysis.py --policy paris_agreement
    python policy_analysis.py --policy us_ira --threshold 0.40 --freq Y
    python policy_analysis.py --policy eu_green_deal --market STOXX600

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# Add the path to find semantic_searcher
sys.path.append(str(Path(__file__).parent))

try:
    from base_searcher import SemanticClimateSearcher
except ImportError:
    print("❌ Could not import SemanticClimateSearcher")
    print("Make sure semantic_searcher.py is in the same directory")
    sys.exit(1)


def create_policy_timeseries(index_path: str, 
                            policy: str = 'paris_agreement',
                            threshold: float = 0.45,
                            start_date: str = None,
                            end_date: str = None,
                            freq: str = 'Q') -> pd.DataFrame:
    """Create climate policy attention time series."""
    
    policy_display = policy.replace('_', ' ').title()
    print(f"🔍 Analyzing {policy_display} attention with threshold {threshold}")
    print(f"📊 Using index: {index_path}")
    
    # Load searcher
    searcher = SemanticClimateSearcher(index_path)
    
    # Get statistics
    stats = searcher.get_statistics()
    print(f"📈 Index contains {stats['total_snippets']:,} snippets")
    print(f"🏢 Covering {stats['unique_companies']} unique companies")
    print(f"📅 Date range: {stats['year_range']}")
    print()
    
    # Create policy time series
    pa_ts = searcher.create_policy_attention_timeseries(
        policy_type=policy,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        min_score=threshold
    )
    
    if len(pa_ts) == 0:
        print(f"❌ No {policy_display} mentions found with threshold {threshold}")
        print("💡 Try lowering the threshold (e.g., --threshold 0.35)")
        return pd.DataFrame()
    
    print(f"✅ Found {policy_display} mentions in {len(pa_ts)} time periods")
    
    # Add some useful columns
    pa_ts['period_label'] = pa_ts['date'].dt.strftime('%Y-Q%q' if freq == 'Q' else '%Y')
    pa_ts['cumulative_mentions'] = pa_ts['count_sum'].cumsum()
    
    # Summary statistics
    total_mentions = pa_ts['count_sum'].sum()
    avg_mentions_per_period = pa_ts['count_sum'].mean()
    peak_period = pa_ts.loc[pa_ts['count_sum'].idxmax()]
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Total mentions: {total_mentions}")
    print(f"   Average per period: {avg_mentions_per_period:.1f}")
    print(f"   Peak period: {peak_period['period_label']} ({int(peak_period['count_sum'])} mentions)")
    print(f"   Active periods: {(pa_ts['count_sum'] > 0).sum()}/{len(pa_ts)}")
    
    return pa_ts


def create_comparison_analysis(index_path: str, threshold: float = 0.45) -> pd.DataFrame:
    """Compare all climate policies."""
    
    print("\n🔄 Creating comparative policy analysis...")
    
    searcher = SemanticClimateSearcher(index_path)
    
    policies = ['paris_agreement', 'eu_green_deal', 'us_ira', 'cop_meetings']
    comparison_data = []
    
    for policy in policies:
        try:
            ts = searcher.create_policy_attention_timeseries(
                policy_type=policy,
                freq='Q',
                min_score=threshold
            )
            
            if len(ts) > 0:
                comparison_data.append({
                    'policy': policy.replace('_', ' ').title(),
                    'total_mentions': ts['count_sum'].sum(),
                    'active_quarters': (ts['count_sum'] > 0).sum(),
                    'peak_mentions': ts['count_sum'].max(),
                    'avg_similarity': ts['similarity_score_mean'].mean()
                })
                print(f"   ✅ {policy}: {ts['count_sum'].sum()} total mentions")
            else:
                print(f"   ❌ {policy}: No mentions found")
                
        except Exception as e:
            print(f"   ⚠️ {policy}: Error - {e}")
    
    return pd.DataFrame(comparison_data)


def setup_jof_style():
    """Set up Journal of Finance publication style."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 13,
        'axes.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.labelsize': 13,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'legend.frameon': False
    })


def create_visualizations(pa_ts: pd.DataFrame, policy: str, output_dir: Path = Path("outputs")):
    """Create JoF-style visualizations for climate policy attention."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_jof_style()
    
    policy_display = policy.replace('_', ' ').title()
    
    # Define policy-specific colors and events
    policy_configs = {
        'paris_agreement': {
            'color': '#1f4e79',
            'events': {
                '2015-12-12': 'Paris Agreement',
                '2017-06-01': 'US Withdrawal', 
                '2021-01-20': 'US Re-entry'
            }
        },
        'us_ira': {
            'color': '#c5504b',
            'events': {
                '2022-08-16': 'IRA Signed',
            }
        },
        'eu_green_deal': {
            'color': '#70ad47',
            'events': {
                '2019-12-11': 'EU Green Deal',
            }
        },
        'cop_meetings': {
            'color': '#7030a0',
            'events': {
                '2021-11-01': 'COP26 Glasgow',
                '2022-11-06': 'COP27 Egypt',
                '2023-11-30': 'COP28 Dubai'
            }
        }
    }
    
    config = policy_configs.get(policy, {'color': '#1f4e79', 'events': {}})
    
    # 1. Main time series plot (JoF style)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main time series line
    ax.plot(pa_ts['date'], pa_ts['count_sum'], 
            color=config['color'], linewidth=2.5, marker='o', markersize=3)
    
    # Add key events as vertical lines
    key_events = config['events']
    
    y_max = pa_ts['count_sum'].max()
    for i, (date_str, label) in enumerate(key_events.items()):
        try:
            event_date = pd.to_datetime(date_str)
            if pa_ts['date'].min() <= event_date <= pa_ts['date'].max():
                ax.axvline(event_date, color='#c5504b', alpha=0.7, 
                          linestyle='--', linewidth=2)
                
                # Alternate label heights to avoid overlap
                label_height = y_max * (0.85 + 0.1 * (i % 2))
                ax.text(event_date, label_height, label, 
                       rotation=90, ha='right', va='top', fontsize=13, 
                       color='#c5504b', alpha=0.9)
        except:
            continue
    
    # JoF style formatting
    ax.set_xlabel('Year', fontsize=15, color='black')
    ax.set_ylabel(f'{policy_display} Mentions', fontsize=14, color='black')
    
    # Format x-axis to show years
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Remove top and right spines (JoF style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Add subtle grid lines
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Tick styling
    ax.tick_params(colors='black', which='both')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Save main plot
    plot_path = output_dir / f'{policy}_timeseries_jof.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"📊 JoF-style visualization saved: {plot_path}")
    
    # 2. Cumulative plot (separate, JoF style)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(pa_ts['date'], pa_ts['cumulative_mentions'], 
            color=config['color'], linewidth=2.5, marker='s', markersize=3)
    
    # JoF style formatting
    ax.set_xlabel('Year', fontsize=14, color='black')
    ax.set_ylabel(f'Cumulative {policy_display} Mentions', fontsize=14, color='black')
    
    # Format axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    
    # JoF styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.tick_params(colors='black', which='both')
    
    plt.tight_layout()
    
    # Save cumulative plot
    cumulative_path = output_dir / f'{policy}_cumulative_jof.png'
    plt.savefig(cumulative_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"📊 JoF-style cumulative plot saved: {cumulative_path}")
    
    # 3. Annual aggregation plot (more suitable for JoF)
    if len(pa_ts) > 8:  # Only if we have sufficient data
        # Aggregate by year for cleaner visualization
        pa_ts['year'] = pa_ts['date'].dt.year
        yearly_data = pa_ts.groupby('year').agg({
            'count_sum': 'sum',
            'similarity_score_mean': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot with JoF styling
        bars = ax.bar(yearly_data['year'], yearly_data['count_sum'], 
                     color=config['color'], alpha=0.8, width=0.7)
        
        # Add value labels on bars for key years
        for bar in bars:
            height = bar.get_height()
            if height > yearly_data['count_sum'].mean():  # Only label above-average years
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, color='black')
        
        # JoF formatting
        ax.set_xlabel('Year', fontsize=14, color='black')
        ax.set_ylabel(f'Annual {policy_display} Mentions', fontsize=14, color='black')
        
        # JoF styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        
        ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray', axis='y')
        ax.set_axisbelow(True)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.tick_params(colors='black', which='both')
        
        # Format x-axis to show all years
        ax.set_xticks(yearly_data['year'])
        ax.set_xticklabels(yearly_data['year'], rotation=45)
        
        plt.tight_layout()
        
        # Save annual plot
        annual_path = output_dir / f'{policy}_annual_jof.png'
        plt.savefig(annual_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"📊 JoF-style annual plot saved: {annual_path}")


def create_comparative_policy_plots(index_path: str, threshold: float = 0.45, 
                                   output_dir: Path = Path("outputs")):
    """Create JoF-style comparative plots for multiple climate policies."""
    
    setup_jof_style()
    searcher = SemanticClimateSearcher(index_path)
    
    # Define policies and their display names/colors
    policies_info = {
        'paris_agreement': {'name': 'Paris Agreement', 'color': '#1f4e79'},
        'eu_green_deal': {'name': 'EU Green Deal', 'color': '#70ad47'},
        'us_ira': {'name': 'US IRA', 'color': '#c5504b'},
        'cop_meetings': {'name': 'COP Meetings', 'color': '#7030a0'}
    }
    
    # Collect data for all policies
    policy_data = {}
    for policy_key, info in policies_info.items():
        try:
            ts = searcher.create_policy_attention_timeseries(
                policy_type=policy_key,
                freq='Q',
                min_score=threshold
            )
            if len(ts) > 0:
                policy_data[policy_key] = {
                    'data': ts,
                    'name': info['name'],
                    'color': info['color']
                }
                print(f"✅ {info['name']}: {ts['count_sum'].sum()} total mentions")
        except Exception as e:
            print(f"⚠️ {info['name']}: Error - {e}")
    
    if not policy_data:
        print("❌ No policy data found for comparison")
        return
    
    # 1. Multi-line time series comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for policy_key, info in policy_data.items():
        ts = info['data']
        ax.plot(ts['date'], ts['count_sum'], 
               color=info['color'], linewidth=2.5, marker='o', markersize=3,
               label=info['name'])
    
    # Add key climate events
    key_events = {
        '2015-12-12': 'Paris Agreement',
        '2019-12-11': 'EU Green Deal',
        '2022-08-16': 'US IRA'
    }
    
    # Find y-axis range for event labels
    all_values = []
    for info in policy_data.values():
        all_values.extend(info['data']['count_sum'].values)
    
    if all_values:
        y_max = max(all_values)
        for i, (date_str, label) in enumerate(key_events.items()):
            try:
                event_date = pd.to_datetime(date_str)
                # Check if any policy has data in this range
                has_data = any(info['data']['date'].min() <= event_date <= info['data']['date'].max() 
                              for info in policy_data.values())
                if has_data:
                    ax.axvline(event_date, color='gray', alpha=0.6, 
                              linestyle=':', linewidth=1.5)
                    # Alternate label heights
                    label_height = y_max * (0.85 + 0.1 * (i % 2))
                    ax.text(event_date, label_height, label, 
                           rotation=90, ha='right', va='top', fontsize=9, 
                           color='gray', alpha=0.8)
            except:
                continue
    
    # JoF formatting
    ax.set_xlabel('Year', fontsize=14, color='black')
    ax.set_ylabel('Policy Mentions in Earnings Calls', fontsize=14, color='black')
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    
    # JoF styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False, fontsize=12)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.tick_params(colors='black', which='both')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = output_dir / 'climate_policies_comparison_jof.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"📊 JoF-style policy comparison saved: {comparison_path}")
    
    # 2. Summary bar chart (total mentions by policy)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    policy_names = [info['name'] for info in policy_data.values()]
    total_mentions = [info['data']['count_sum'].sum() for info in policy_data.values()]
    colors = [info['color'] for info in policy_data.values()]
    
    bars = ax.bar(policy_names, total_mentions, color=colors, alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=13, color='black')
    
    # JoF formatting
    ax.set_ylabel('Total Mentions', fontsize=14, color='black')
    
    # JoF styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.3, color='gray', axis='y')
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.tick_params(colors='black', which='both')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save summary plot
    summary_path = output_dir / 'climate_policies_summary_jof.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"📊 JoF-style policy summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze climate policy attention in earnings calls',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic Paris Agreement analysis
    python policy_analysis.py --policy paris_agreement
    
    # US IRA with lower threshold
    python policy_analysis.py --policy us_ira --threshold 0.40
    
    # EU Green Deal with yearly frequency
    python policy_analysis.py --policy eu_green_deal --freq Y
    
    # Specific market and date range
    python policy_analysis.py --policy cop_meetings --market SP500 --start-date 2020-01-01
    
    # Include comparison with other policies
    python policy_analysis.py --policy paris_agreement --comparison
        """
    )
    
    parser.add_argument(
        '--index-path',
        type=str,
        default='data/semantic_indexes/combined',
        help='Path to semantic index (default: combined index)'
    )
    
    parser.add_argument(
        '--policy',
        choices=['paris_agreement', 'eu_green_deal', 'us_ira', 'cop_meetings'],
        default='paris_agreement',
        help='Policy to analyze (default: paris_agreement)'
    )
    
    parser.add_argument(
        '--market',
        choices=['SP500', 'STOXX600', 'combined'],
        default='combined',
        help='Market to analyze (default: combined)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.45,
        help='Similarity threshold (default: 0.45)'
    )
    
    parser.add_argument(
        '--freq',
        choices=['Q', 'Y', 'M'],
        default='Q',
        help='Time frequency (default: Q for quarterly)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Include comparison with other climate policies'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating plots (export data only)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Set index path based on market
    if args.market != 'combined':
        index_path = f"data/semantic_indexes/{args.market}"
    else:
        index_path = args.index_path
    
    policy_display = args.policy.replace('_', ' ').title()
    print(f"🌍 {policy_display} Attention Analysis")
    print("=" * 50)
    print(f"Policy: {policy_display}")
    print(f"Market: {args.market}")
    print(f"Threshold: {args.threshold}")
    print(f"Frequency: {args.freq}")
    print(f"Date range: {args.start_date or 'all'} to {args.end_date or 'all'}")
    print()
    
    try:
        # Main analysis
        pa_ts = create_policy_timeseries(
            index_path=index_path,
            policy=args.policy,
            threshold=args.threshold,
            start_date=args.start_date,
            end_date=args.end_date,
            freq=args.freq
        )
        
        if pa_ts.empty:
            return
        
        # Export main results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamic output filename
        output_filename = output_dir / f'{args.policy}_timeseries_{args.market}_{args.freq}.csv'        
        pa_ts.to_csv(output_filename, index=False)
        print(f"💾 Time series exported: {output_filename}")
        
        # Comparison analysis
        if args.comparison:
            comparison_df = create_comparison_analysis(index_path, args.threshold)
            if not comparison_df.empty:
                print(f"\n📊 Policy Comparison (threshold {args.threshold}):")
                print(comparison_df.to_string(index=False))
                
                comp_file = output_dir / f'policy_comparison_{args.market}.csv'
                comparison_df.to_csv(comp_file, index=False)
                print(f"💾 Comparison exported: {comp_file}")
        
        # Create visualizations
        if not args.no_plots:
            try:
                create_visualizations(pa_ts, args.policy, output_dir)
                
                # Add comparative plots if comparison was requested
                if args.comparison:
                    print("\n📊 Creating comparative policy visualizations...")
                    create_comparative_policy_plots(index_path, args.threshold, output_dir)
                    
            except Exception as e:
                print(f"⚠️ Could not create plots: {e}")
                print("💡 Install matplotlib and seaborn for visualizations")
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Results saved in: {output_dir}")
        
        print(f"\n💡 Next steps for normalization:")
        print(f"   1. Create total sentences per {args.freq.lower()} file")
        print(f"   2. Merge with this data: normalized = {policy_display}_mentions / total_sentences")
        print(f"   3. Use the 'count_sum' column as numerator")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've built the semantic indexes first:")
        print("   python scripts/5_semantic_search/build_semantic_index.py --all")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()