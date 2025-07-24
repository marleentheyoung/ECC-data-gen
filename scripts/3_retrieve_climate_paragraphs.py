#!/usr/bin/env python3
"""
Script to retrieve climate-related paragraphs using semantic search.

This script uses multiple climate-related queries to find potentially climate-related
paragraphs with high recall, then saves them for further classification.

Usage:
    # Retrieve from SP500 index, get top 20% of paragraphs
    python scripts/retrieve_climate_paragraphs.py SP500 --top-percent 20
    
    # Retrieve from both indices with custom queries
    python scripts/retrieve_climate_paragraphs.py --all --top-percent 15 --min-score 0.3
    
    # Use custom climate query file
    python scripts/retrieve_climate_paragraphs.py STOXX600 --query-file my_climate_queries.txt

Author: Marleen de Jonge
Date: 2025
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import SUPPORTED_INDICES
from src.analysis.semantic_indexer import SemanticSearcher


# Default climate-related queries for semantic search
DEFAULT_CLIMATE_QUERIES = [
    # Direct climate terms
    "climate change",
    "climate risk",
    "climate impact",
    "global warming",
    
    # Environmental terms
    "environmental impact",
    "environmental sustainability", 
    "environmental regulations",
    "environmental compliance",
    
    # Carbon and emissions
    "carbon emissions",
    "carbon footprint", 
    "carbon neutral",
    "carbon reduction",
    "greenhouse gas emissions",
    "scope 1 2 3 emissions",
    
    # Energy and renewables
    "renewable energy",
    "clean energy",
    "energy transition",
    "solar wind energy",
    "decarbonization",
    "net zero",
    
    # Sustainability and ESG
    "sustainability initiatives",
    "ESG environmental social governance",
    "sustainable practices",
    "sustainability reporting",
    "sustainability goals",
    
    # Physical risks
    "extreme weather",
    "natural disasters",
    "flooding drought",
    "supply chain disruption weather",
    "climate adaptation",
    "climate resilience",
    
    # Transition risks
    "carbon pricing",
    "carbon tax",
    "emissions regulations",
    "climate regulations",
    "stranded assets",
    "transition costs",
    
    # Opportunities
    "green technology",
    "clean technology",
    "sustainable products",
    "green finance",
    "climate solutions",
    
    # Reporting and disclosure
    "TCFD Task Force Climate",
    "climate disclosure",
    "climate reporting",
    "CDP carbon disclosure",
    "science based targets"
]


def load_custom_queries(query_file: Path) -> List[str]:
    """Load custom queries from a text file."""
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                queries.append(line)
    return queries


def retrieve_climate_paragraphs(searcher: SemanticSearcher, 
                               climate_queries: List[str],
                               top_k_per_query: int = 10000,
                               min_score: float = 0.40) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve climate-related paragraphs using multiple queries.
    
    Args:
        searcher: SemanticSearcher instance
        climate_queries: List of climate-related queries
        top_k_per_query: Number of results per query (used as maximum, actual filtering by min_score)
        min_score: Minimum similarity score to include (default: 0.40)
        
    Returns:
        Dictionary mapping paragraph IDs to paragraph data with scores
    """
    all_results = {}  # paragraph_id -> {paragraph_data, max_score, matching_queries}
    
    logger = logging.getLogger(__name__)
    logger.info(f"Searching with {len(climate_queries)} climate queries...")
    
    for i, query in enumerate(climate_queries, 1):
        logger.info(f"Query {i}/{len(climate_queries)}: '{query}'")
        
        try:
            # Get more results to ensure we capture all paragraphs > min_score
            # Use larger top_k as a safety net, but filter by min_score
            search_k = max(top_k_per_query, 200)  # Ensure we get enough results to filter
            results = searcher.search(query, top_k=search_k)
            
            # Filter results by minimum score
            high_score_results = [r for r in results if r['score'] >= min_score]
            
            for result in high_score_results:
                score = result['score']
                # Score filtering already done above, but keeping this for safety
                if score < min_score:
                    continue
                
                # Create unique paragraph ID
                para_id = f"{result['filename']}_{result['section']}_{result['paragraph_id']}"
                
                if para_id not in all_results:
                    all_results[para_id] = {
                        'paragraph_data': result,
                        'max_score': score,
                        'matching_queries': [query],
                        'query_scores': {query: score}
                    }
                else:
                    # Update if this score is higher
                    if score > all_results[para_id]['max_score']:
                        all_results[para_id]['max_score'] = score
                    
                    # Add query to matching queries
                    all_results[para_id]['matching_queries'].append(query)
                    all_results[para_id]['query_scores'][query] = score
            
            logger.info(f"  Found {len(high_score_results)} results with score >= {min_score}")
            
        except Exception as e:
            logger.warning(f"Error with query '{query}': {e}")
            continue
    
    logger.info(f"✅ Retrieved {len(all_results)} unique climate candidate paragraphs")
    return all_results


def filter_top_percent(climate_results: Dict[str, Dict[str, Any]], 
                      top_percent: float) -> Dict[str, Dict[str, Any]]:
    """Filter to keep only top percentage by score."""
    if not climate_results:
        return {}
    
    # Sort by max score descending
    sorted_results = sorted(
        climate_results.items(), 
        key=lambda x: x[1]['max_score'], 
        reverse=True
    )
    
    # Calculate how many to keep
    num_to_keep = max(1, int(len(sorted_results) * (top_percent / 100)))
    
    # Keep top results
    filtered_results = dict(sorted_results[:num_to_keep])
    
    logger = logging.getLogger(__name__)
    logger.info(f"Filtered to top {top_percent}%: {len(filtered_results)} paragraphs")
    
    return filtered_results


def save_climate_candidates(climate_results: Dict[str, Dict[str, Any]], 
                          output_file: Path,
                          stock_index: str,
                          metadata: Dict[str, Any]):
    """Save climate candidate paragraphs to JSON file."""
    
    # Prepare data for saving
    save_data = {
        'metadata': {
            'stock_index': stock_index,
            'total_candidates': len(climate_results),
            'retrieval_timestamp': pd.Timestamp.now().isoformat(),
            **metadata
        },
        'climate_candidates': []
    }
    
    for para_id, result_data in climate_results.items():
        candidate = {
            'paragraph_id': para_id,
            'max_similarity_score': result_data['max_score'],
            'num_matching_queries': len(result_data['matching_queries']),
            'matching_queries': result_data['matching_queries'],
            'query_scores': result_data['query_scores'],
            **result_data['paragraph_data']  # Include all original paragraph metadata
        }
        save_data['climate_candidates'].append(candidate)
    
    # Sort by score for easier review
    save_data['climate_candidates'].sort(key=lambda x: x['max_similarity_score'], reverse=True)
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"💾 Saved {len(climate_results)} climate candidates to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Retrieve climate-related paragraphs using semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Get paragraphs from SP500 with similarity >= 0.40 (default)
    python scripts/retrieve_climate_paragraphs.py SP500 --top-percent 20
    
    # Search both indices with custom minimum score
    python scripts/retrieve_climate_paragraphs.py --all --min-score 0.50 --top-percent 15
    
    # Use custom queries with different threshold
    python scripts/retrieve_climate_paragraphs.py STOXX600 --query-file my_queries.txt --min-score 0.35
        """
    )
    
    parser.add_argument(
        'stock_index',
        nargs='?',
        choices=SUPPORTED_INDICES,
        help='Stock index to search (SP500 or STOXX600)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Search all available indices'
    )
    
    parser.add_argument(
        '--top-percent',
        type=float,
        default=20.0,
        help='Percentage of top-scoring paragraphs to keep (default: 20)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=200,
        help='Maximum number of results per query to consider (default: 200, actual filtering by min-score)'
    )
    
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.40,
        help='Minimum similarity score to include (default: 0.40)'
    )
    
    parser.add_argument(
        '--query-file',
        type=Path,
        help='File containing custom climate queries (one per line)'
    )
    
    parser.add_argument(
        '--index-folder',
        type=Path,
        default=Path("data/generated/semantic_indexes"),
        help='Folder containing semantic indexes'
    )
    
    parser.add_argument(
        '--output-folder',
        type=Path,
        default=Path("outputs/climate_candidates"),
        help='Output folder for climate candidates'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.stock_index:
        parser.error("Must specify either a stock index or --all")
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load queries
    if args.query_file:
        if not args.query_file.exists():
            print(f"❌ Query file not found: {args.query_file}")
            return
        climate_queries = load_custom_queries(args.query_file)
        logger.info(f"Loaded {len(climate_queries)} custom queries")
    else:
        climate_queries = DEFAULT_CLIMATE_QUERIES
        logger.info(f"Using {len(climate_queries)} default climate queries")
    
    if not args.index_folder.exists():
        print(f"❌ Index folder not found: {args.index_folder}")
        return
    
    print(f"🌍 Climate Paragraph Retrieval")
    print(f"Queries: {len(climate_queries)} climate concepts")
    print(f"Top-K per query: {args.top_k}")
    print(f"Top percent to keep: {args.top_percent}%")
    print(f"Minimum score: {args.min_score}")
    
    try:
        indices_to_process = SUPPORTED_INDICES if args.all else [args.stock_index]
        
        for stock_index in indices_to_process:
            print(f"\n{'='*60}")
            print(f"PROCESSING {stock_index}")
            print(f"{'='*60}")
            
            index_folder = args.index_folder / stock_index
            if not index_folder.exists():
                print(f"❌ Index not found for {stock_index}: {index_folder}")
                continue
            
            # Load searcher
            index_name = f"{stock_index.lower()}_paragraphs"
            try:
                searcher = SemanticSearcher(index_folder, index_name)
                stats = searcher.get_stats()
                print(f"📊 Index loaded: {stats['total_paragraphs']:,} paragraphs")
            except Exception as e:
                print(f"❌ Error loading index for {stock_index}: {e}")
                continue
            
            # Retrieve climate candidates
            climate_results = retrieve_climate_paragraphs(
                searcher=searcher,
                climate_queries=climate_queries,
                top_k_per_query=args.top_k,
                min_score=args.min_score
            )
            
            if not climate_results:
                print(f"⚠️ No climate candidates found for {stock_index}")
                continue
            
            # Filter to top percentage
            filtered_results = filter_top_percent(climate_results, args.top_percent)
            
            # Save results
            output_file = args.output_folder / f"{stock_index.lower()}_climate_candidates.json"
            
            metadata = {
                'total_paragraphs_in_index': stats['total_paragraphs'],
                'num_climate_queries': len(climate_queries),
                'top_k_per_query': args.top_k,
                'min_score': args.min_score,
                'top_percent_kept': args.top_percent,
                'climate_queries_used': climate_queries[:10]  # Save first 10 for reference
            }
            
            save_climate_candidates(filtered_results, output_file, stock_index, metadata)
            
            # Print summary
            print(f"✅ {stock_index} Results:")
            print(f"  Total candidates found: {len(climate_results):,}")
            print(f"  Top {args.top_percent}% kept: {len(filtered_results):,}")
            print(f"  Saved to: {output_file}")
            
            if filtered_results:
                scores = [r['max_score'] for r in filtered_results.values()]
                print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
    
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        print(f"❌ Error: {e}")
        return
    
    print(f"\n✅ Climate paragraph retrieval completed!")
    print(f"📁 Results saved to: {args.output_folder}")


if __name__ == "__main__":
    import pandas as pd  # For timestamp
    main()