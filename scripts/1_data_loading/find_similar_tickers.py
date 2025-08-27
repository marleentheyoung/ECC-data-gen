import pandas as pd
from difflib import SequenceMatcher
import re
from collections import defaultdict
import json

def clean_company_name(name):
    """Clean company name for better matching"""
    name = name.lower()
    suffixes = ['ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation', 
                'plc', 'sa', 'se', 'nv', 'ag', 'gmbh','holding', 'international', 'group', 'spa', 'srl', 'bv']
    
    for suffix in suffixes:
        pattern = rf'\b{re.escape(suffix)}\b\.?'
        name = re.sub(pattern, '', name)
    
    name = re.sub(r'[^\w\s]', ' ', name)
    return ' '.join(name.split()).strip()

def similarity_score(name1, name2):
    """Calculate similarity between two company names"""
    clean1 = clean_company_name(name1)
    clean2 = clean_company_name(name2)
    return SequenceMatcher(None, clean1, clean2).ratio()

def clean_ticker(ticker):
    """Clean ticker by removing country extension"""
    if pd.isna(ticker):
        return ""
    ticker = str(ticker).strip()
    ticker = re.sub(r'-[A-Z]{2,3}$', '', ticker)
    return ticker.replace('-', ' ')

def find_similar_companies(df, threshold=0.8):
    """Find companies with similar names"""
    similar_pairs = []
    companies = df[['ticker_normalized', 'company_name']].values
    
    for i, (ticker1, name1) in enumerate(companies):
        for j, (ticker2, name2) in enumerate(companies[i+1:], i+1):
            similarity = similarity_score(name1, name2)
            if similarity >= threshold:
                is_same_company = True
                
                # Ask for manual verification if similarity is below 0.85
                if similarity < 0.85:
                    print(f"\n--- Manual Verification Needed ---")
                    print(f"Similarity: {similarity:.3f}")
                    print(f"Company 1: {name1} ({ticker1})")
                    print(f"Company 2: {name2} ({ticker2})")
                    print(f"Cleaned 1: {clean_company_name(name1)}")
                    print(f"Cleaned 2: {clean_company_name(name2)}")
                    
                    while True:
                        response = input("Are these the same company? (y/n/skip): ").lower().strip()
                        if response in ['y', 'yes']:
                            is_same_company = True
                            break
                        elif response in ['n', 'no']:
                            is_same_company = False
                            break
                        elif response in ['s', 'skip']:
                            is_same_company = False
                            print("Skipping this pair...")
                            break
                        else:
                            print("Please enter 'y' for yes, 'n' for no, or 'skip' to skip")
                
                if is_same_company:
                    similar_pairs.append({
                        'ticker1': ticker1,
                        'company1': name1,
                        'ticker2': ticker2,
                        'company2': name2,
                        'similarity_score': round(similarity, 4),
                        'cleaned_name1': clean_company_name(name1),
                        'cleaned_name2': clean_company_name(name2),
                        'manually_verified': similarity < 0.85
                    })
    
    return pd.DataFrame(similar_pairs)

def cluster_similar_companies(similar_pairs_df):
    """Group all similar tickers into clusters using graph theory"""
    # Create graph where edges represent similarity
    graph = defaultdict(set)
    for _, row in similar_pairs_df.iterrows():
        ticker1 = clean_ticker(row['ticker1'])
        ticker2 = clean_ticker(row['ticker2'])
        graph[ticker1].add(ticker2)
        graph[ticker2].add(ticker1)
    
    # Find connected components (clusters)
    visited = set()
    clusters = []
    
    for ticker in graph:
        if ticker not in visited:
            cluster = set()
            stack = [ticker]
            
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    cluster.add(current)
                    stack.extend(graph[current] - visited)
            
            if len(cluster) > 1:  # Only keep clusters with multiple tickers
                clusters.append(list(cluster))
    
    return clusters

def create_ticker_mapping(clusters, issuer_tickers_set):
    """Create mapping from clusters, prioritizing tickers that exist in data"""
    ticker_mapping = {}
    
    for cluster in clusters:
        # Find which tickers in cluster exist in the data
        existing_tickers = [t for t in cluster if t in issuer_tickers_set]
        
        if existing_tickers:
            # Use first existing ticker as primary
            primary_ticker = existing_tickers[0]
            alternatives = [t for t in cluster if t != primary_ticker]
            ticker_mapping[primary_ticker] = alternatives
    
    return ticker_mapping

def main():
    csv_file_path = "/Users/marleendejonge/Desktop/ECC-data-generation/outputs/data_summary/unique_firms_stoxx600.csv"
    excel_file_path = "/Users/marleendejonge/Library/CloudStorage/OneDrive-UvA/PhD/PhD planning/Papers/PaperI/Equity pricing/Data_Rianne/Sustainability/EU/SXXP november 2024 - sustainability data2.xlsx"
    
    # Load data
    df_csv = pd.read_csv(csv_file_path)
    df_excel = pd.read_excel(excel_file_path)
    
    issuer_tickers_set = set(df_excel['ISSUER_TICKER'].dropna().astype(str))
    
    print(f"Loaded {len(df_csv)} companies from CSV")
    print(f"Found {len(issuer_tickers_set)} tickers in Excel data")
    
    # Find similar companies and cluster them
    similar_pairs = find_similar_companies(df_csv, threshold=0.8)
    
    if len(similar_pairs) > 0:
        print(f"Found {len(similar_pairs)} similar pairs")
        
        # Cluster similar companies
        clusters = cluster_similar_companies(similar_pairs)
        print(f"Created {len(clusters)} clusters")
        
        # Create ticker mapping
        ticker_mapping = create_ticker_mapping(clusters, issuer_tickers_set)
        
        # Save results
        with open("/Users/marleendejonge/Desktop/ECC-data-generation/outputs/data_summary/ticker_mapping.json", 'w') as f:
            json.dump(ticker_mapping, f, indent=2)
        
        # Display results
        print(f"\nTicker clusters found:")
        for i, cluster in enumerate(clusters, 1):
            print(f"Cluster {i}: {cluster}")
        
        print(f"\nTicker mapping:")
        for primary, alternatives in ticker_mapping.items():
            print(f"{primary} -> {alternatives}")
    
    else:
        print("No similar companies found")

if __name__ == "__main__":
    main()