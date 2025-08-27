import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
import statsmodels.api as sm
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

def carbon_causal_forest_analysis(df):
    """
    Causal Forest: High Carbon Firms Post-Paris vs Low Carbon (Before/After)
    
    Treatment Groups:
    - Treated (T=1): High carbon firms in post-Paris period (2016+)
    - Control (T=0): Low carbon firms (all periods) + High carbon firms pre-Paris
    """
    
    print("="*70)
    print("CAUSAL FOREST: HIGH CARBON POST-PARIS TREATMENT")
    print("="*70)
    
    # Data preparation
    df['MONTH'] = pd.to_datetime(df['MONTH'])
    df['post_paris'] = (df['MONTH'] >= '2016-01-01').astype(int)
    
    # Create carbon intensity groups (median split)
    carbon_median = df['LN_SCOPE_12_INTEN'].median()
    df['high_carbon'] = (df['LN_SCOPE_12_INTEN'] > carbon_median).astype(int)
    
    # Treatment assignment: High carbon firms post-Paris = 1, all others = 0
    df['treatment'] = ((df['high_carbon'] == 1) & (df['post_paris'] == 1)).astype(int)
    
    # Filter time window (2013-2019)
    df_analysis = df[(df['MONTH'] >= '2013-01-01') & (df['MONTH'] <= '2019-12-31')].copy()
    
    print(f"Treatment assignment:")
    print(f"Treated (High carbon post-Paris): {df_analysis['treatment'].sum():,}")
    print(f"Control (Low carbon all + High carbon pre-Paris): {(df_analysis['treatment'] == 0).sum():,}")
    print(f"Carbon threshold (median): {carbon_median:.3f}")
    
    # Select variables
    semantic_vars = [
        'climate_attention_ratio', 'climate_opportunity_ratio', 
        'climate_physical_risk_ratio', 'climate_transition_risk_ratio',
        'climate_transparency_disclosure_ratio'
    ]
    
    control_vars = ['LN_SIZE', 'BTM', 'Rf_M', 'Rm_M']
    region_vars = ['region'] if 'region' in df_analysis.columns else []
    
    required_vars = ['EXCESS_RET_M', 'treatment'] + semantic_vars + control_vars
    df_clean = df_analysis.dropna(subset=required_vars).copy()
    
    # Clean data
    df_clean = clean_data(df_clean, semantic_vars + control_vars)
    
    print(f"\nFinal sample: {len(df_clean):,} observations")
    print(f"Firms: {df_clean['ISSUER_TICKER'].nunique()}")
    print(f"Period: {df_clean['MONTH'].min().strftime('%Y-%m')} to {df_clean['MONTH'].max().strftime('%Y-%m')}")
    
    # Prepare variables for causal forest
    Y = df_clean['EXCESS_RET_M'].values
    T = df_clean['treatment'].values
    
    # Create region dummy if available
    if 'region' in df_clean.columns:
        df_clean['region_EU'] = (df_clean['region'] == 'EU').astype(int)
        X_vars = semantic_vars + control_vars + ['region_EU']
    else:
        X_vars = semantic_vars + control_vars
    
    X = df_clean[X_vars].fillna(0)
    
    print(f"Treatment rate: {T.mean():.3f}")
    print(f"Covariates: {len(X_vars)} variables")
    
    # Estimate causal forest
    print(f"\nEstimating Causal Forest...")
    cf_model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, random_state=42),
        discrete_treatment=True,
        n_estimators=200,
        random_state=42
    )
    
    cf_model.fit(Y, T, X=X.values)
    
    # Get treatment effects
    tau_hat = cf_model.effect(X.values)
    df_clean['treatment_effect'] = tau_hat
    
    print(f"\nResults:")
    print(f"Mean treatment effect: {tau_hat.mean():.4f}")
    print(f"Std of effects: {tau_hat.std():.4f}")
    print(f"Range: [{tau_hat.min():.4f}, {tau_hat.max():.4f}]")
    
    if tau_hat.mean() > 0:
        print(f"→ High carbon firms had {tau_hat.mean():.4f}pp HIGHER returns post-Paris")
    else:
        print(f"→ High carbon firms had {abs(tau_hat.mean()):.4f}pp LOWER returns post-Paris")
    
    # Analyze heterogeneity
    heterogeneity_results = analyze_heterogeneity(df_clean, semantic_vars)
    
    # Feature importance
    importance = cf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_vars,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Important Features:")
    for i, (_, row) in enumerate(importance_df.head().iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.3f}")
    
    # Validation
    validation_results = validate_model(cf_model, df_clean, X.values)
    
    # Create visualizations
    create_visualizations(df_clean, heterogeneity_results, importance_df)
    
    return {
        'df_clean': df_clean,
        'cf_model': cf_model,
        'heterogeneity_results': heterogeneity_results,
        'importance_df': importance_df,
        'validation_results': validation_results,
        'carbon_threshold': carbon_median
    }

def clean_data(df, vars_to_clean):
    """Clean infinite and extreme values"""
    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Winsorize at 1st and 99th percentiles
    for var in vars_to_clean:
        if var in df.columns:
            q01, q99 = df[var].quantile([0.01, 0.99])
            df[var] = df[var].clip(lower=q01, upper=q99)
    
    return df

def analyze_heterogeneity(df_clean, semantic_vars):
    """Analyze treatment effect heterogeneity by semantic variables"""
    
    print(f"\n" + "="*50)
    print("HETEROGENEITY ANALYSIS")
    print("="*50)
    
    results = {}
    
    for var in semantic_vars:
        if var in df_clean.columns and df_clean[var].notna().sum() > 100:
            # Split by median
            median_val = df_clean[var].median()
            high_group = df_clean[df_clean[var] > median_val]['treatment_effect']
            low_group = df_clean[df_clean[var] <= median_val]['treatment_effect']
            
            if len(high_group) > 10 and len(low_group) > 10:
                t_stat, p_val = ttest_ind(high_group, low_group)
                
                results[var] = {
                    'high_mean': high_group.mean(),
                    'low_mean': low_group.mean(),
                    'difference': high_group.mean() - low_group.mean(),
                    'p_value': p_val
                }
                
                print(f"{var}:")
                print(f"  High: {high_group.mean():.4f} (N={len(high_group)})")
                print(f"  Low:  {low_group.mean():.4f} (N={len(low_group)})")
                print(f"  Diff: {high_group.mean() - low_group.mean():.4f} (p={p_val:.3f})")
    
    return results

def validate_model(cf_model, df_clean, X_values):
    """Validate causal forest using best linear projection"""
    
    print(f"\n" + "="*40)
    print("MODEL VALIDATION")
    print("="*40)
    
    # Best Linear Projection
    Y_res, T_res, _, _ = cf_model.residuals_
    tau_hat = cf_model.effect(X_values)
    
    interaction_term = tau_hat.flatten() * T_res.flatten()
    X_blp = sm.add_constant(interaction_term)
    
    # Cluster by firm if possible
    if 'ISSUER_TICKER' in df_clean.columns:
        firm_clusters = df_clean['ISSUER_TICKER'].astype('category').cat.codes
        blp_model = sm.OLS(Y_res, X_blp).fit(cov_type='cluster', cov_kwds={'groups': firm_clusters})
    else:
        blp_model = sm.OLS(Y_res, X_blp).fit()
    
    print(f"Best Linear Projection:")
    print(f"Coefficient: {blp_model.params[1]:.4f}")
    print(f"P-value: {blp_model.pvalues[1]:.4f}")
    print(f"R-squared: {blp_model.rsquared:.4f}")
    
    return {
        'coefficient': blp_model.params[1],
        'p_value': blp_model.pvalues[1],
        'r_squared': blp_model.rsquared
    }

def create_visualizations(df_clean, heterogeneity_results, importance_df):
    """Create key visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Treatment effect distribution
    axes[0, 0].hist(df_clean['treatment_effect'], bins=30, alpha=0.7, color='steelblue')
    axes[0, 0].axvline(df_clean['treatment_effect'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df_clean["treatment_effect"].mean():.3f}')
    axes[0, 0].set_title('Treatment Effect Distribution')
    axes[0, 0].set_xlabel('Treatment Effect')
    axes[0, 0].legend()
    
    # 2. Feature importance
    top_features = importance_df.head(8)
    axes[0, 1].barh(range(len(top_features)), top_features['importance'], color='coral')
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'])
    axes[0, 1].set_title('Feature Importance')
    
    # 3. Treatment effects over time
    monthly_effects = df_clean.groupby(df_clean['MONTH'].dt.to_period('M'))['treatment_effect'].mean()
    monthly_effects.plot(ax=axes[1, 0], marker='o', color='darkgreen')
    axes[1, 0].axvline(pd.Period('2015-12'), color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Treatment Effects Over Time')
    axes[1, 0].set_ylabel('Mean Treatment Effect')
    
    # 4. Heterogeneity by climate attention
    if 'climate_attention_ratio' in df_clean.columns:
        sns.regplot(data=df_clean, x='climate_attention_ratio', y='treatment_effect',
                   scatter_kws={'alpha': 0.3}, ax=axes[1, 1])
        axes[1, 1].set_title('Effects vs Climate Attention')
    
    plt.suptitle('High Carbon Post-Paris: Causal Forest Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Treatment groups breakdown
    if 'high_carbon' in df_clean.columns and 'post_paris' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create 2x2 breakdown
        breakdown = df_clean.groupby(['high_carbon', 'post_paris'])['EXCESS_RET_M'].mean().unstack()
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, breakdown[0], width, label='Pre-Paris', alpha=0.7, color='lightblue')
        bars2 = ax.bar(x + width/2, breakdown[1], width, label='Post-Paris', alpha=0.7, color='darkred')
        
        ax.set_xlabel('Carbon Intensity')
        ax.set_ylabel('Average Excess Returns')
        ax.set_title('Average Returns by Carbon Intensity & Period')
        ax.set_xticks(x)
        ax.set_xticklabels(['Low Carbon', 'High Carbon'])
        ax.legend()
        
        # Highlight the treatment group
        bars2[1].set_color('red')
        bars2[1].set_alpha(1.0)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def main(df_full_with_semantics):
    """
    Main function to run the causal forest analysis
    
    Parameters:
    df_full_with_semantics: Your dataframe with all variables
    
    Returns:
    Dictionary with analysis results
    """
    
    print("Starting Causal Forest Analysis...")
    print(f"Input data shape: {df_full_with_semantics.shape}")
    
    # Check required columns
    required_cols = ['EXCESS_RET_M', 'LN_SCOPE_12_INTEN', 'MONTH', 'climate_attention_ratio']
    missing_cols = [col for col in required_cols if col not in df_full_with_semantics.columns]
    
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return None
    
    # Run analysis
    try:
        results = carbon_causal_forest_analysis(df_full_with_semantics)
        
        print(f"\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return results
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Usage
if __name__ == "__main__":
    # Assuming df_full_with_semantics is loaded
    df_full_with_semantics = pd.read_csv("/Users/marleendejonge/Desktop/ECC-data-generation/data/asset_pricing/df_fin_sus_sem_filled.csv")
    results = main(df_full_with_semantics)