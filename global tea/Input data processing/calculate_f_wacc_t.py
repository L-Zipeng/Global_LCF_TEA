import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def convert_learning_rate(rate):
    """Convert learning rate to float if it's a percentage string."""
    if isinstance(rate, str):
        if rate.endswith('%'):
            return float(rate[:-1]) / 100
        else:
            return float(rate)
    return float(rate)

def calculate_f_wacc_t(input_file='data/technology_wacc_input.csv', output_file='output/f_wacc_t_results.csv'):
    """
    Calculate f_wacc_t for each technology based on TRL, scale, and learning rates.
    
    IMPORTANT: Scale data represents global cumulative installed capacity (Mt H2eq/year).
    This is the total production capacity installed globally, not annual production.
    
    METHODOLOGY FOR MATURE TECHNOLOGIES (TRL_start = 9):
    For technologies like HB (Haber-Bosch) that are already at maximum TRL:
    - No TRL progression benefits since they're already mature
    - Use target values (Tp_target=0.01, Lm_target=0.005) directly as baseline
    - Scale effects still apply through learning curves (λ_e, λ_d)
    - Only cost reductions come from economies of scale, not technology maturation
    
    DYNAMIC LEVERAGE CALCULATION:
    The debt-to-value ratio D/V(x,t) is calculated dynamically as:
    D/V(x,t) = (D/V)₀ + γ(TRLₓ,ₜ - TRLₓ,₀) + δ ln(Sₓ,ₜ)
    
    Where:
    - (D/V)₀ depends on initial technology maturity and scale:
      * 0.75 for mature, large-scale technologies (TRL_start≥8, Scale_start≥50, e.g., HB)
      * 0.6 for mature but smaller-scale technologies (TRL_start≥8, Scale_start<50)
      * 0.5 for immature technologies (TRL_start<8)
    - γ = 0.05 (TRL progression effect: 5% increase per TRL level)
    - δ = 0.02 (scale effect coefficient)
    - Sₓ,ₜ is the scale ratio (current/reference cumulative capacity)
    - D/V is bounded between 0.2 and 0.9 for realistic financing constraints
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file with technology data (Scale = cumulative capacity)
    output_file : str
        Path to save results CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with calculated f_wacc_t values
    """
    
    # Read the data
    print("Reading input data...")
    df = pd.read_csv(input_file)
    
    # Ensure proper column names (handle any variations)
    if 'Scale' in df.columns:
        df = df.rename(columns={'Scale': 'Scale_MtH2eq'})
    
    # Convert learning rate to float if needed
    df['Learning_rate'] = df['Learning_rate'].apply(convert_learning_rate)
    
    # Constants - CORRECTED for economic logic
    Tp_0 = 0.05    # baseline technology premium for immature tech (5%)
    Lm_0 = 0.03    # baseline lender margin for immature tech (3%)
    Tp_target = 0.01  # target technology premium at TRL 9 (1%)
    Lm_target = 0.005  # target lender margin at TRL 9 (0.5%)
    base_wacc = 0.08  # normalization base (8%)
    
    # REMOVED: Scale reference to allow all technologies to benefit from their actual scale
    # All capacities will now get scale advantages proportional to their absolute size
    # Scale ratio S will be calculated directly from capacity without reference threshold
    
    print("Calculating technology-specific start values...")
    
    # Calculate start TRL and start Scale for each technology
    tech_starts = df.groupby('Technology').agg({
        'TRL': 'min',
        'Scale_MtH2eq': 'min'
    }).reset_index()
    tech_starts.columns = ['Technology', 'TRL_start', 'Scale_start']
    
    # Merge start values back to main dataframe
    df = df.merge(tech_starts, on='Technology', how='left')
    
    # Check for mature technologies (TRL_start = 9)
    mature_techs = df[df['TRL_start'] == 9]['Technology'].unique()
    if len(mature_techs) > 0:
        print(f"Found mature technologies (TRL_start = 9): {', '.join(mature_techs)}")
        print("These will use target premium/margin values with scale effects only.")
    
    print("Calculating roll-off rates...")
    
    # Calculate roll-off rates for each technology
    # CORRECTED: Ensure TRL progression doesn't create negative premiums/margins
    df['TRL_range'] = 9 - df['TRL_start']
    
    # Calculate alpha_x and beta_x more conservatively to prevent negative values
    # The progression should reduce premiums/margins proportionally, not absolutely
    df['alpha_x'] = np.where(df['TRL_range'] > 0, 
                            (Tp_0 - Tp_target) / (df['TRL_range'] + 1),  # Add 1 to prevent over-reduction
                            0)
    df['beta_x'] = np.where(df['TRL_range'] > 0, 
                           (Lm_0 - Lm_target) / (df['TRL_range'] + 1),  # Add 1 to prevent over-reduction
                           0)
    
    print("Calculating scale and learning factors...")
    
    # REVISED: Calculate scale ratio S using absolute scale reference
    # This ensures larger absolute deployments get better financing conditions
    
    # Use actual capacity data without demo plant adjustments
    # Zero capacity values remain as zero (no artificial scaling)
    df['Scale_adjusted'] = df['Scale_MtH2eq']  # Use actual cumulative capacity as-is
    
    # Use raw input data directly as scale factor (no normalization)
    # S = actual capacity in Mt H2eq/year (no reference scale division)
    df['S'] = np.maximum(df['Scale_adjusted'], 0.01)  # Minimum S = 0.01 to avoid division issues
    
    # Calculate learning factors
    df['lambda_e'] = 0.25 * df['Learning_rate']
    df['lambda_d'] = 0.125 * df['Learning_rate']
    
    print("Calculating technology premium and lender margin...")
    
    # Calculate technology premium Tp
    # CORRECTED: Base premium depends on TRL_start (lower TRL_start = higher risk)
    df['Tp_base'] = np.where(df['TRL_start'] == 9, 
                            Tp_target,  # Mature technology
                            Tp_target + (Tp_0 - Tp_target) * (9 - df['TRL_start']) / 8)  # Higher premium for lower TRL_start
    
    # Apply TRL progression benefits on top of TRL_start baseline
    df['Tp_with_progression'] = np.where(df['TRL_start'] == 9,
                                        Tp_target,  # No progression possible for mature tech
                                        df['Tp_base'] - df['alpha_x'] * (df['TRL'] - df['TRL_start']))
    
    # Apply scale effects - now using absolute scale factor
    df['Tp'] = df['Tp_with_progression'] * (df['S'] ** (-df['lambda_e']))
    
    # Calculate lender margin Lm (similar logic)
    df['Lm_base'] = np.where(df['TRL_start'] == 9, 
                            Lm_target,  # Mature technology
                            Lm_target + (Lm_0 - Lm_target) * (9 - df['TRL_start']) / 8)  # Higher margin for lower TRL_start
    
    # Apply TRL progression benefits
    df['Lm_with_progression'] = np.where(df['TRL_start'] == 9,
                                        Lm_target,  # No progression possible for mature tech
                                        df['Lm_base'] - df['beta_x'] * (df['TRL'] - df['TRL_start']))
    
    # Apply scale effects - now using absolute scale factor
    df['Lm'] = df['Lm_with_progression'] * (df['S'] ** (-df['lambda_d']))
    
    print("Calculating dynamic debt-to-value ratio...")
    
    # Dynamic D/V calculation based on TRL progression and scale effects
    # D/V(x,t) = (D/V)₀ + γ(TRLₓ,ₜ - TRLₓ,₀) + δ ln(Sₓ,ₜ)
    
    # Base D/V depends on technology maturity at start
    # For mature technologies (TRL_start ≥ 8) with large scale: start higher
    # For immature technologies (TRL_start < 8): start at 0.5
    
    gamma = 0.05  # TRL progression effect (5% increase per TRL level)
    delta = 0.02  # Scale effect coefficient
    
    # Calculate base D/V based on initial technology maturity and scale
    def calculate_base_dv(row):
        if row['TRL_start'] >= 8:  # Mature technology
            # For mature technologies, base D/V depends on initial scale
            if row['Scale_start'] >= 50:  # Large initial scale (like HB)
                return 0.75  # Start at mature industrial level
            else:
                return 0.6   # Mature but smaller scale
        else:  # Immature technology
            return 0.5       # Start low for unproven technologies
    
    df['D_V_base'] = df.apply(calculate_base_dv, axis=1)
    
    # Calculate TRL progression effect
    df['TRL_progression'] = df['TRL'] - df['TRL_start']
    
    # Calculate dynamic D/V ratio
    df['D_V_calculated'] = (df['D_V_base'] + 
                           gamma * df['TRL_progression'] + 
                           delta * np.log(df['S']))
    
    # Apply reasonable bounds to D/V (between 20% and 90%)
    df['D_V'] = np.clip(df['D_V_calculated'], 0.2, 0.9)
    
    print("Calculating cost of equity and debt...")
    
    # Calculate cost of equity Ke and cost of debt Kd
    df['Ke'] = 0.05 + 0.10 + df['Tp']  # risk-free rate + equity risk premium + technology premium
    df['Kd'] = 0.06 + df['Lm']         # risk-free rate + lender margin
    
    print("Calculating WACC and f_wacc_t...")
    
    # Calculate WACC*
    df['WACC_star'] = (1 - df['D_V']) * df['Ke'] + df['D_V'] * df['Kd']
    
    # Calculate f_wacc_t (normalized WACC)
    df['f_wacc_t'] = df['WACC_star'] / base_wacc
    
    # Create scenario-year combinations for wide format
    df['scenario_year'] = df['Scenario'] + '_' + df['Year'].astype(str)
    
    # Create wide format output
    print("Creating wide format output...")
    pivot_df = df.pivot_table(
        index='Technology', 
        columns='scenario_year', 
        values='f_wacc_t',
        aggfunc='first'  # In case of duplicates, take first value
    )
    
    # Reset index to make Technology a column
    pivot_df = pivot_df.reset_index()
    
    # Rename the Technology column to 'tech'
    pivot_df = pivot_df.rename(columns={'Technology': 'tech'})
    
    # Define the desired column order
    desired_columns = [
        'tech',
        'Base_2024',     # Updated from 2022 to 2024
        'Base_2030', 
        '2degree_2030',
        '1.5degree_2030',
        'Base_2050',
        '2degree_2050',
        '1.5degree_2050'
    ]
    
    # Check which columns exist and reorder
    available_columns = ['tech'] + [col for col in desired_columns[1:] if col in pivot_df.columns]
    missing_columns = [col for col in desired_columns if col not in pivot_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        # Add missing columns with NaN values if needed
        for col in missing_columns:
            if col != 'tech':
                pivot_df[col] = np.nan
    
    # Reorder columns according to specification
    result_df = pivot_df[desired_columns].copy()
    
    # Round f_wacc_t values to 4 decimal places
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(4)
    
    # Sort by technology name
    result_df = result_df.sort_values('tech').reset_index(drop=True)
    
    # Save results in wide format
    print(f"Saving wide format results to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Also save long format for compatibility
    long_output_file = output_file.replace('.csv', '_long.csv')
    long_cols = ['Technology', 'Scenario', 'Year', 'TRL', 'TRL_start', 'Scale_MtH2eq', 'S', 'D_V', 'WACC_star', 'f_wacc_t']
    long_df = df[long_cols].copy()
    long_df['f_wacc_t'] = long_df['f_wacc_t'].round(4)
    long_df['D_V'] = long_df['D_V'].round(4)
    long_df['WACC_star'] = long_df['WACC_star'].round(4)
    long_df.to_csv(long_output_file, index=False)
    print(f"Also saved long format to {long_output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics for f_wacc_t (Wide Format):")
    print(f"Technologies: {len(result_df)}")
    print(f"Columns: {list(result_df.columns)}")
    
    # Overall statistics across all scenario-year combinations
    numeric_cols = [col for col in result_df.columns if col != 'tech']
    all_values = result_df[numeric_cols].values.flatten()
    all_values = all_values[~np.isnan(all_values)]  # Remove NaN values
    
    print(f"\nOverall f_wacc_t statistics:")
    print(f"  Count: {len(all_values)}")
    print(f"  Mean: {np.mean(all_values):.4f}")
    print(f"  Min: {np.min(all_values):.4f}")
    print(f"  Max: {np.max(all_values):.4f}")
    print(f"  Std: {np.std(all_values):.4f}")
    
    print("\nf_wacc_t by Technology (mean across scenarios):")
    tech_means = result_df.set_index('tech')[numeric_cols].mean(axis=1).round(4)
    print(tech_means.sort_values(ascending=False))
    
    print("\nf_wacc_t by Scenario-Year (mean across technologies):")
    scenario_means = result_df[numeric_cols].mean().round(4)
    print(scenario_means)
    
    print("\nDebt-to-Value (D/V) ratio statistics:")
    print(f"  Mean D/V: {df['D_V'].mean():.4f}")
    print(f"  Min D/V: {df['D_V'].min():.4f}")
    print(f"  Max D/V: {df['D_V'].max():.4f}")
    print(f"  Std D/V: {df['D_V'].std():.4f}")
    
    print("\nD/V by Technology (mean across scenarios):")
    dv_by_tech = df.groupby('Technology')['D_V'].mean().round(4).sort_values(ascending=False)
    print(dv_by_tech)
    
    # Create visualizations using long format data for compatibility
    create_visualizations(df, long_df)
    
    # Create global map for country-specific WACC factors
    create_f_wacc_c_global_map()
    
    return result_df, df  # Return wide format result and detailed dataframes

def create_visualizations(detailed_df, result_df):
    """Create high-quality visualizations for f_wacc_t analysis with connected technology lines."""
    
    print("\nCreating high-quality visualizations...")
    
    # Set up publication-quality plotting style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Define a consistent color palette for technologies
    tech_list = sorted(result_df['Technology'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(tech_list)))
    tech_colors = dict(zip(tech_list, colors))
    
    # Create the main figure with 3 subplots (first row only)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('WACC Factor Analysis: Technology Trajectories Across Scenarios', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. f_wacc_t vs TRL - Connect points for same technology
    ax1 = axes[0]
    
    # Group data by technology and plot connected lines
    for tech in tech_list:
        tech_data = result_df[result_df['Technology'] == tech].copy()
        
        # Sort by TRL for proper line connection
        tech_data = tech_data.sort_values('TRL')
        
        # Plot line connecting all points for this technology
        ax1.plot(tech_data['TRL'], tech_data['f_wacc_t'], 
                color=tech_colors[tech], marker='o', linestyle='-', 
                alpha=0.8, linewidth=2, markersize=5, label=tech)
        

    
    ax1.set_xlabel('Technology Readiness Level (TRL)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('f_wacc_t', fontsize=12, fontweight='bold')
    ax1.set_title('f_wacc_t vs TRL by Technology', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(3.5, 9.5)
    
    # 2. f_wacc_t vs Scale - Connect points for same technology
    ax2 = axes[1]
    
    for tech in tech_list:
        tech_data = result_df[result_df['Technology'] == tech].copy()
        
        # Sort by Scale for proper line connection
        tech_data = tech_data.sort_values('Scale_MtH2eq')
        
        # Plot line connecting all points for this technology
        ax2.plot(tech_data['Scale_MtH2eq'], tech_data['f_wacc_t'], 
                color=tech_colors[tech], marker='o', linestyle='-', 
                alpha=0.8, linewidth=2, markersize=5, label=tech)
        

    
    ax2.set_xlabel('Cumulative Scale (Mt H$_2$ equivalent)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('f_wacc_t', fontsize=12, fontweight='bold')
    ax2.set_title('f_wacc_t vs Scale by Technology', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. D/V ratio vs TRL - Connect points for same technology
    ax3 = axes[2]
    
    for tech in tech_list:
        tech_data = detailed_df[detailed_df['Technology'] == tech].copy()
        
        # Sort by TRL for proper line connection
        tech_data = tech_data.sort_values('TRL')
        
        # Plot line connecting all points for this technology
        ax3.plot(tech_data['TRL'], tech_data['D_V'], 
                color=tech_colors[tech], marker='o', linestyle='-', 
                alpha=0.8, linewidth=2, markersize=5, label=tech)
        

    
    ax3.set_xlabel('Technology Readiness Level (TRL)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Debt-to-Value Ratio (D/V)', fontsize=12, fontweight='bold')
    ax3.set_title('D/V Ratio vs TRL by Technology', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(3.5, 9.5)
    ax3.set_ylim(0.4, 0.9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save high-quality figure
    plt.savefig('figures/wacc_analysis_high_quality.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    #plt.show()
    
    # Create a separate legend figure for clarity
    create_technology_legend(tech_list, tech_colors)
    
    # Create individual high-quality figures for each subplot
    create_individual_figures(detailed_df, result_df, tech_colors)

def create_technology_legend(tech_list, tech_colors):
    """Create a separate legend figure for all technologies."""
    
    fig_legend, ax_legend = plt.subplots(figsize=(10, 8))
    ax_legend.axis('off')
    
    # Create legend handles
    legend_handles = []
    for tech in tech_list:
        handle = plt.Line2D([0], [0], color=tech_colors[tech], 
                           marker='o', linestyle='-', linewidth=2, 
                           markersize=6, label=tech)
        legend_handles.append(handle)
    
    # Create legend in multiple columns
    ncols = 3 if len(tech_list) > 15 else 2
    legend = ax_legend.legend(handles=legend_handles, loc='center', 
                             ncol=ncols, fontsize=12, 
                             title='Technologies', title_fontsize=14)
    
    plt.savefig('figures/technology_legend.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    #plt.show()

def create_individual_figures(detailed_df, result_df, tech_colors):
    """Create individual high-quality figures for each analysis."""
    
    tech_list = sorted(result_df['Technology'].unique())
    
    # Individual Figure 1: f_wacc_t vs TRL
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    for tech in tech_list:
        tech_data = result_df[result_df['Technology'] == tech].copy()
        tech_data = tech_data.sort_values('TRL')
        
        ax1.plot(tech_data['TRL'], tech_data['f_wacc_t'], 
                color=tech_colors[tech], marker='o', linestyle='-', 
                alpha=0.8, linewidth=2.5, markersize=7, label=tech)
    
    ax1.set_xlabel('Technology Readiness Level (TRL)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('WACC Factor (f_wacc_t)', fontsize=14, fontweight='bold')
    ax1.set_title('WACC Factor Evolution with Technology Maturity', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(3.5, 9.5)
    
    # Add subtle background shading for TRL ranges
    ax1.axvspan(3.5, 6, alpha=0.1, color='red', label='Early Stage')
    ax1.axvspan(6, 8, alpha=0.1, color='orange', label='Development')
    ax1.axvspan(8, 9.5, alpha=0.1, color='green', label='Commercial')
    
    plt.tight_layout()
    plt.savefig('figures/f_wacc_vs_trl_individual.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    #plt.show()
    
    # Individual Figure 2: f_wacc_t vs Scale
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    for tech in tech_list:
        tech_data = result_df[result_df['Technology'] == tech].copy()
        tech_data = tech_data.sort_values('Scale_MtH2eq')
        
        ax2.plot(tech_data['Scale_MtH2eq'], tech_data['f_wacc_t'], 
                color=tech_colors[tech], marker='o', linestyle='-', 
                alpha=0.8, linewidth=2.5, markersize=7, label=tech)
    
    ax2.set_xlabel('Cumulative Scale (Mt H$_2$ equivalent/year)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('WACC Factor (f_wacc_t)', fontsize=14, fontweight='bold')
    ax2.set_title('WACC Factor Evolution with Project Scale', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('figures/f_wacc_vs_scale_individual.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    #plt.show()
    
    # Individual Figure 3: D/V ratio vs TRL
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    for tech in tech_list:
        tech_data = detailed_df[detailed_df['Technology'] == tech].copy()
        tech_data = tech_data.sort_values('TRL')
        
        ax3.plot(tech_data['TRL'], tech_data['D_V'], 
                color=tech_colors[tech], marker='o', linestyle='-', 
                alpha=0.8, linewidth=2.5, markersize=7, label=tech)
    
    ax3.set_xlabel('Technology Readiness Level (TRL)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Debt-to-Value Ratio (D/V)', fontsize=14, fontweight='bold')
    ax3.set_title('Debt-to-Value Ratio Evolution with Technology Maturity', 
                  fontsize=16, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(3.5, 9.5)
    ax3.set_ylim(0.4, 0.9)
    
    # Add horizontal lines for typical D/V ranges
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Typical Project Finance')
    ax3.axhline(y=0.7, color='darkgray', linestyle=':', alpha=0.7, label='Infrastructure Finance')
    
    plt.tight_layout()
    plt.savefig('figures/dv_ratio_vs_trl_individual.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    #plt.show()

def preprocess_geojson_safely(geojson_path):
    """
    Load and preprocess GeoJSON, skipping invalid geometries and removing Antarctica.
    """
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union
    
    with open(geojson_path, 'r') as file:
        raw_data = json.load(file)
    
    processed_features = []
    for feature in raw_data["features"]:
        try:
            geom = shape(feature["geometry"])
            # Remove Antarctica by ISO_A3 code
            if feature["properties"].get("iso") == "ATA":
                continue
            # Fix invalid geometries
            if not geom.is_valid:
                geom = geom.buffer(0)  # Attempt to fix
            # Simplify MultiPolygons
            if geom.geom_type == "MultiPolygon":
                geom = unary_union(geom)
            if geom.is_valid and not geom.is_empty:
                processed_features.append({
                    "geometry": geom,
                    "ISO_A3": feature["properties"].get("iso", None),
                    "name": feature["properties"].get("name", None)
                })
        except Exception as e:
            print(f"Skipping problematic feature: {e}")
    
    try:
        import geopandas as gpd
        return gpd.GeoDataFrame(processed_features, crs="EPSG:4326")
    except ImportError:
        print("Error: geopandas not available")
        return None

def create_f_wacc_c_global_map():
    """Create a global map showing country-specific WACC factors (f_wacc_c)."""
    
    print("\nCreating global map for country-specific WACC factors (f_wacc_c)...")
    
    # Load country WACC factors
    try:
        wacc_df = pd.read_csv('input/f_wacc_c.csv', encoding='latin-1')
        wacc_df = wacc_df.rename(columns={'ISO A3': 'ISO_A3'})
    except Exception as e:
        print(f"Error loading f_wacc_c data: {e}")
        return
    
    # Get world geometry data using the same approach as plot_individual_maps.py
    try:
        geojson_file = 'input/world_by_iso_geo.json'
        world = preprocess_geojson_safely(geojson_file)
        if world is None:
            print("Could not load world geometry data")
            return
    except Exception as e:
        print(f"Error loading world geometry: {e}")
        return
    
    # Clean and prepare data
    wacc_df['f_wacc_c'] = pd.to_numeric(wacc_df['f_wacc_c'], errors='coerce')
    wacc_df = wacc_df.dropna(subset=['f_wacc_c'])
    
    # Merge with world geometry
    world_wacc = world.merge(wacc_df, left_on='ISO_A3', right_on='ISO_A3', how='left')
    
    # Create the map
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # Define color scheme and breaks
    vmin = wacc_df['f_wacc_c'].min()
    vmax = wacc_df['f_wacc_c'].max()
    
    # Create custom breaks for better visualization
    breaks = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.5]
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fdbf6f', '#ff7f00', '#e31a1c', '#800026']
    
    # Plot countries with data
    world_wacc.plot(column='f_wacc_c', 
                   ax=ax,
                   legend=True,
                   cmap='RdYlBu_r',
                   missing_kwds={'color': 'lightgrey', 'edgecolor': 'white', 'hatch': '///', 'label': 'No data'},
                   edgecolor='white',
                   linewidth=0.5,
                   vmin=vmin,
                   vmax=vmax)
    
    # Customize the map
    ax.set_title('Country-Specific WACC Factors (f_wacc_c)\nFinancing Cost Multipliers for Clean Energy Projects', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Lower values indicate better financing conditions', fontsize=14)
    ax.axis('off')
    
    # Add text box with explanation
    textstr = '\n'.join([
        'f_wacc_c = Country WACC Factor',
        'Values closer to 0.7 = Lower financing costs',
        'Values above 2.0 = Higher financing costs',
        'Gray areas = No data available'
    ])
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Add statistics text
    stats_text = f'Global Statistics:\nMean: {wacc_df["f_wacc_c"].mean():.2f}\nMin: {vmin:.2f}\nMax: {vmax:.2f}'
    ax.text(0.02, 0.25, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('figures/global_f_wacc_c_map.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    #plt.show()
    
    # Create a detailed regional breakdown
    create_f_wacc_c_regional_analysis(wacc_df)

def create_f_wacc_c_regional_analysis(wacc_df):
    """Create detailed regional analysis of WACC factors."""
    
    # Define regions
    regions = {
        'North America': ['USA', 'CAN', 'MEX'],
        'Europe': ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'BEL', 'AUT', 'CHE', 'NOR', 'SWE', 'DNK', 'FIN'],
        'Asia-Pacific': ['JPN', 'KOR', 'CHN', 'IND', 'AUS', 'NZL', 'SGP', 'THA', 'MYS', 'IDN', 'VNM', 'PHL'],
        'Middle East': ['SAU', 'ARE', 'QAT', 'KWT', 'OMN', 'BHR', 'ISR', 'JOR', 'LBN', 'IRN', 'IRQ'],
        'Africa': ['ZAF', 'NGA', 'EGY', 'MAR', 'DZA', 'TUN', 'LBY', 'GHA', 'KEN', 'ETH', 'UGA', 'TZA'],
        'Latin America': ['BRA', 'ARG', 'CHL', 'COL', 'PER', 'URY', 'BOL', 'VEN', 'ECU', 'PRY']
    }
    
    # Add region column to dataframe
    wacc_df['Region'] = 'Other'
    for region, countries in regions.items():
        wacc_df.loc[wacc_df['ISO_A3'].isin(countries), 'Region'] = region
    
    # Create regional comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Box plot by region
    region_data = []
    region_labels = []
    for region in regions.keys():
        region_values = wacc_df[wacc_df['Region'] == region]['f_wacc_c'].dropna()
        if len(region_values) > 0:
            region_data.append(region_values)
            region_labels.append(f'{region}\n(n={len(region_values)})')
    
    bp1 = ax1.boxplot(region_data, tick_labels=region_labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
        patch.set_facecolor(color)
    
    ax1.set_title('Regional Distribution of Country WACC Factors', fontsize=14, fontweight='bold')
    ax1.set_ylabel('f_wacc_c', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Top 10 highest and lowest countries
    top_10_highest = wacc_df.nlargest(10, 'f_wacc_c')
    top_10_lowest = wacc_df.nsmallest(10, 'f_wacc_c')
    
    y_pos_high = np.arange(len(top_10_highest))
    y_pos_low = np.arange(len(top_10_lowest))
    
    # Plot top 10 highest
    bars_high = ax2.barh(y_pos_high, top_10_highest['f_wacc_c'], 
                        color='red', alpha=0.7, label='Highest Risk')
    
    # Plot top 10 lowest (offset to avoid overlap)
    bars_low = ax2.barh(y_pos_low + 11, top_10_lowest['f_wacc_c'], 
                       color='green', alpha=0.7, label='Lowest Risk')
    
    # Add country labels
    for i, (idx, row) in enumerate(top_10_highest.iterrows()):
        ax2.text(row['f_wacc_c'] + 0.05, i, f"{row['Name'][:15]}{'...' if len(row['Name']) > 15 else ''}", 
                va='center', fontsize=10)
    
    for i, (idx, row) in enumerate(top_10_lowest.iterrows()):
        ax2.text(row['f_wacc_c'] + 0.05, i + 11, f"{row['Name'][:15]}{'...' if len(row['Name']) > 15 else ''}", 
                va='center', fontsize=10)
    
    ax2.set_title('Top 10 Highest vs Lowest Risk Countries', fontsize=14, fontweight='bold')
    ax2.set_xlabel('f_wacc_c (Country WACC Factor)', fontsize=12, fontweight='bold')
    ax2.set_yticks([])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figures/f_wacc_c_regional_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    #plt.show()
    
    # Print summary statistics
    print("\nRegional WACC Factor Summary:")
    print("=" * 50)
    for region in regions.keys():
        region_values = wacc_df[wacc_df['Region'] == region]['f_wacc_c'].dropna()
        if len(region_values) > 0:
            print(f"{region:15s}: Mean={region_values.mean():.2f}, "
                  f"Std={region_values.std():.2f}, "
                  f"Range=[{region_values.min():.2f}, {region_values.max():.2f}], "
                  f"Countries={len(region_values)}")
    
    # Print extreme countries
    print(f"\nHighest Risk Countries (f_wacc_c > 2.0):")
    high_risk = wacc_df[wacc_df['f_wacc_c'] > 2.0].sort_values('f_wacc_c', ascending=False)
    for _, row in high_risk.iterrows():
        name = str(row['Name']) if not pd.isna(row['Name']) else 'Unknown'
        print(f"  {name:30s}: {row['f_wacc_c']:.2f}")
    
    print(f"\nLowest Risk Countries (f_wacc_c < 0.8):")
    low_risk = wacc_df[wacc_df['f_wacc_c'] < 0.8].sort_values('f_wacc_c')
    for _, row in low_risk.iterrows():
        name = str(row['Name']) if not pd.isna(row['Name']) else 'Unknown'
        print(f"  {name:30s}: {row['f_wacc_c']:.2f}")

if __name__ == "__main__":
    # Run the calculation
    result_df, detailed_df = calculate_f_wacc_t()
    
    print("\nCalculation completed!")
    print(f"Wide format results saved to output/f_wacc_t_results.csv")
    print(f"Long format results saved to output/f_wacc_t_results_long.csv")
    print(f"Visualizations saved to figures/")
    
    # Display first few rows of wide format results
    print("\nFirst 10 rows of wide format results:")
    print(result_df.head(10).to_string(index=False)) 