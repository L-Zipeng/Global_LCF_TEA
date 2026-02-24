import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d, PchipInterpolator
import warnings
warnings.filterwarnings('ignore')

def convert_learning_rate(rate):
    """Convert learning rate to float if it's a percentage string."""
    if isinstance(rate, str):
        if rate.endswith('%'):
            return float(rate[:-1]) / 100
        else:
            return float(rate)
    return float(rate)

def calculate_future_capex(input_file='data/capex learning rate input.xlsx', 
                          output_file='output/future_capex_results.csv',
                          base_year=2024):
    """
    Calculate future CAPEX for technologies using learning rates and scale effects.
    
    METHODOLOGY:
    Future CAPEX is calculated using the learning curve approach:
    CAPEX(t) = CAPEX(base) * (Scale(t) / Scale(base))^(-learning_rate)
    
    Where:
    - CAPEX(base) is the 2024 base CAPEX from the 'capex' sheet
    - Scale(t) is the cumulative capacity at time t
    - Scale(base) is the base year cumulative capacity
    - learning_rate is the technology-specific learning rate
    
    Parameters:
    -----------
    input_file : str
        Path to input Excel file with capacity/scale and CAPEX data
    output_file : str
        Path to save results CSV file
    base_year : int
        Base year for CAPEX reference (default: 2024)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with calculated future CAPEX values
    """
    
    print("Reading input data from Excel file...")
    
    try:
        # Read both sheets from the Excel file
        capacity_df = pd.read_excel(input_file, sheet_name='capacity and size')
        capex_df = pd.read_excel(input_file, sheet_name='capex')
        
        print(f"Loaded capacity data: {len(capacity_df)} rows")
        print(f"Loaded CAPEX data: {len(capex_df)} rows")
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Please ensure the file exists and has the correct sheet names:")
        print("- 'capacity and size' sheet with columns: Technology, Scenario, Year, Learning_rate, Scale")
        print("- 'capex' sheet with columns: Technology, CAPEX (EUR/kW)")
        return None
    
    # Display sheet structures for verification
    print("\nCapacity and Scale sheet columns:", capacity_df.columns.tolist())
    print("CAPEX sheet columns:", capex_df.columns.tolist())
    
    print("\nFirst few rows of capacity data:")
    print(capacity_df.head())
    
    print("\nFirst few rows of CAPEX data:")
    print(capex_df.head())
    
    # Clean column names to handle any variations
    capacity_df.columns = capacity_df.columns.str.strip()
    capex_df.columns = capex_df.columns.str.strip()
    
    # Handle potential column name variations
    capex_col = None
    for col in capex_df.columns:
        if 'CAPEX' in col.upper() or 'EUR/KW' in col.upper():
            capex_col = col
            break
    
    if capex_col is None:
        print("Error: Could not find CAPEX column in the capex sheet")
        return None
    
    print(f"Using CAPEX column: '{capex_col}'")
    
    # Standardize column names
    if capex_col != 'CAPEX_EUR_per_kW':
        capex_df = capex_df.rename(columns={capex_col: 'CAPEX_EUR_per_kW'})
    
    # Convert learning rate to decimal if needed
    capacity_df['Learning_rate'] = capacity_df['Learning_rate'].apply(convert_learning_rate)
    
    # Ensure numeric types
    capacity_df['Year'] = pd.to_numeric(capacity_df['Year'], errors='coerce')
    capacity_df['Scale'] = pd.to_numeric(capacity_df['Scale'], errors='coerce')
    capex_df['CAPEX_EUR_per_kW'] = pd.to_numeric(capex_df['CAPEX_EUR_per_kW'], errors='coerce')
    
    # Remove rows with missing critical data
    capacity_df = capacity_df.dropna(subset=['Technology', 'Scenario', 'Year', 'Learning_rate', 'Scale'])
    capex_df = capex_df.dropna(subset=['Technology', 'CAPEX_EUR_per_kW'])
    
    print(f"\nAfter cleaning:")
    print(f"Capacity data: {len(capacity_df)} rows")
    print(f"CAPEX data: {len(capex_df)} rows")
    
    # Merge capacity data with base CAPEX data
    merged_df = capacity_df.merge(capex_df, on='Technology', how='left')
    
    # Check for technologies without CAPEX data
    missing_capex = merged_df[merged_df['CAPEX_EUR_per_kW'].isna()]['Technology'].unique()
    if len(missing_capex) > 0:
        print(f"Warning: Missing CAPEX data for technologies: {missing_capex}")
        merged_df = merged_df.dropna(subset=['CAPEX_EUR_per_kW'])
    
    print(f"Merged data: {len(merged_df)} rows")
    
    # Calculate base scale for each technology (reference scale)
    # Use the minimum scale for each technology across all scenarios as the base
    base_scales = merged_df.groupby('Technology')['Scale'].min().to_dict()
    merged_df['Scale_base'] = merged_df['Technology'].map(base_scales)
    
    # For technologies where all scales are the same, use a small reference scale
    # to avoid division by zero in learning curve calculation
    merged_df['Scale_base'] = merged_df['Scale_base'].replace(0, 0.1)  # Minimum scale of 0.1
    
    print("\nCalculating future CAPEX using learning curves...")
    
    # Calculate future CAPEX using learning curve formula
    # CAPEX(t) = CAPEX(base) * (Scale(t) / Scale(base))^(-learning_rate)
    
    # Ensure scale ratio is at least 1 (no negative learning)
    merged_df['Scale_ratio'] = np.maximum(merged_df['Scale'] / merged_df['Scale_base'], 1.0)
    
    # Apply learning curve
    merged_df['CAPEX_future'] = (merged_df['CAPEX_EUR_per_kW'] * 
                                (merged_df['Scale_ratio'] ** (-merged_df['Learning_rate'])))
    
    # Calculate cost reduction percentage
    merged_df['Cost_reduction_pct'] = ((merged_df['CAPEX_EUR_per_kW'] - merged_df['CAPEX_future']) / 
                                      merged_df['CAPEX_EUR_per_kW'] * 100)
    
    # Round results for clarity
    merged_df['CAPEX_future'] = merged_df['CAPEX_future'].round(2)
    merged_df['Cost_reduction_pct'] = merged_df['Cost_reduction_pct'].round(2)
    
    # Create output dataframe with relevant columns
    result_df = merged_df[['Technology', 'Scenario', 'Year', 'Learning_rate', 'Scale', 
                          'Scale_base', 'Scale_ratio', 'CAPEX_EUR_per_kW', 'CAPEX_future', 
                          'Cost_reduction_pct']].copy()
    
    # Sort by Technology, Scenario, and Year
    result_df = result_df.sort_values(['Technology', 'Scenario', 'Year']).reset_index(drop=True)
    
    # Save detailed results
    print(f"\nSaving detailed results to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Create wide format for easy comparison
    create_wide_format_output(result_df, output_file)
    
    # Generate summary statistics and visualizations
    create_capex_analysis(result_df, merged_df)
    
    # Print summary statistics
    print_summary_statistics(result_df)
    
    return result_df

def create_wide_format_output(result_df, output_file):
    """Create wide format output for easy comparison across scenarios."""
    
    print("Creating wide format output...")
    
    # Create scenario-year combinations
    result_df['scenario_year'] = result_df['Scenario'] + '_' + result_df['Year'].astype(str)
    
    # Create wide format for future CAPEX
    wide_capex = result_df.pivot_table(
        index='Technology',
        columns='scenario_year',
        values='CAPEX_future',
        aggfunc='first'
    ).reset_index()
    
    # Create wide format for cost reduction percentages
    wide_reduction = result_df.pivot_table(
        index='Technology',
        columns='scenario_year',
        values='Cost_reduction_pct',
        aggfunc='first'
    ).reset_index()
    
    # Add base CAPEX for reference
    base_capex = result_df.groupby('Technology')['CAPEX_EUR_per_kW'].first().reset_index()
    wide_capex = wide_capex.merge(base_capex, on='Technology', how='left')
    
    # Reorder columns to put base CAPEX first
    cols = ['Technology', 'CAPEX_EUR_per_kW'] + [col for col in wide_capex.columns 
                                                if col not in ['Technology', 'CAPEX_EUR_per_kW']]
    wide_capex = wide_capex[cols]
    
    # Rename base CAPEX column for clarity
    wide_capex = wide_capex.rename(columns={'CAPEX_EUR_per_kW': 'Base_CAPEX_2024_EUR_per_kW'})
    
    # Save wide format files
    wide_capex_file = output_file.replace('.csv', '_wide_capex.csv')
    wide_reduction_file = output_file.replace('.csv', '_wide_cost_reduction.csv')
    
    wide_capex.to_csv(wide_capex_file, index=False)
    wide_reduction.to_csv(wide_reduction_file, index=False)
    
    print(f"Wide format CAPEX saved to: {wide_capex_file}")
    print(f"Wide format cost reduction saved to: {wide_reduction_file}")

def create_capex_analysis(result_df, merged_df):
    """Create comprehensive analysis and visualizations."""
    
    print("\nCreating CAPEX analysis visualizations...")
    
    # Set up plotting style
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
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Create main analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CAPEX Learning Curve Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. CAPEX learning curves over time with uncertainty bands
    ax1 = axes[0, 0]
    
    # Get technologies to display - include more than just top 10
    all_techs = result_df['Technology'].unique()
    # Sort by base CAPEX for consistent ordering
    tech_capex = result_df.groupby('Technology')['CAPEX_EUR_per_kW'].first().sort_values(ascending=False)
    
    # Exclude HTL and HTSE as requested, and include up to 15 technologies
    excluded_techs = ['HTL', 'HTSE']
    available_techs = [tech for tech in tech_capex.index if tech not in excluded_techs]
    display_techs = available_techs[:15]
    
    # Color map for different technologies
    tech_colors = plt.cm.tab20(np.linspace(0, 1, len(display_techs)))
    
    # Define consistent time range for all curves
    time_range = np.linspace(2024, 2050, 200)  # Higher resolution for smoother curves
    
    for i, tech in enumerate(display_techs):
        tech_data = result_df[result_df['Technology'] == tech]
        base_capex = tech_data['CAPEX_EUR_per_kW'].iloc[0]
        learning_rate = tech_data['Learning_rate'].iloc[0]
        base_scale = tech_data['Scale_base'].iloc[0]
        
        # Organize data by scenario and create smooth curves using actual data points
        scenario_curves = {}
        for scenario in tech_data['Scenario'].unique():
            scenario_data = tech_data[tech_data['Scenario'] == scenario].sort_values('Year')
            
            if len(scenario_data) > 0:
                # Get actual data points
                data_years = scenario_data['Year'].values
                data_capex = scenario_data['CAPEX_future'].values
                learning_rate = scenario_data['Learning_rate'].iloc[0]
                
                # Ensure data is sorted and monotonically decreasing (or at least non-increasing)
                # Sort by year to ensure proper order
                sort_idx = np.argsort(data_years)
                data_years = data_years[sort_idx]
                data_capex = data_capex[sort_idx]
                
                # Create smooth interpolation that respects learning curve behavior
                if len(data_years) == 1:
                    # If only one data point, create constant line
                    capex_smooth = np.full_like(time_range, data_capex[0])
                elif len(data_years) == 2:
                    # For two points, use linear interpolation
                    capex_smooth = np.interp(time_range, data_years, data_capex)
                else:
                    # For multiple points, use learning rate formula with smooth scale interpolation
                    # Get corresponding scale data
                    data_scales = scenario_data.sort_values('Year')['Scale'].values
                    base_scale = scenario_data['Scale_base'].iloc[0]
                    
                    # Create smooth scale interpolation
                    if len(data_years) == 3:
                        # For 3 points, use monotonic interpolation for scales
                        # Use pchip (Piecewise Cubic Hermite Interpolating Polynomial) for monotonic interpolation
                        
                        # Ensure scales are monotonically increasing
                        if np.all(np.diff(data_scales) >= 0):
                            scale_interp = PchipInterpolator(data_years, data_scales)
                            scale_smooth = scale_interp(time_range)
                        else:
                            # Fallback to linear if scales are not monotonic
                            scale_smooth = np.interp(time_range, data_years, data_scales)
                    else:
                        # Linear interpolation for other cases
                        scale_smooth = np.interp(time_range, data_years, data_scales)
                    
                    # Ensure scale never goes below base scale and is monotonically increasing
                    scale_smooth = np.maximum(scale_smooth, base_scale)
                    
                    # Apply learning curve formula: CAPEX(t) = CAPEX(base) * (Scale(t)/Scale(base))^(-learning_rate)
                    scale_ratio_smooth = scale_smooth / base_scale
                    capex_smooth = base_capex * (scale_ratio_smooth ** (-learning_rate))
                    
                    # Ensure the result is monotonically decreasing (or non-increasing)
                    # If any point increases, fix it by making it equal to the previous point
                    for k in range(1, len(capex_smooth)):
                        if capex_smooth[k] > capex_smooth[k-1]:
                            capex_smooth[k] = capex_smooth[k-1]
                    
                    # Final bounds check
                    capex_smooth = np.maximum(capex_smooth, base_capex * 0.05)  # Min 5% of base CAPEX
                    capex_smooth = np.minimum(capex_smooth, base_capex * 1.05)  # Max 105% of base CAPEX
                
                scenario_curves[scenario] = {
                    'years': time_range,
                    'capex': capex_smooth,
                    'data_years': data_years,
                    'data_capex': data_capex
                }
        
        # Plot with 2 degree as main line and others as uncertainty band
        color = tech_colors[i]
        tech_label = tech[:15] + '...' if len(tech) > 15 else tech
        
        if '2degree' in scenario_curves:
            # Main line: 2 degree scenario
            main_curve = scenario_curves['2degree']
            ax1.plot(main_curve['years'], main_curve['capex'], 
                    color=color, alpha=0.9, linewidth=2.5, 
                    label=f'{tech_label} (2°C)', zorder=3)
            
            # Create uncertainty band with other scenarios
            upper_curve = main_curve['capex'].copy()
            lower_curve = main_curve['capex'].copy()
            
            for scenario in ['1.5degree', 'Base']:
                if scenario in scenario_curves:
                    curve = scenario_curves[scenario]
                    upper_curve = np.maximum(upper_curve, curve['capex'])
                    lower_curve = np.minimum(lower_curve, curve['capex'])
            
            # Fill uncertainty band only if there are other scenarios
            if len(scenario_curves) > 1:
                ax1.fill_between(main_curve['years'], lower_curve, upper_curve,
                               color=color, alpha=0.15, zorder=1)
        
        elif '1.5degree' in scenario_curves:
            # Fallback to 1.5 degree if 2 degree not available
            main_curve = scenario_curves['1.5degree']
            ax1.plot(main_curve['years'], main_curve['capex'], 
                    color=color, alpha=0.9, linewidth=2.5, 
                    label=f'{tech_label} (1.5°C)', zorder=3)
            
            # Create uncertainty band with other scenarios
            upper_curve = main_curve['capex'].copy()
            lower_curve = main_curve['capex'].copy()
            
            for scenario in ['Base']:
                if scenario in scenario_curves:
                    curve = scenario_curves[scenario]
                    upper_curve = np.maximum(upper_curve, curve['capex'])
                    lower_curve = np.minimum(lower_curve, curve['capex'])
            
            # Fill uncertainty band only if there are other scenarios
            if len(scenario_curves) > 1:
                ax1.fill_between(main_curve['years'], lower_curve, upper_curve,
                               color=color, alpha=0.15, zorder=1)
        
        elif 'Base' in scenario_curves:
            # Last fallback to Base scenario
            main_curve = scenario_curves['Base']
            ax1.plot(main_curve['years'], main_curve['capex'], 
                    color=color, alpha=0.9, linewidth=2.5, 
                    label=f'{tech_label} (Base)', zorder=3)
    
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('CAPEX (EUR/kW)', fontweight='bold')
    ax1.set_title('CAPEX Evolution Over Time (Top 15 Technologies)\nSmooth monotonic curves, Main lines: 2°C scenario, Bands: uncertainty range', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    
    # 2. Cost reduction vs Learning rate with trend curves
    ax2 = axes[0, 1]
    
    # Aggregate by technology to avoid overplotting
    tech_summary = result_df.groupby('Technology').agg({
        'Learning_rate': 'first',
        'Cost_reduction_pct': 'max',  # Maximum reduction achieved
        'Scale_ratio': 'max'  # Maximum scale achieved
    }).reset_index()
    
    # Create theoretical curves for different scale ratios
    lr_range = np.linspace(tech_summary['Learning_rate'].min(), 
                          tech_summary['Learning_rate'].max(), 100)
    
    # Plot theoretical curves for different scale scenarios
    scale_scenarios = [2, 5, 10, 20, 50]
    colors_curves = plt.cm.viridis(np.linspace(0, 1, len(scale_scenarios)))
    
    for j, scale_ratio in enumerate(scale_scenarios):
        theoretical_reduction = (1 - (scale_ratio ** (-lr_range))) * 100
        ax2.plot(lr_range * 100, theoretical_reduction, 
                color=colors_curves[j], alpha=0.7, linewidth=2,
                linestyle='--', label=f'Scale Ratio {scale_ratio}x')
    
    # Plot actual data points
    scatter = ax2.scatter(tech_summary['Learning_rate'] * 100, tech_summary['Cost_reduction_pct'],
                         alpha=0.8, s=100, c=tech_summary['Cost_reduction_pct'], 
                         cmap='RdYlGn', edgecolors='black', linewidth=1, zorder=5)
    
    ax2.set_xlabel('Learning Rate (%)', fontweight='bold')
    ax2.set_ylabel('Maximum Cost Reduction (%)', fontweight='bold')
    ax2.set_title('Cost Reduction vs Learning Rate', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Cost Reduction (%)', fontweight='bold')
    
    # 3. Scale effects with full learning curves for each technology
    ax3 = axes[1, 0]
    
    # Create theoretical learning curves for each technology
    scale_range = np.logspace(0, 2, 100)  # Scale ratio from 1 to 100
    
    # Get technologies with diverse learning rates for better visualization
    tech_lr_df = result_df.groupby('Technology')['Learning_rate'].first().reset_index()
    tech_lr_df = tech_lr_df.sort_values('Learning_rate')
    
    # Select technologies that span the full range of learning rates
    # Exclude HTL and HTSE as requested
    excluded_from_scale = ['HTL', 'HTSE']
    available_for_scale = tech_lr_df[~tech_lr_df['Technology'].isin(excluded_from_scale)]
    
    # Select 12 technologies to ensure good visibility
    n_curves = 12
    selected_indices = np.linspace(0, len(available_for_scale) - 1, n_curves, dtype=int)
    selected_techs = available_for_scale.iloc[selected_indices]['Technology'].tolist()
    
    # Use a more diverse color palette for better distinction
    colors_tech = plt.cm.Set3(np.linspace(0, 1, len(selected_techs)))
    
    for j, tech in enumerate(selected_techs):
        lr = result_df[result_df['Technology'] == tech]['Learning_rate'].iloc[0]
        theoretical_reduction = (1 - (scale_range ** (-lr))) * 100
        
        tech_label = tech[:10] + '...' if len(tech) > 10 else tech
        ax3.plot(scale_range, theoretical_reduction,
                color=colors_tech[j], alpha=0.9, linewidth=3,
                label=f'{tech_label} (LR={lr:.1%})', zorder=3)
        
        # Add actual data points for this technology
        tech_data = result_df[result_df['Technology'] == tech]
        ax3.scatter(tech_data['Scale_ratio'], tech_data['Cost_reduction_pct'],
                   color=colors_tech[j], alpha=1.0, s=100, edgecolors='white', 
                   linewidth=2, zorder=5)
    
    # Add reference learning rate curves with lighter styling
    reference_lrs = [0.05, 0.10, 0.15, 0.20, 0.25]
    for ref_lr in reference_lrs:
        theoretical_reduction = (1 - (scale_range ** (-ref_lr))) * 100
        ax3.plot(scale_range, theoretical_reduction,
                color='gray', alpha=0.2, linewidth=1, linestyle=':', zorder=1)
    
    # Add reference learning rate labels in a better position
    ax3.text(0.02, 0.98, 'Reference LR lines:\n5%, 10%, 15%, 20%, 25%', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             fontsize=8)
    
    ax3.set_xlabel('Scale Ratio (Scale/Base Scale)', fontweight='bold')
    ax3.set_ylabel('Cost Reduction (%)', fontweight='bold')
    ax3.set_title('Scale Effects on Cost Reduction\nFull Learning Curves by Technology (12 Technologies)', fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7, loc='center right', ncol=1)
    
    # 4. Technology comparison - current vs future CAPEX
    ax4 = axes[1, 1]
    
    # Get final year data for each technology (maximum reduction)
    final_data = result_df.loc[result_df.groupby(['Technology', 'Scenario'])['Year'].idxmax()]
    tech_comparison = final_data.groupby('Technology').agg({
        'CAPEX_EUR_per_kW': 'first',
        'CAPEX_future': 'min'  # Best case (minimum future CAPEX)
    }).reset_index()
    
    # Exclude HTL and HTSE for consistency
    tech_comparison = tech_comparison[~tech_comparison['Technology'].isin(['HTL', 'HTSE'])]
    
    # Sort by current CAPEX for better visualization
    tech_comparison = tech_comparison.sort_values('CAPEX_EUR_per_kW', ascending=True)
    
    # Take top 15 for readability
    tech_comparison = tech_comparison.head(15)
    
    x_pos = np.arange(len(tech_comparison))
    width = 0.35
    
    # Calculate cost reduction for color coding
    tech_comparison['reduction_pct'] = ((tech_comparison['CAPEX_EUR_per_kW'] - tech_comparison['CAPEX_future']) / 
                                       tech_comparison['CAPEX_EUR_per_kW'] * 100)
    
    bars1 = ax4.barh(x_pos - width/2, tech_comparison['CAPEX_EUR_per_kW'], 
                     width, label='Current CAPEX (2024)', alpha=0.8, color='lightcoral')
    bars2 = ax4.barh(x_pos + width/2, tech_comparison['CAPEX_future'], 
                     width, label='Future CAPEX (Best Case)', alpha=0.8, 
                     color=plt.cm.RdYlGn(tech_comparison['reduction_pct']/tech_comparison['reduction_pct'].max()))
    
    ax4.set_xlabel('CAPEX (EUR/kW)', fontweight='bold')
    ax4.set_ylabel('Technology', fontweight='bold')
    ax4.set_title('Current vs Future CAPEX Comparison', fontweight='bold')
    ax4.set_yticks(x_pos)
    ax4.set_yticklabels([tech[:20] + '...' if len(tech) > 20 else tech 
                        for tech in tech_comparison['Technology']], fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('figures/capex_learning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual detailed plots
    create_detailed_technology_plots(result_df)

def create_detailed_technology_plots(result_df):
    """Create detailed plots for individual technologies."""
    
    print("Creating detailed technology plots...")
    
    # Get unique technologies
    technologies = result_df['Technology'].unique()
    
    # Create plots for top 12 technologies (by base CAPEX), excluding HTL and HTSE
    all_tech_capex = result_df.groupby('Technology')['CAPEX_EUR_per_kW'].first()
    available_detailed_techs = [tech for tech in all_tech_capex.index if tech not in ['HTL', 'HTSE']]
    top_12_techs = all_tech_capex[available_detailed_techs].nlargest(12).index
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Individual Technology CAPEX Trajectories', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, tech in enumerate(top_12_techs):
        if i >= 12:
            break
            
        ax = axes[i]
        tech_data = result_df[result_df['Technology'] == tech]
        
        # Plot trajectories for each scenario
        for scenario in tech_data['Scenario'].unique():
            scenario_data = tech_data[tech_data['Scenario'] == scenario].sort_values('Year')
            ax.plot(scenario_data['Year'], scenario_data['CAPEX_future'], 
                   marker='o', label=scenario, linewidth=2, markersize=4)
        
        # Add horizontal line for base CAPEX
        base_capex = tech_data['CAPEX_EUR_per_kW'].iloc[0]
        ax.axhline(y=base_capex, color='red', linestyle='--', alpha=0.7, 
                  label=f'Base CAPEX ({base_capex:.0f})')
        
        ax.set_title(tech[:25] + '...' if len(tech) > 25 else tech, fontweight='bold', fontsize=10)
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('CAPEX (EUR/kW)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add learning rate annotation
        lr = tech_data['Learning_rate'].iloc[0]
        ax.text(0.02, 0.98, f'LR: {lr:.1%}', transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=8)
    
    # Hide unused subplots
    for i in range(len(top_12_techs), 12):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('figures/detailed_technology_capex_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(result_df):
    """Print comprehensive summary statistics."""
    
    print("\n" + "="*60)
    print("CAPEX LEARNING CURVE ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall statistics
    print(f"\nDataset Overview:")
    print(f"  Total records: {len(result_df):,}")
    print(f"  Technologies: {result_df['Technology'].nunique()}")
    print(f"  Scenarios: {result_df['Scenario'].nunique()}")
    print(f"  Years: {result_df['Year'].min():.0f} - {result_df['Year'].max():.0f}")
    
    # Learning rate statistics
    lr_stats = result_df.groupby('Technology')['Learning_rate'].first()
    print(f"\nLearning Rate Statistics:")
    print(f"  Mean: {lr_stats.mean():.2%}")
    print(f"  Median: {lr_stats.median():.2%}")
    print(f"  Range: {lr_stats.min():.2%} - {lr_stats.max():.2%}")
    print(f"  Std Dev: {lr_stats.std():.2%}")
    
    # CAPEX statistics
    print(f"\nBase CAPEX Statistics (2024):")
    base_capex_stats = result_df.groupby('Technology')['CAPEX_EUR_per_kW'].first()
    print(f"  Mean: {base_capex_stats.mean():.0f} EUR/kW")
    print(f"  Median: {base_capex_stats.median():.0f} EUR/kW")
    print(f"  Range: {base_capex_stats.min():.0f} - {base_capex_stats.max():.0f} EUR/kW")
    
    print(f"\nFuture CAPEX Statistics:")
    print(f"  Mean: {result_df['CAPEX_future'].mean():.0f} EUR/kW")
    print(f"  Median: {result_df['CAPEX_future'].median():.0f} EUR/kW")
    print(f"  Range: {result_df['CAPEX_future'].min():.0f} - {result_df['CAPEX_future'].max():.0f} EUR/kW")
    
    # Cost reduction statistics
    print(f"\nCost Reduction Statistics:")
    print(f"  Mean reduction: {result_df['Cost_reduction_pct'].mean():.1f}%")
    print(f"  Median reduction: {result_df['Cost_reduction_pct'].median():.1f}%")
    print(f"  Max reduction: {result_df['Cost_reduction_pct'].max():.1f}%")
    print(f"  Min reduction: {result_df['Cost_reduction_pct'].min():.1f}%")
    
    # Top technologies by cost reduction potential
    print(f"\nTop 10 Technologies by Maximum Cost Reduction:")
    top_reductions = result_df.groupby('Technology')['Cost_reduction_pct'].max().sort_values(ascending=False).head(10)
    for i, (tech, reduction) in enumerate(top_reductions.items(), 1):
        tech_name = tech[:40] + '...' if len(tech) > 40 else tech
        print(f"  {i:2d}. {tech_name:<40} {reduction:6.1f}%")
    
    # Technologies with highest learning rates
    print(f"\nTop 10 Technologies by Learning Rate:")
    top_lr = result_df.groupby('Technology')['Learning_rate'].first().sort_values(ascending=False).head(10)
    for i, (tech, lr) in enumerate(top_lr.items(), 1):
        tech_name = tech[:40] + '...' if len(tech) > 40 else tech
        print(f"  {i:2d}. {tech_name:<40} {lr:6.1%}")
    
    # Scenario comparison
    print(f"\nAverage Cost Reduction by Scenario:")
    scenario_reductions = result_df.groupby('Scenario')['Cost_reduction_pct'].mean().sort_values(ascending=False)
    for scenario, reduction in scenario_reductions.items():
        print(f"  {scenario:<20} {reduction:6.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    Path('output').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)
    
    # Run the calculation
    result_df = calculate_future_capex()
    
    if result_df is not None:
        print("\n" + "="*50)
        print("CALCULATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Results saved to:")
        print(f"  - output/future_capex_results.csv (detailed)")
        print(f"  - output/future_capex_results_wide_capex.csv (wide format)")
        print(f"  - output/future_capex_results_wide_cost_reduction.csv (cost reduction %)")
        print(f"  - figures/capex_learning_analysis.png")
        print(f"  - figures/detailed_technology_capex_trajectories.png")
        
        print(f"\nFirst 10 rows of results:")
        print(result_df[['Technology', 'Scenario', 'Year', 'CAPEX_EUR_per_kW', 
                        'CAPEX_future', 'Cost_reduction_pct']].head(10).to_string(index=False))
    else:
        print("Calculation failed. Please check the input file and try again.") 