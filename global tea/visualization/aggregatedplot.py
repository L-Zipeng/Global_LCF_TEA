# This script is used to plot the cost components for the aggregated technologies
# Figure 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import matplotlib as mpl

def set_plot_style():
    """Set the style for the plots to match high-impact journal aesthetics"""
    # Use a clean, professional base style
    plt.style.use('default')
    
    # High-impact journal style elements
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 15
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['legend.fontsize'] = 13
    mpl.rcParams['figure.titlesize'] = 19
    
    # Crisp, clear grid
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = ':'
    mpl.rcParams['grid.linewidth'] = 0.6
    mpl.rcParams['grid.alpha'] = 0.5
    
    # Professional looking elements
    mpl.rcParams['lines.linewidth'] = 1.8
    mpl.rcParams['patch.linewidth'] = 0.8
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '#333333'
    mpl.rcParams['xtick.major.width'] = 1.2
    mpl.rcParams['ytick.major.width'] = 1.2
    
    # Publication-quality figure size
    mpl.rcParams['figure.figsize'] = (12, 14)
    mpl.rcParams['figure.dpi'] = 300
    
    # Better contrast for readability
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['figure.facecolor'] = 'white'
    
    # Better margins
    mpl.rcParams['figure.constrained_layout.use'] = True
    mpl.rcParams['axes.xmargin'] = 0.03
    mpl.rcParams['axes.ymargin'] = 0.05

def load_results(excel_file):
    """Load results from Excel file with proper numeric conversion"""
    print(f"Opening Excel file: {excel_file}")
    
    # Check if we should use Monte Carlo stats files instead for consistency
    output_dir = Path('output')
    monte_carlo_files = list(output_dir.glob('monte_carlo_stats_*.xlsx'))
    if monte_carlo_files:
        print("Warning: Found Monte Carlo statistics files. Consider using load_monte_carlo_results() for consistency with global_cost_distribution.py")
        print(f"Monte Carlo files available: {[f.name for f in monte_carlo_files]}")
    
    xlsx = pd.ExcelFile(excel_file)
    
    # Get sheet lists
    lcox_sheets = [s for s in xlsx.sheet_names if s.startswith('LCOX_')]
    comp_sheets = [s for s in xlsx.sheet_names if s.startswith('Comp_')]
    
    print(f"Found {len(lcox_sheets)} LCOX sheets and {len(comp_sheets)} component sheets")
    
    # Load LCOX data
    lcox_data = {}
    for sheet in lcox_sheets:
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            # Make sure to set the first column as the index if it doesn't have a header
            if df.columns[0] == 'Unnamed: 0':
                df = df.set_index(df.columns[0])
            # Convert to numeric values
            lcox_data[sheet] = df.apply(pd.to_numeric, errors='coerce')
            print(f"Loaded LCOX sheet: {sheet}, shape: {df.shape}")
        except Exception as e:
            print(f"Error loading sheet {sheet}: {str(e)}")
    
    # Load component data with careful handling
    component_data = {}
    for sheet in comp_sheets:
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            
            # Check if first column contains countries (should be the index)
            if df.columns[0] == 'Unnamed: 0':
                index_col = df.iloc[:, 0]
                value_cols = df.iloc[:, 1:]
                
                # Clean up the data - make all numeric columns properly float
                for col in value_cols.columns:
                    value_cols[col] = pd.to_numeric(value_cols[col], errors='coerce')
                
                # Recreate the DataFrame
                clean_df = pd.DataFrame(value_cols.values, index=index_col, columns=value_cols.columns)
                component_data[sheet] = clean_df
            else:
                # If already indexed, just convert to numeric
                component_data[sheet] = df.apply(pd.to_numeric, errors='coerce')
            
            print(f"Loaded component sheet: {sheet}, shape: {df.shape}")
        except Exception as e:
            print(f"Error loading sheet {sheet}: {str(e)}")
    
    return lcox_data, component_data

def plot_technology_costs(component_data, tech_groups, title, subfig_label, ax, y_max=None, include_h2_co2=False, colors=None, component_names=None, mc_statistics=None):
    """
    Plot cost components for specified technology groups using component-wise median approach.
    
    This function uses a component-wise median methodology:
    1. Calculates the median value for each cost component across all countries
    2. Uses these component medians directly for plotting (no representative country)
    3. Optionally scales components to match Monte Carlo median totals when available
    4. Ensures each stacked bar represents a "typical" breakdown for that scenario
    
    This approach ensures that bars are always comparable and each represents
    the median behavior across countries for each individual cost component.
    """
    # Use provided colors and component names dictionaries
    if colors is None or component_names is None:
        raise ValueError("Colors and component_names dictionaries must be provided")
    
    # Track which components are actually used in this plot
    used_components = set()
    
    # Track CO2 values and positions for biogenic CO2 line
    co2_values_and_positions = []
    
    # Scenario names mapping - simplified but including year
    scenario_names = {
        'Base24': 'Ref.',
        'Base50': 'BAU 2050',
        '2deg50': '2°C 2050',
        '15deg50': '1.5°C 2050'
    }
    scenarios = ['Base24', 'Base50', '2deg50', '15deg50']
    
    x_pos = 0
    x_ticks = []
    x_labels = []
    tech_positions = []  # Track technology positions for labels
    tech_labels = []     # Track technology labels
    
    for group_name, techs in tech_groups.items():
        group_start = x_pos
        
        for tech in techs:
            tech_start = x_pos
            
            for scenario in scenarios:
                # Handle shortened sheet names
                base_sheet_name = f'Comp_{scenario}_{tech}'
                # Find matching sheet - use startswith to handle truncation in Excel sheet names
                matching_sheets = [s for s in component_data.keys() if s.startswith(base_sheet_name[:20])]
                
                if matching_sheets:
                    sheet_name = matching_sheets[0]
                    print(f"Found sheet for {tech} in {scenario}: {sheet_name}")
                    data = component_data[sheet_name]
                    
                    # Clean up data before calculating mean
                    # Make sure all columns are numeric
                    numeric_cols = []
                    for col in data.columns:
                        if col != 'Unnamed: 0':  # Skip the index column
                            try:
                                data[col] = pd.to_numeric(data[col], errors='coerce')
                                numeric_cols.append(col)
                            except:
                                print(f"Warning: Could not convert column {col} to numeric")
                    
                    # Check if there are valid numeric values
                    if len(numeric_cols) == 0:
                        print(f"Warning: No valid numeric data in {sheet_name}")
                        continue
                    
                    # Calculate the total cost for each country to determine distribution
                    total_costs = data[numeric_cols].sum(axis=1)
                    component_median_total = total_costs.median()
                    
                    # Check if Monte Carlo statistics are available for this technology and scenario
                    mc_median_total = None
                    if mc_statistics and scenario in mc_statistics and tech in mc_statistics[scenario]:
                        # Calculate median from Monte Carlo statistics across countries
                        mc_country_medians = [v for v in mc_statistics[scenario][tech].values() if not pd.isna(v)]
                        if mc_country_medians:
                            mc_median_total = np.median(mc_country_medians)
                            print(f"Using Monte Carlo median for {tech}-{scenario}: {mc_median_total:.4f}")
                    
                    # Calculate component-wise medians for each cost component
                    component_medians = data[numeric_cols].median()
                    
                    # Use component medians directly instead of representative country approach
                    median_representative_values = component_medians
                    
                    # If Monte Carlo data is available, scale components to match Monte Carlo total
                    if mc_median_total is not None:
                        current_total = component_medians.sum()
                        if current_total > 0:
                            scaling_factor = mc_median_total / current_total
                            median_representative_values = component_medians * scaling_factor
                            print(f"Scaled component medians by factor {scaling_factor:.3f} to match Monte Carlo total")
                        else:
                            median_representative_values = component_medians
                    p10 = total_costs.quantile(0.1)
                    p90 = total_costs.quantile(0.9)
                    min_cost = total_costs.min()
                    max_cost = total_costs.max()
                    
                    print(f"Debug for {tech} in {scenario}:")
                    print(f"Available components: {list(component_medians.index)}")
                    print(f"Country cost range: P10={p10:.4f}, P90={p90:.4f}, Min={min_cost:.4f}, Max={max_cost:.4f}")
                    print(f"Component-wise median total: {component_medians.sum():.4f}")
                    if mc_median_total is not None:
                        print(f"Monte Carlo median total: {mc_median_total:.4f} ← SCALING TARGET")
                        print(f"Difference vs component medians: {((mc_median_total - component_medians.sum()) / component_medians.sum() * 100):.1f}%")
                        print(f"Final scaled component medians sum: {median_representative_values.sum():.4f}")
                    else:
                        print(f"No Monte Carlo data available, using component-wise medians: {component_medians.sum():.4f}")
                        print(f"Final component medians sum: {median_representative_values.sum():.4f}")
                    print(f"Method: {'Component-wise Medians + Monte Carlo Scaling' if mc_median_total else 'Component-wise Medians Only'}")
                    
                    bottom = 0
                    
                    # Sort components for consistent stacking order
                    ordered_components = ['c_capex', 'c_upgrade', 'c_om', 'repex', 
                                         'c_elec', 'c_heat', 'c_bio', 'c_ng', 
                                         'c_h2', 'c_h2_storage', 'c_co2', 'c_co2_storage',
                                         'c_pw', 'c_iw']
                    
                    # Filter to only include components that exist in the data
                    components_to_plot = [comp for comp in ordered_components if comp in median_representative_values.index]
                    
                    # Plot components in consistent order (using representative country near median total)
                    for comp in components_to_plot:
                        value = float(median_representative_values[comp])
                        if value > 0.000001:  # Threshold to exclude negligible values
                            used_components.add(comp)
                        
                        # Track CO2 values and positions for biogenic CO2 line
                        if comp == 'c_co2' and value > 0.000001:
                            co2_values_and_positions.append({
                                'x_pos': x_pos,
                                'co2_bottom': bottom,
                                'co2_value': value,
                                'biogenic_height': bottom + (value * 0.25)  # 25% of DAC CO2
                            })
                            
                        ax.bar(x_pos, value, bottom=bottom,
                              color=colors.get(comp, '#888888'),  # Default color if not defined
                              width=0.8,
                              edgecolor='#333333',  # Add edge color for clearer bars
                              linewidth=0.5,        # Thin border for bars
                              label=comp if x_pos == 0 else "")
                        bottom += value
                    
                    # Add country distribution range indicators on top of the bar
                    total_median_components = median_representative_values[numeric_cols].sum()
                    
                    # Add whisker for P10-P90 range
                    ax.plot([x_pos-0.3, x_pos+0.3], [p10, p10], '-', color='#333333', linewidth=1.5)
                    ax.plot([x_pos-0.3, x_pos+0.3], [p90, p90], '-', color='#333333', linewidth=1.5)
                    ax.plot([x_pos, x_pos], [p10, p90], '-', color='#333333', linewidth=1.5)
                    
                    # Add small caps for min-max range
                    ax.plot([x_pos-0.15, x_pos+0.15], [min_cost, min_cost], '-', color='#333333', linewidth=1.0)
                    ax.plot([x_pos-0.15, x_pos+0.15], [max_cost, max_cost], '-', color='#333333', linewidth=1.0)
                    ax.plot([x_pos, x_pos], [p90, max_cost], '--', color='#333333', linewidth=1.0)
                    ax.plot([x_pos, x_pos], [min_cost, p10], '--', color='#333333', linewidth=1.0)
                    
                    x_ticks.append(x_pos)
                    x_labels.append(scenario_names[scenario])
                    x_pos += 1
                    
                else:
                    print(f"Warning: No sheet found for {tech} in {scenario}")
            
            # Only add technology label if at least one scenario was plotted
            if tech_start < x_pos:
                # Clean up technology name for display
                display_tech = tech.replace('_kerosene', '').replace('_diesel', '').replace('_methanol', '').replace('_DME', '')
                
                # Convert to formal names
                display_tech = display_tech.replace('SMR_CCS', 'SMR+CCS')
                display_tech = display_tech.replace('ATR_CCS', 'ATR+CCS')
                display_tech = display_tech.replace('TG_CCS', 'TG+CCS')
                display_tech = display_tech.replace('_FT', '-FT')
                
                # Add product suffix if needed - BUT NOT FOR KEROSENE
                if '_diesel' in tech:
                    display_tech += ' (Diesel)'
                elif '_methanol' in tech:
                    display_tech += ' (MeOH)'
                elif '_DME' in tech:
                    display_tech += ' (DME)'
                
                # Store tech position and label for later use
                tech_center = (tech_start + x_pos - 1) / 2
                tech_positions.append(tech_center)
                tech_labels.append(display_tech)
                
                x_pos += 0.5  # Space between technologies
            
        # Only add group background and label if at least one technology was plotted
        if group_start < x_pos:
            ax.axvspan(group_start - 0.25, x_pos - 0.75,
                      facecolor='#f0f0f0', alpha=0.4, zorder=-1)
            ax.text((group_start + x_pos - 1)/2, y_max * 0.9,  # Lower position for combined figure
                   group_name,
                   ha='center', va='top',
                   fontsize=12,  # Increased font for better readability
                   fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))  # Add background for readability
        
            x_pos += 1  # Space between groups
    
    # Add biogenic CO2 lines (25% of DAC CO2) after all bars are plotted
    biogenic_co2_plotted = False
    for co2_data in co2_values_and_positions:
        # Draw horizontal line at 25% of CO2 component height
        ax.hlines(co2_data['biogenic_height'], 
                 co2_data['x_pos'] - 0.4, co2_data['x_pos'] + 0.4,
                 colors='#DC4D01',  # Red color for biogenic CO2 line
                 linewidth=2.5,
                 linestyle='-',
                 alpha=0.9,
                 zorder=10)  # High zorder to appear on top
        
        # Add small markers at the ends
        ax.plot([co2_data['x_pos'] - 0.4, co2_data['x_pos'] + 0.4], 
               [co2_data['biogenic_height'], co2_data['biogenic_height']], 
               'o', color='#DC4D01', markersize=3, zorder=10)
        
        if not biogenic_co2_plotted:
            biogenic_co2_plotted = True
            used_components.add('biogenic_co2_line')
    
    # Add subfigure label (a, b, c, d)
    # ax.text(0.02, 1.05, subfig_label, transform=ax.transAxes, fontsize=16, 
    #         fontweight='bold', va='center', ha='left',
    #         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
    
    # Set title and labels
    ax.set_title(title, pad=20, fontsize=17, fontweight='bold')
    ax.set_ylabel('Levelized Cost [EUR/kWh]', fontsize=15, fontweight='bold')
    
    # Add reference price ranges for different fuel types
    x_min = -0.5  # Left edge of plot
    x_max = x_pos  # Right edge of plot
    
    # Define text properties for reference prices
    text_props = dict(
        fontsize=11,  # Increased font for better readability
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#333333', linewidth=0.5, pad=2)
    )
    
    if 'Kerosene' in title:
        # Add shaded area for kerosene reference price: 0.049-0.052 EUR/kWh
        ax.axhspan(0.049, 0.052, facecolor='#666666', alpha=0.5, zorder=-1, edgecolor='#333333', linewidth=0.8)
        ax.text(x_max * 1, 0.051, 'Fossil Jet Fuel', 
                va='bottom', ha='right', color='black', **text_props)
    
    elif 'Diesel' in title:
        # Add shaded area for Diesel reference price: 0.1-0.11 EUR/kWh
        ax.axhspan(0.1, 0.11, facecolor='#666666', alpha=0.5, zorder=-1, edgecolor='#333333', linewidth=0.8)
        ax.text(x_max * 1, 0.109, 'Fossil Diesel', 
                va='bottom', ha='right', color='black', **text_props)
    
    elif 'Hydrogen' in title:
        # Add shaded areas for SMR hydrogen: 0.027-0.054 EUR/kWh
        ax.axhspan(0.027, 0.054, facecolor='#F2D7D9', alpha=0.5, zorder=-1, edgecolor='#333333', linewidth=0.8)
        ax.text(x_max * 1, 0.052, 'SMR H$_2$', 
                va='bottom', ha='right', color='black', **text_props)
    
    elif 'Other Fuels' in title:
        # Add shaded area for Methanol: 0.052-0.149 EUR/kWh
        ax.axhspan(0.052, 0.149, facecolor='#D3B5E5', alpha=0.5, zorder=-1, edgecolor='#333333', linewidth=0.8)
        ax.text(x_max * 1, 0.145, 'FossilMethanol', 
                va='bottom', ha='right', color='black', **text_props)
        
        # Add shaded area for Methane: 0.01-0.014 EUR/kWh
        ax.axhspan(0.01, 0.014, facecolor='#FFD966', alpha=0.5, zorder=-1, edgecolor='#333333', linewidth=0.8)
        ax.text(x_max * 1, 0.013, 'Natural Gas', 
                va='bottom', ha='right', color='black', **text_props)
    
    # Set x-axis ticks for scenario names - now vertical under the x-axis
    if x_ticks:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=10, rotation=30, ha='center', va='top')
        
        # Add some extra space below the plot for the vertical labels
        plt.setp(ax.get_xticklabels(), y=0)
        
        # Add technology labels below scenario labels
        for pos, label in zip(tech_positions, tech_labels):
            ax.text(pos, -0.2, label,
                   ha='center', va='top',
                   transform=ax.get_xaxis_transform(),
                   fontsize=11,
                   fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Format y-axis with two decimal places
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Calculate appropriate y-max if not explicitly set
    if y_max is None:
        # Find maximum value in the plot data and add 20% margin
        max_value = 0
        for tech_list in tech_groups.values():
            for tech in tech_list:
                for scenario in scenarios:
                    base_sheet_name = f'Comp_{scenario}_{tech}'
                    matching_sheets = [s for s in component_data.keys() if s.startswith(base_sheet_name[:20])]
                    if matching_sheets:
                        sheet_name = matching_sheets[0]
                        try:
                            # Get numeric columns only
                            df = component_data[sheet_name]
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                total = df[numeric_cols].sum(axis=1).median()  # Use median for consistency
                                max_value = max(max_value, total)
                        except Exception as e:
                            print(f"Error calculating max for {sheet_name}: {str(e)}")
        if max_value > 0:
            y_max = max_value * 1.2  # Add 20% margin
        else:
            y_max = 0.6  # Default if no data
    
    # Set sensible tick intervals based on the y-max value
    if y_max <= 0.2:
        tick_interval = 0.02
    elif y_max <= 0.5:
        tick_interval = 0.05
    elif y_max <= 1.0:
        tick_interval = 0.1
    else:
        tick_interval = 0.2
        
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
    
    ax.set_ylim(0, y_max)
    
    # Make spines thicker for publication quality
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)  # Slightly thinner for combined figure
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, linewidth=0.6)
    
    # Create individual legend for this subplot
    if used_components:
        # Sort used components for consistent legend order
        component_order = ['c_capex', 'c_upgrade', 'c_om', 'repex', 'c_elec', 'c_heat', 
                         'c_bio', 'c_ng', 'c_h2', 'c_h2_storage', 'c_co2', 'biogenic_co2_line', 
                         'c_co2_storage', 'c_pw', 'c_iw']
        
        sorted_used_components = sorted(used_components, 
                                      key=lambda x: component_order.index(x) if x in component_order else 999)
        
        # Create patches for the legend with the correct colors
        handles = []
        labels = []
        for comp in sorted_used_components:
            color = colors[comp]
            if comp == 'biogenic_co2_line':
                # Create a line handle for biogenic CO2
                from matplotlib.lines import Line2D
                line = Line2D([0], [0], color=color, linewidth=2.5, linestyle='-', 
                             marker='o', markersize=3, markerfacecolor=color)
                handles.append(line)
            else:
                # Create a patch handle for bar components
                patch = plt.Rectangle((0,0), 1, 1, facecolor=color, edgecolor='#333333', linewidth=0.5)
                handles.append(patch)
            labels.append(component_names[comp])
        
        # Add the legend to the right side of the subplot
        legend = ax.legend(handles, labels,
                          loc='center left',
                          bbox_to_anchor=(1.02, 0.5),
                          frameon=True,
                          framealpha=0.95,
                          edgecolor='#333333',
                          fancybox=False,
                          title='Cost Components',
                          fontsize=12,
                          title_fontsize=13)
        
        # Ensure legend title is bold
        legend.get_title().set_fontweight('bold')
    
    # Return used components for reference
    return used_components

def load_monte_carlo_statistics():
    """Load Monte Carlo statistics for consistent median values"""
    from pathlib import Path
    import pandas as pd
    import numpy as np
    
    output_dir = Path('output')
    monte_carlo_files = list(output_dir.glob('monte_carlo_stats_*.xlsx'))
    
    if not monte_carlo_files:
        print("No Monte Carlo statistics files found. Using raw component data only.")
        return None
    
    print(f"Loading Monte Carlo statistics from {len(monte_carlo_files)} files for consistency...")
    mc_stats = {}
    
    # Scenario mapping to match aggregatedplot scenarios
    scenario_mapping = {
        'Base24': 'Base_2024',
        'Base50': 'Base_2050', 
        '2deg50': '2 degree_2050',
        '15deg50': '1.5 degree_2050'
    }
    
    for mc_file in monte_carlo_files:
        scenario_code = mc_file.stem.replace('monte_carlo_stats_', '')
        scenario_name = scenario_mapping.get(scenario_code, scenario_code)
        
        try:
            xl = pd.ExcelFile(mc_file)
            mc_stats[scenario_code] = {}
            
            for sheet_name in xl.sheet_names:
                if sheet_name.startswith('Stats_'):
                    tech_name = sheet_name.replace('Stats_', '')
                    df = pd.read_excel(mc_file, sheet_name=sheet_name, index_col=0)
                    
                    # Store median values for each country
                    mc_stats[scenario_code][tech_name] = {}
                    for country, row in df.iterrows():
                        mc_stats[scenario_code][tech_name][country] = row.get('median', np.nan)
                        
            print(f"Loaded Monte Carlo stats for {scenario_name}: {len(mc_stats[scenario_code])} technologies")
            
        except Exception as e:
            print(f"Error loading {mc_file}: {e}")
    
    return mc_stats

def main():
    # Set the plot style to Nature journal aesthetics
    set_plot_style()
    
    # Load Monte Carlo statistics for consistency
    mc_statistics = load_monte_carlo_statistics()
    
    # Load results
    results_file = 'output/lcox_results.xlsx'
    print(f"Loading results from {results_file}")
    lcox_data, component_data = load_results(results_file)
    
    # Print methodology being used
    if mc_statistics:
        print("\n🎯 METHODOLOGY: Component-wise Medians + Monte Carlo Scaling")
        print("   - Calculating median for each cost component across all countries")
        print("   - Using component medians directly (no representative country)")
        print("   - Scaling component medians to match Monte Carlo totals when available")
        print("   - This ensures comparable 'typical' breakdowns for each scenario")
    else:
        print("\n📊 METHODOLOGY: Component-wise Medians Only")
        print("   - Calculating median for each cost component across all countries")
        print("   - Using component medians directly for plotting")
        print("   - Each bar represents the typical breakdown for that scenario")
    
    # Print available component data sheets to debug
    print("\nAvailable component data sheets:")
    for key in sorted(component_data.keys()):
        print(f"  - {key}")
    
    # Define consistent color palette for components across all plots
    colors = {
        'c_capex': '#6ea3ff',         # deep blue for CAPEX
        'c_om': '#0d2746',            # Deep dark blue for O&M
        'repex': '#fd8d3c',           # Orange for replacement costs
        'c_elec': '#f0eebb',          # Light yellow for electricity
        'c_heat': '#e65773',          # Dark red for heat
        'c_ng': '#c0c0c0',            # Dark gray for natural gas
        'c_bio': '#3a854c',           # Green for biomass
        'c_pw': '#ffffff',            # White for pure water
        'c_iw': '#a1d99b',            # Pale green for industrial water
        'c_h2': '#abdcab',            # blue for hydrogen
        'c_co2': '#f4c28f',           # Orange-tan for DAC CO2
        'c_h2_storage': '#1f4e79',    # White blue (light blue) for H2 storage
        'c_co2_storage': '#fb6a4a',   # Light red for CO2 storage
        'c_upgrade': '#9467bd',       # Purple for upgrading costs
        'biogenic_co2_line': '#DC4D01'  # Red for biogenic CO2 line
    }
    
    # Component name mapping
    component_names = {
        'c_capex': 'CAPEX',
        'c_om': 'FOC',
        'repex': 'REPEX',
        'c_elec': 'Electricity',
        'c_heat': 'Heat',
        'c_ng': 'Natural Gas',
        'c_bio': 'Biomass',
        'c_pw': 'Pure Water',
        'c_iw': 'Industrial Water',
        'c_h2': r'H$_2$',
        'c_co2': r'DAC CO$_2$',  # Updated to specify DAC CO2
        'c_h2_storage': r'H$_2$ Storage',
        'c_co2_storage': r'CO$_2$ Storage',
        'c_upgrade': 'Upgrading CAPEX',
        'biogenic_co2_line': r'Biogenic CO$_2$'  # New entry for biogenic CO2 line
    }
    
    # Use the same colors for all plots - no special colors for H2 and CO2
    
    # Define technology groups for each plot - SIMPLIFIED to only four requested plots
    
    # 1. Hydrogen technologies
    hydrogen_techs = {
        'Green H$_2$': ['PEM', 'AE', 'SOEC'],
        'Pink H$_2$': ['HTSE', 'CuCl'],
        'Blue H$_2$': ['SMR_CCS', 'ATR_CCS', 'CLR'],
        'Turquoise H$_2$': ['M_PYR'],
        'Bio H$_2$': ['TG_CCS']
    }
    
    # 2.  / Kerosene technologies
    kerosene_techs = {
        'Solar kerosene': ['SR_FT_kerosene', 'ST_FT_kerosene'],
        'Bio kerosene': ['TG_FT_kerosene', 'HTL', 'HVO_kerosene', 'B_PYR_kerosene'],
        'Power-to-Liquid kerosene': ['RWGS_FT_kerosene', 'RWGS_MeOH_kerosene']
    }
    
    # 3. Diesel technologies
    diesel_techs = {
        'Solar Diesel': ['SR_FT_diesel', 'ST_FT_diesel'],
        'Bio Diesel': ['TG_FT_diesel', 'HVO_diesel', 'FAME'],
        'Power-to-Liquid Diesel': ['RWGS_FT_diesel']
    }
    
    # 4. Other fuels (Ammonia, Methanol, DME, Methane)
    other_fuels_techs = {
        'Ammonia': ['HB'],
        'Methanol': ['RWGS_MeOH_methanol'],
        'DME': ['RWGS_MeOH_DME'],
        'Methane': ['PTM', 'AD']
    }
    
    # Create output directory
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    # Debug available sheets for specific technologies
    tech_to_check = ['HTL', 'B_PYR_kerosene', 'SR_FT_kerosene', 'SR_FT_diesel']
    for tech in tech_to_check:
        print(f"\nChecking sheets for {tech}:")
        matching_sheets = [sheet for sheet in component_data.keys() if tech in sheet]
        for sheet in matching_sheets:
            print(f"  Found sheet: {sheet}")
            # Print representative values for debugging
            data = component_data[sheet]
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                total_costs = data[numeric_cols].sum(axis=1)
                median_total = total_costs.median()
                closest_idx = (total_costs - median_total).abs().idxmin()
                representative_values = data.loc[closest_idx]
                print(f"  Components: {list(representative_values.index)}")
                print(f"  Median total cost: {median_total:.4f}")
    
    # Create a single figure with 4 subfigures
    fig = plt.figure(figsize=(20, 26), dpi=300)  # Increased width for right-side legends and height for better visibility
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1], hspace=0.5)  # Increased spacing for vertical labels
    
    # Create axes for each subfigure
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    
    # Create plots - ONLY THE FOUR REQUESTED PLOTS
    print("\nGenerating hydrogen technologies plot...")
    plot_technology_costs(
        component_data, hydrogen_techs, 
        'Hydrogen Production Technologies - Cost Components',
        'a', ax1, y_max=0.3,
        include_h2_co2=False,
        colors=colors,
        component_names=component_names,
        mc_statistics=mc_statistics
    )
    
    print("\nGenerating kerosene technologies plot...")
    plot_technology_costs(
        component_data, kerosene_techs,
        'Kerosene Production Technologies - Cost Components',
        'b', ax2, y_max=0.8,
        include_h2_co2=True,
        colors=colors,
        component_names=component_names,
        mc_statistics=mc_statistics
    )
    
    print("\nGenerating diesel technologies plot...")
    plot_technology_costs(
        component_data, diesel_techs,
        'Diesel Production Technologies - Cost Components',
        'c', ax3, y_max=0.7,
        include_h2_co2=True,
        colors=colors,
        component_names=component_names,
        mc_statistics=mc_statistics
    )
    
    print("\nGenerating other fuels technologies plot...")
    plot_technology_costs(
        component_data, other_fuels_techs,
        'Other Fuels (Ammonia, Methanol, DME, Methane) - Cost Components',
        'd', ax4, y_max=0.8,
        include_h2_co2=True,
        colors=colors,
        component_names=component_names,
        mc_statistics=mc_statistics
    )
    
    # Add a common super title
    fig.suptitle('Levelized Cost Comparison of Sustainable Fuel Technologies', 
                fontsize=21, fontweight='bold', y=0.995)
    
    # Adjust layout to accommodate the right-side legends
    plt.tight_layout(rect=[0, 0.02, 0.82, 0.98])  # Make room for the right-side legends
    
    # Save the combined figure
    plt.savefig(output_dir / 'combined_fuel_technologies_cost_components.png',
               bbox_inches='tight',
               dpi=600,
               facecolor='white',
               edgecolor='none',
               transparent=False,
               format='png',
               pad_inches=0.1)
    
    # Also save individual plots for separate use
    print("All plots generated successfully")

if __name__ == '__main__':
    main()