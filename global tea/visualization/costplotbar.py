# This script is used to plot the cost components from the LCOX results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import matplotlib.patheffects as path_effects
import matplotlib as mpl
import matplotlib.path as mpath

def set_plot_style():
    """Set the style for the plots to match Nature journal aesthetics"""
    # Replace deprecated 'seaborn-whitegrid' with direct style settings
    # Use a base style and customize it
    plt.style.use('default')
    
    # Nature-style elements
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 11                # Increased from 9
    mpl.rcParams['axes.labelsize'] = 11           # Increased from 9
    mpl.rcParams['axes.titlesize'] = 12           # Increased from 10
    mpl.rcParams['xtick.labelsize'] = 10          # Increased from 8
    mpl.rcParams['ytick.labelsize'] = 10          # Increased from 8
    mpl.rcParams['legend.fontsize'] = 10          # Increased from 8
    mpl.rcParams['figure.titlesize'] = 14         # Increased from 12
    
    # Clean, minimal grid
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = ':'
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.alpha'] = 0.5
    
    # Professional looking lines and markers
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['patch.linewidth'] = 0.5
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.edgecolor'] = '#555555'
    
    # Elegant figure size
    mpl.rcParams['figure.figsize'] = (10, 7)      # Increased from (8, 6)
    mpl.rcParams['figure.dpi'] = 300
    
    # Better contrast for readability
    mpl.rcParams['axes.facecolor'] = '#f8f8f8'
    mpl.rcParams['figure.facecolor'] = 'white'
    
    # Better margins
    mpl.rcParams['figure.constrained_layout.use'] = True
    mpl.rcParams['axes.xmargin'] = 0.05
    mpl.rcParams['axes.ymargin'] = 0.05

def load_component_data(excel_file):
    """Load cost component data from the Excel file with improved structure"""
    try:
        print(f"Loading cost components from: {excel_file}")
        
        # Read all sheets
        xlsx = pd.ExcelFile(excel_file)
        all_sheets = xlsx.sheet_names
        print(f"Found sheets: {all_sheets}")
        
        # Find component sheets
        comp_sheets = [s for s in all_sheets if s.startswith('Comp_')]
        if not comp_sheets:
            print("No component sheets found. Check Excel file structure.")
            return None
            
        print(f"Found {len(comp_sheets)} component sheets: {comp_sheets}")
        
        # Dictionary to hold component data
        component_data = {}
        
        # Process each component sheet
        for sheet in comp_sheets:
            try:
                # Extract scenario and technology
                parts = sheet.split('_', 2)
                if len(parts) < 3:
                    scenario = parts[1]
                    tech = ""
                else:
                    scenario = parts[1]
                    tech = parts[2]
                
                # Handle multi-product technologies (e.g. SR_FT_diesel, RWGS_MeOH_DME)
                # The technology might already include a product suffix
                tech_parts = tech.split('_')
                if len(tech_parts) > 2 and tech_parts[-1] in ['diesel', 'kerosene', 'methanol', 'DME']:
                    # This is a multi-product technology, keep the full name
                    tech = tech
                elif len(tech_parts) > 3 and tech_parts[-1] in ['diesel', 'kerosene', 'methanol', 'DME']:
                    # Handle RWGS_MeOH_methanol, RWGS_MeOH_kerosene, RWGS_MeOH_DME patterns
                    tech = tech
                
                # Load the raw data
                df = pd.read_excel(excel_file, sheet_name=sheet, index_col=0)
                print(f"Raw data from {sheet}: shape {df.shape}")
                
                # Convert to numeric values
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate statistics for each component
                means = {}
                stds = {}
                for component in df.columns:
                    values = df[component].dropna().values
                    if len(values) > 0:
                        means[component] = np.mean(values)
                        stds[component] = np.std(values)
                
                # Special handling for RWGS_MeOH_ - create separate entries for each product
                if tech == 'RWGS_MeOH_':
                    # Create entries for the three RWGS_MeOH products
                    for product_tech in ['RWGS_MeOH_methanol', 'RWGS_MeOH_kerosene', 'RWGS_MeOH_DME']:
                        if product_tech not in component_data:
                            component_data[product_tech] = {}
                        if scenario not in component_data[product_tech]:
                            component_data[product_tech][scenario] = {}
                        
                        # Use the same data for all three products (they share the same cost structure)
                        component_data[product_tech][scenario]['raw_data'] = df.copy()
                        component_data[product_tech][scenario]['mean'] = means.copy()
                        component_data[product_tech][scenario]['std'] = stds.copy()
                    
                    # Don't include the original RWGS_MeOH_ in the final results
                    continue
                
                # Create entry for this technology if it doesn't exist
                if tech not in component_data:
                    component_data[tech] = {}
                
                if scenario not in component_data[tech]:
                    component_data[tech][scenario] = {}
                
                # Store the raw data for violin plots
                component_data[tech][scenario]['raw_data'] = df
                
                # Store the statistics
                component_data[tech][scenario]['mean'] = means
                component_data[tech][scenario]['std'] = stds
                
            except Exception as e:
                print(f"Error processing sheet {sheet}: {str(e)}")
                continue
        
        # Debug: Check a few entries to verify structure
        if component_data:
            first_tech = next(iter(component_data))
            first_scenario = next(iter(component_data[first_tech]))
            print(f"\nDebug - Data structure for {first_tech}, {first_scenario}:")
            print(f"Keys available: {list(component_data[first_tech][first_scenario].keys())}")
            print(f"Mean keys: {list(component_data[first_tech][first_scenario]['mean'].keys())}")
        
        return component_data
        
    except Exception as e:
        print(f"Error loading component data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Map the technical cost component names to readable names
component_name_mapping = {
    'c_capex': 'CAPEX',
    'c_om': 'O&M',
    'c_bio': 'Biomass',
    'c_elec': 'Electricity',
    'c_heat': 'Heat',
    'c_co2': r'CO$_2$',                  
    'c_co2_storage': r'CO$_2$ Storage',  
    'c_h2': r'H$_2$',                    
    'c_h2_storage': r'H$_2$ Storage', 
    'c_ng': 'Natural Gas',
    'c_pw': 'Process Water',
    'c_iw': 'Industrial Water',
    'repex': 'Replacement Costs',
    'c_upgrade': 'Upgrading CAPEX'  # Add upgrading CAPEX
}

# Map scenario codes to readable names
scenario_name_mapping = {
    'Base24': 'Ref. 2024',
    'Base30': '2030 BAU',
    '2deg30': '2030 2°C',
    '15deg30': '2030 1.5°C',
    'Base50': '2050 BAU',
    '2deg50': '2050 2°C',
    '15deg50': '2050 1.5°C'
}

# Define scenario groups for cleaner plotting
scenario_groups = {
    '2030': ['Base30', '2deg30', '15deg30'],
    '2050': ['Base50', '2deg50', '15deg50']
}

# Define the preferred order of scenarios
preferred_order = ['Base24'] + scenario_groups['2030'] + scenario_groups['2050']

def plot_cost_components(component_data, output_dir='figures'):
    """
    Plot cost components as stacked bar charts with error bars and violin plots
    showing the distribution of total costs across countries.
    
    Args:
        component_data (dict): Dictionary containing cost component data
        output_dir (str): Directory to save the plots
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set consistent plot style
    set_plot_style()
    
    # Components/technologies to plot
    component_names = sorted(list(component_data.keys()))
    
    # Color palette for cost components
    colors = {
        # Readable component names - Nature palette (more subdued)
        'CAPEX': '#3c5488',          # Muted blue
        'O&M': '#4592ab',            # Teal blue
        'Biomass': '#5d9a60',        # Muted green
        'Electricity': '#ad494a',    # Muted red
        'Heat': '#8a5d88',           # Muted purple
        r'CO$_2$': '#dd8047',            # Muted orange - mathtext version
        r'CO$_2$ Storage': '#b279a2',    # Muted pink - mathtext version
        r'H$_2$ Storage': '#6798ce',     # Light blue - mathtext version
        r'H$_2$': '#80999a',             # Gray-teal - mathtext version
        'Natural Gas': '#a19a97',    # Warm gray
        'Process Water': '#7dc6c9',  # Light teal
        'Industrial Water': '#b0c4de', # Light steel blue
        'Replacement Costs': '#FFB366', # Orange for replacement costs
        'Upgrading CAPEX': '#ff7f50', # Coral for upgrading costs
        
        # Unicode versions kept for backward compatibility
        'CO₂': '#dd8047',            # Muted orange - Unicode version
        'CO₂ Storage': '#b279a2',    # Muted pink - Unicode version
        'H₂ Storage': '#6798ce',     # Light blue - Unicode version
        'H₂': '#80999a',             # Gray-teal - Unicode version
        
        # Original technical names (as fallback)
        'c_capex': '#3c5488',        # Muted blue
        'c_om': '#4592ab',           # Teal blue
        'c_bio': '#5d9a60',          # Muted green
        'c_elec': '#ad494a',         # Muted red
        'c_heat': '#8a5d88',         # Muted purple
        'c_co2': '#dd8047',          # Muted orange
        'c_CO2_storage': '#b279a2',  # Muted pink
        'c_co2_storage': '#b279a2',  # Pink for consistency
        'c_h2_storage': '#6798ce',   # Light blue
        'c_h2': '#80999a',           # Gray-teal
        'c_ng': '#a19a97',           # Warm gray
        'c_pw': '#7dc6c9',           # Light teal
        'c_iw': '#b0c4de',           # Light steel blue
        'c_upgrade': '#ff7f50'       # Coral for upgrading costs
    }
    
    # Fallback colors for any components not in the dict - using ColorBrewer spectral for scientific appeal
    tab10_colors = plt.cm.Spectral(np.linspace(0.1, 0.9, 10))
    
    # Iterate through components
    for component_name in component_names:
        print(f"Creating cost component plot for {component_name}")
        scenario_data = component_data[component_name]
        
        # Check available scenarios for this component
        available_scenarios = [scenario for scenario in preferred_order if scenario in scenario_data]
        print(f"  Available scenarios: {available_scenarios}")
        
        if not available_scenarios:
            print(f"  No data available for {component_name}, skipping")
            continue
            
        # Debug check - verify structure for first scenario
        if available_scenarios:
            first_scenario = available_scenarios[0]
            print(f"  Debug - First scenario: {first_scenario}")
            print(f"  Keys available: {list(scenario_data[first_scenario].keys())}")
            if 'mean' not in scenario_data[first_scenario]:
                print(f"  Warning: 'mean' key not found in data for {component_name}, {first_scenario}")
                continue
        
        n_scenarios = len(available_scenarios)
        # Adjust figure width based on number of scenarios - wider than before
        fig_width = max(8, n_scenarios * 1.3)  # Adjusted for Nature style (less wide)
        
        # Create figure with two subplots side by side - Nature style figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 5.5),  # Increased height from 4.8
                                       gridspec_kw={'width_ratios': [3, 1]})
        
        # Set a light gray background for the figure area (common in Nature)
        fig.patch.set_facecolor('#f9f9f9')
        
        # Bar width calculation - slightly thinner for Nature style
        bar_width = 0.65
        indices = np.arange(len(available_scenarios))
        
        # Track bottom positions for stacking and collect total costs for violin plot
        bottoms = np.zeros(len(available_scenarios))
        all_totals_by_scenario = {}
        
        # Get unique cost component categories across all scenarios
        all_categories = set()
        for scenario in available_scenarios:
            if 'mean' in scenario_data[scenario]:
                all_categories.update(scenario_data[scenario]['mean'].keys())
            else:
                print(f"  Warning: 'mean' key not found for {scenario}")
        all_categories = sorted(list(all_categories))
        
        # First pass to calculate total costs for error bars and violin plot
        for scenario_idx, scenario in enumerate(available_scenarios):
            if 'mean' not in scenario_data[scenario]:
                continue
                
            means = scenario_data[scenario]['mean']
            
            # Calculate total cost for this scenario (sum of all components)
            total_mean = sum(means.values())
            
            # Get all country totals for this scenario for the violin plot
            if 'raw_data' in scenario_data[scenario]:
                # Sum across all cost components for each country
                raw_data = scenario_data[scenario]['raw_data']
                country_totals = raw_data.sum(axis=1)
                all_totals_by_scenario[scenario] = country_totals
        
        # Track maximum y value needed for axis scaling
        max_y_value = 0
        
        # Create dictionary to map technical component names to readable ones
        readable_categories = []
        for category in all_categories:
            readable_name = component_name_mapping.get(category, category)
            readable_categories.append((category, readable_name))
        
        # Sort readable categories alphabetically for consistent legend
        readable_categories.sort(key=lambda x: x[1])
        
        # Second pass to plot the stacked bars
        for i, (category, readable_name) in enumerate(readable_categories):
            values = []
            for scenario in available_scenarios:
                means = scenario_data[scenario]['mean']
                # Use get with default 0 in case this category doesn't exist for this scenario
                values.append(means.get(category, 0))
            
            # Get color with better fallback mechanism
            if readable_name in colors:
                color = colors[readable_name]
            elif category in colors:
                color = colors[category]
            else:
                # Use a color from the scientific color palette if not in our dictionary
                color_idx = i % len(tab10_colors)
                color = tab10_colors[color_idx]
                print(f"  Using fallback color for category: {category} -> {readable_name}")
            
            # Plot stacked bar with distinct color - thinner edges for Nature style
            ax1.bar(indices, values, bar_width, bottom=bottoms, label=readable_name, 
                   color=color, edgecolor='#333333', linewidth=0.3)
            
            # Update bottoms for next stack
            bottoms += values
            
            # Update maximum y value
            max_y_value = max(max_y_value, np.max(bottoms))
        
        # Now add error bars for total costs only
        for scenario_idx, scenario in enumerate(available_scenarios):
            if 'std' in scenario_data[scenario]:
                # Calculate total standard deviation
                total_std = 0
                for category in all_categories:
                    if category in scenario_data[scenario]['std']:
                        total_std += scenario_data[scenario]['std'][category]**2
                total_std = np.sqrt(total_std)  # Combined std dev
                
                # Get total mean again
                total_mean = sum(scenario_data[scenario]['mean'].values())
                
                # Add error bar for total cost
                error = total_std
                # Ensure error doesn't go below zero
                lower_error = min(error, total_mean)
                
                # Update max_y_value to accommodate error bars
                max_y_value = max(max_y_value, total_mean + error)
                
                # Plot error bars - thinner lines for Nature style
                ax1.errorbar(
                    indices[scenario_idx], 
                    total_mean, 
                    yerr=[[lower_error], [error]], 
                    fmt='none', 
                    ecolor='black', 
                    capsize=3,     # Smaller capsize for Nature style
                    capthick=0.8,  # Thinner caps for Nature style
                    elinewidth=0.8 # Thinner error lines for Nature style
                )
        
        # Plot violin plots for distribution of total costs by scenario
        violin_data = []
        violin_labels = []  # Store scenario labels for violin plots
        for scenario in available_scenarios:
            # Use readable scenario name for label
            violin_labels.append(scenario_name_mapping.get(scenario, scenario))
            
            if scenario in all_totals_by_scenario:
                violin_data.append(all_totals_by_scenario[scenario])
            else:
                # If no raw data, create dummy data based on mean and std
                if 'std' in scenario_data[scenario]:
                    total_mean = sum(scenario_data[scenario]['mean'].values())
                    total_std = 0
                    for category in scenario_data[scenario]['std']:
                        total_std += scenario_data[scenario]['std'][category]**2
                    total_std = np.sqrt(total_std)
                    # Create dummy normal distribution
                    violin_data.append(np.random.normal(total_mean, total_std, 100))
                else:
                    # No std dev info, just use mean
                    total_mean = sum(scenario_data[scenario]['mean'].values())
                    violin_data.append([total_mean] * 10)
        
        # Only plot violins if we have data
        if violin_data:
            # Calculate mean and standard deviation for setting limits
            for i, data in enumerate(violin_data):
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                # Count countries outside 2 SD range (just for clipping, no annotation)
                outliers_high = sum(x > mean_val + 2*std_val for x in data)
                outliers_low = sum(x < mean_val - 2*std_val for x in data)
                
                # Clip values for plotting (but keep original for statistics)
                violin_data[i] = np.clip(data, 
                                        mean_val - 2*std_val,
                                        mean_val + 2*std_val)
                
                # Remove annotation about outliers - not needed
            
            # Position violins at 0, 1, 2, etc.
            violin_positions = np.arange(len(available_scenarios))
            violin_parts = ax2.violinplot(
                violin_data, 
                positions=violin_positions,
                showmeans=True,
                showmedians=True,  # Show medians in addition to means
                showextrema=True
            )
            
            # Customize violin appearance for Nature style (more subdued)
            for i, pc in enumerate(violin_parts['bodies']):
                # Use a gradient of colors from the Nature palette
                pc.set_facecolor('#486a9a')  # Consistent blue tone
                pc.set_edgecolor('#333333')
                pc.set_linewidth(0.5)       # Thinner outline
                pc.set_alpha(0.6 + (i * 0.05))  # Slight variation in transparency
            
            # Customize mean markers - thinner lines for Nature style
            violin_parts['cmeans'].set_color('#d1495b')  # Distinct red color
            violin_parts['cmeans'].set_linewidth(1.2)
            
            # Customize median markers - thinner lines for Nature style
            violin_parts['cmedians'].set_color('#66a182')  # Distinct green color
            violin_parts['cmedians'].set_linewidth(1.2)
            
            # Make whiskers and caps less prominent for Nature style
            for part in ['cbars', 'cmins', 'cmaxes']:
                if part in violin_parts:
                    violin_parts[part].set_linewidth(0.8)
                    violin_parts[part].set_color('#555555')
                    
            # Add horizontal grid to the violin plot for better readability - more subtle grid
            ax2.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
            
            # Set y-axis label with scientific styling
            ax2.set_ylabel('Cost Distribution (EUR/kWh)', fontsize=11)
        
        # Set the y-axis upper limit with 15% padding (Nature uses less empty space)
        y_max = max_y_value * 1.15
        ax1.set_ylim(0, y_max)
        if violin_data:
            ax2.set_ylim(0, y_max)
        
        # Set x-tick labels on the primary axis with readable scenario names
        ax1.set_xticks(indices)
        readable_scenario_names = [scenario_name_mapping.get(s, s) for s in available_scenarios]
        ax1.set_xticklabels(readable_scenario_names, rotation=45, ha='right')
        
        # Add year grouping brackets on x-axis
        # First, identify year groups in the available scenarios
        current_year_group = None
        group_start_idx = None
        
        # Dictionary to track groups
        group_indices = {}
        
        # Identify groups
        for i, scenario in enumerate(available_scenarios):
            # Check which year group this scenario belongs to
            year_group = None
            for year, scenarios in scenario_groups.items():
                if scenario in scenarios:
                    year_group = year
                    break
            
            # Skip Reference 2024 for grouping
            if scenario == 'Base24':
                continue
            
            # Start a new group or continue existing
            if year_group != current_year_group:
                # Close previous group if exists
                if current_year_group and group_start_idx is not None:
                    group_indices[current_year_group] = (group_start_idx, i-1)
                
                # Start new group
                current_year_group = year_group
                group_start_idx = i
        
        # Close the last group if exists
        if current_year_group and group_start_idx is not None:
            group_indices[current_year_group] = (group_start_idx, len(available_scenarios)-1)
        
        # Draw brackets for each group - thinner and more subtle for Nature style
        bracket_height = y_max * 0.04
        text_height = y_max * 0.10
        bracket_linewidth = 0.8
        
        for year, (start_idx, end_idx) in group_indices.items():
            # Only draw if we have at least 2 scenarios in the group
            if end_idx >= start_idx:
                # Calculate bracket position
                x_left = indices[start_idx] - bar_width/2
                x_right = indices[end_idx] + bar_width/2
                y_bottom = -y_max * 0.10  # Increased from 0.07 (move down)
                y_middle = -y_max * 0.14  # Increased from 0.10 (move down)
                
                # Draw left vertical line
                ax1.plot([x_left, x_left], [y_bottom, y_middle], 
                        color='#333333', linewidth=bracket_linewidth)
                
                # Draw right vertical line
                ax1.plot([x_right, x_right], [y_bottom, y_middle], 
                        color='#333333', linewidth=bracket_linewidth)
                
                # Draw horizontal line
                ax1.plot([x_left, x_right], [y_middle, y_middle], 
                        color='#333333', linewidth=bracket_linewidth)
                
                # # Add year text - moved down
                # text_x = (x_left + x_right) / 2
                # text_y = -y_max * 0.22  # Increased from 0.14 (move down)
                # year_text = ax1.text(text_x, text_y, year, 
                #                   ha='center', va='center', fontsize=12, fontweight='bold')  # Increased from 10
                
                # Add outline to make text more readable
                # year_text.set_path_effects([
                #     path_effects.Stroke(linewidth=2.0, foreground='white'),  # Increased from 1.5
                #     path_effects.Normal()
                #])
        
        # Adjust the bottom margin to make room for brackets and labels
        plt.subplots_adjust(bottom=0.22)  # Increased from 0.18 to accommodate lower year labels
        
        # Clean up the violin plot axis
        if violin_data:
            ax2.set_xticks(violin_positions)
            ax2.set_xticklabels(violin_labels, rotation=45, ha='right', fontsize=10)  # Increased from 8
            ax2.set_title('Distribution', fontsize=12, pad=5)  # Increased from 10
        
        # Add labels and title with correct units
        ax1.set_title(f'{component_name} Cost Components', fontsize=13, pad=10)  # Increased from 11, pad from 8
        
        # Position the legend optimally based on data
        # For Nature style: legends are typically positioned more efficiently
        total_by_scenario = [sum(scenario_data[scenario]['mean'].values()) for scenario in available_scenarios]
        
        # Calculate the maximum y value including error bars and legend space
        max_y_with_error = max_y_value
        for scenario_idx, scenario in enumerate(available_scenarios):
            if 'std' in scenario_data[scenario]:
                total_mean = sum(scenario_data[scenario]['mean'].values())
                total_std = 0
                for category in scenario_data[scenario]['std']:
                    total_std += scenario_data[scenario]['std'][category]**2
                total_std = np.sqrt(total_std)
                max_y_with_error = max(max_y_with_error, total_mean + total_std)
        
        # Add 20% padding to the y-axis to accommodate the legend
        y_max = max_y_with_error * 1.2
        ax1.set_ylim(0, y_max)
        if violin_data:
            ax2.set_ylim(0, y_max)
        
        # Create the legend with improved formatting and Nature-style
        # Position the legend inside the plot at the top right
        legend = ax1.legend(
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),  # Slightly adjusted to be inside plot margins
            ncol=2,  # Two columns for better space utilization
            framealpha=0.9,             # Increased from 0.85
            fontsize=9,                 # Increased from 7
            fancybox=False,
            handlelength=1.5,           # Increased from 1.2
            columnspacing=1.2,          # Increased from 1.0
        )
        
        # Add a more visible border around the legend
        legend.get_frame().set_linewidth(0.8)
        
        # Adjust the y-axis label based on technology
        if component_name == 'DAC':
            ax1.set_ylabel('Cost Components (EUR/kg)', fontsize=11)  # Increased from 9
            if violin_data:
                ax2.set_ylabel('Cost Distribution (EUR/kg)', fontsize=11)  # Increased from 9
        else:
            ax1.set_ylabel('Cost Components (EUR/kWh)', fontsize=11)  # Increased from 9
            if violin_data:
                ax2.set_ylabel('Cost Distribution (EUR/kWh)', fontsize=11)  # Increased from 9
        
        # Adjust layout for Nature style (tighter)
        plt.tight_layout(rect=[0, 0.02, 1, 0.92])  # Adjust the plot area
        
        # Save figure
        output_file = os.path.join(output_dir, f'cost_components_{component_name}.png')
        plt.savefig(output_file, dpi=600, bbox_inches='tight', 
                    format='png', transparent=False, facecolor='#ffffff', pad_inches=0.1)
        print(f"  Saved figure to {output_file}")
        
        # Close the figure to free memory
        plt.close(fig)

def main():
    """Main function to create cost component visualizations"""
    try:
        # Set plot style
        set_plot_style()
        
        # Load results
        results_file = '../../output/results/lcox_results.xlsx'
        print(f"Loading results from {results_file}")
        component_data = load_component_data(results_file)
        
        if component_data is None:
            print("Failed to load data. Exiting.")
            return
        
        # Create output directory
        output_dir = Path('../../output/figures')
        output_dir.mkdir(exist_ok=True)
        
        print("\nCreating cost component plots...")
        plot_cost_components(component_data, output_dir)
        
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"Error in visualization process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()