"""
Script to create:
1. Bar charts showing cost components by country (ranked from highest to lowest)
2. Monte Carlo distribution plots (also ranked)
With each fuel production technology and year-scenario in one figure
"""

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import warnings
import math
warnings.filterwarnings('ignore')

def set_plot_style():
    """Set publication-quality plot style"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'figure.constrained_layout.use': True,
        'mathtext.default': 'regular'  # Add this to improve subscript rendering
    })

def format_sigfigs(value, sig=2):
    """Format a number to two significant digits as a string"""
    if value == 0:
        return "0"
    digits = sig - int(math.floor(math.log10(abs(value)))) - 1
    return f"{value:.{max(digits,0)}f}" if digits >= 0 else f"{round(value, -digits)}"

def load_results():
    """Load both Monte Carlo and component data"""
    # Load Monte Carlo results
    mc_results = load_monte_carlo_results()
    
    # Load component data  
    component_data = load_component_data_from_excel()
    
    if mc_results is None and component_data is None:
        print("No data available for analysis!")
        return None
        
    return {
        'monte_carlo_results': mc_results.get('monte_carlo_results', {}) if mc_results else {},
        'monte_carlo_statistics': mc_results.get('monte_carlo_statistics', {}) if mc_results else {},
        'component_data': component_data if component_data else {}
    }

def load_monte_carlo_results():
    """Load Monte Carlo results from pickle files or reconstruct from Excel files"""
    output_dir = Path('output')
    
    # Try improved results first, then original, then reconstructed
    for filename in ['monte_carlo_results_improved.pkl', 'monte_carlo_results.pkl', 'monte_carlo_results_reconstructed.pkl']:
        file_path = output_dir / filename
        if file_path.exists():
            print(f"Loading Monte Carlo results from {file_path}")
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    # If no pickle files found, try to reconstruct from Excel files
    print("No pickle files found. Attempting to reconstruct from Excel files...")
    return reconstruct_monte_carlo_from_excel(output_dir)

def reconstruct_monte_carlo_from_excel(output_dir):
    """Reconstruct Monte Carlo data from Excel statistics files"""
    # Mapping for scenario standardization
    scenario_mapping = {
        'Base24': 'Base_2024',
        'Base30': 'Base_2030',
        'Base50': 'Base_2050',
        '2deg30': '2 degree_2030',
        '2deg50': '2 degree_2050',
        '15deg30': '1.5 degree_2030',
        '15deg50': '1.5 degree_2050'
    }
    
    results = {
        'monte_carlo_results': {},
        'monte_carlo_statistics': {}
    }
    
    # Find all Monte Carlo stats files
    mc_files = list(output_dir.glob('monte_carlo_stats_*.xlsx'))
    
    if not mc_files:
        print("No Monte Carlo statistics files found!")
        return None
    
    for file_path in mc_files:
        # Extract scenario code from filename
        scenario_code = file_path.stem.replace('monte_carlo_stats_', '')
        if '_' in scenario_code:
            scenario_code = scenario_code.split('_')[0]  # Remove suffix like '_improved'
        
        # Map to full scenario name
        scenario = scenario_mapping.get(scenario_code, scenario_code)
        
        # Initialize scenario in results
        if scenario not in results['monte_carlo_results']:
            results['monte_carlo_results'][scenario] = {}
            results['monte_carlo_statistics'][scenario] = {}
        
        # Load the Excel file with all sheets
        try:
            xl = pd.ExcelFile(file_path)
            print(f"Loading {file_path} with {len(xl.sheet_names)} sheets")
            
            for sheet_name in xl.sheet_names:
                # Extract technology name from sheet name
                if sheet_name.startswith('Stats_'):
                    tech = sheet_name.replace('Stats_', '')
                else:
                    tech = sheet_name
                
                # Clean up tech name
                tech = tech.strip()
                
                try:
                    # Load the sheet data
                    stats_df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
                    
                    # Initialize tech in results
                    if tech not in results['monte_carlo_statistics'][scenario]:
                        results['monte_carlo_statistics'][scenario][tech] = {}
                        results['monte_carlo_results'][scenario][tech] = {}
                    
                    # For each country in the stats, store statistics and generate samples
                    for country, row in stats_df.iterrows():
                        if pd.isna(country) or country == '':
                            continue
                            
                        # Convert row to dictionary
                        stats = row.to_dict()
                        
                        # Store statistics
                        results['monte_carlo_statistics'][scenario][tech][country] = stats
                        
                        # Generate synthetic Monte Carlo samples from statistics
                        num_samples = 1000
                        mean = stats.get('mean', 0)
                        std = stats.get('std', mean * 0.1)  # Default to 10% of mean if std not available
                        
                        if std > 0 and mean > 0:
                            # Generate samples using normal distribution
                            samples = np.random.normal(mean, std, num_samples)
                            # Ensure no negative values for costs
                            samples = np.maximum(samples, 0.001)
                        else:
                            # If std is 0 or mean is 0, generate constant samples
                            samples = np.full(num_samples, max(mean, 0.001))
                        
                        results['monte_carlo_results'][scenario][tech][country] = samples
                        
                except Exception as e:
                    print(f"Error loading sheet {sheet_name} from {file_path}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
    
    if results['monte_carlo_results']:
        print(f"Successfully reconstructed Monte Carlo data for {len(results['monte_carlo_results'])} scenarios")
        
        # Save reconstructed results to pickle for faster loading next time
        try:
            with open(output_dir / 'monte_carlo_results_reconstructed.pkl', 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved reconstructed results to monte_carlo_results_reconstructed.pkl")
        except Exception as e:
            print(f"Warning: Could not save reconstructed results: {e}")
        
        return results
    else:
        print("Failed to reconstruct Monte Carlo data from Excel files")
        return None

def load_component_data_from_excel():
    """Load component data from Excel file"""
    excel_file = 'output/lcox_results.xlsx'
    
    if not Path(excel_file).exists():
        print(f"Excel file {excel_file} not found!")
        return None
    
    print(f"Loading component data from {excel_file}")
    
    try:
        xlsx = pd.ExcelFile(excel_file)
        comp_sheets = [s for s in xlsx.sheet_names if s.startswith('Comp_')]
        print(f"Found {len(comp_sheets)} component sheets")
        
        component_data = {}
        for sheet in comp_sheets:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet, index_col=0)
                # Convert to numeric, replacing errors with NaN
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                component_data[sheet] = df
                print(f"Loaded component sheet: {sheet}, shape: {df.shape}")
            except Exception as e:
                print(f"Error loading sheet {sheet}: {str(e)}")
        
        return component_data
    
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def get_component_styling():
    """Get consistent styling for cost components"""
    colors = {
        'c_capex': '#3182bd',         # Blue for CAPEX
        'c_om': '#6baed6',            # Light blue for O&M
        'repex': '#fd8d3c',           # Orange for replacement costs
        'c_upgrade': '#9467bd',       # Purple for upgrading costs
        'c_elec': '#f0eebb',          # Light yellow for electricity
        'c_heat': '#9b3a4d',          # Dark red for heat
        'c_bio': '#31a354',           # Green for biomass
        'c_ng': '#636363',            # Dark gray for natural gas
        'c_pw': '#74c476',            # Light green for process water
        'c_iw': '#a1d99b',            # Pale green for industrial water
        'c_h2': '#9ecae1',            # Pale blue for hydrogen
        'c_co2': '#f4c28f',           # Orange-tan for CO2
        'c_h2_storage': '#c6dbef',    # Very light blue for H2 storage
        'c_co2_storage': '#fb6a4a',   # Light red for CO2 storage
        'c_shipping': '#756bb1',      # Purple for shipping
    }
    
    # Use HTML-style subscripts for molecule names to improve compatibility
    names = {
        'c_capex': 'CAPEX',
        'c_om': 'O&M',
        'repex': 'Replacement',
        'c_upgrade': 'Upgrading',
        'c_elec': 'Electricity',
        'c_heat': 'Heat',
        'c_bio': 'Biomass',
        'c_ng': 'Natural Gas',
        'c_pw': 'Process Water',
        'c_iw': 'Industrial Water',
        'c_h2': 'H$_2$',              # Changed to matplotlib-compatible subscript
        'c_co2': 'CO$_2$',            # Changed to matplotlib-compatible subscript
        'c_h2_storage': 'H$_2$ Storage',    # Changed to matplotlib-compatible subscript
        'c_co2_storage': 'CO$_2$ Storage',  # Changed to matplotlib-compatible subscript
        'c_shipping': 'Shipping'
    }
    
    return colors, names

def get_country_names():
    """Define country code to name mapping from TEA input.xlsx"""
    try:
        # Load the ISO A3 sheet from TEA_input.xlsx
        iso_data = pd.read_excel('data/TEA input.xlsx', sheet_name='ISO A3')
        # Create mapping from ISO A3 code to country name
        country_map = {row['ISO A3']: row['Country'] for _, row in iso_data.iterrows() if pd.notna(row['ISO A3'])}
        print(f"Loaded {len(country_map)} country mappings from TEA input.xlsx")
        return country_map
    except Exception as e:
        print(f"Error loading country mappings: {e}")
        # Fallback to the existing dictionary if file can't be loaded
        return {
            'USA': 'United States', 'DEU': 'Germany', 'CHN': 'China', 
            'JPN': 'Japan', 'AUS': 'Australia', 'BRA': 'Brazil', 
            'ZAF': 'South Africa', 'GBR': 'United Kingdom', 'FRA': 'France', 
            'ITA': 'Italy', 'ESP': 'Spain', 'CAN': 'Canada', 'MEX': 'Mexico',
            'RUS': 'Russia', 'IND': 'India', 'KOR': 'South Korea', 
            'NOR': 'Norway', 'SWE': 'Sweden', 'NLD': 'Netherlands', 
            'CHE': 'Switzerland', 'NZL': 'New Zealand', 'ARG': 'Argentina',
            'CHL': 'Chile', 'PER': 'Peru', 'COL': 'Colombia', 'THA': 'Thailand', 
            'IDN': 'Indonesia', 'MYS': 'Malaysia', 'SGP': 'Singapore', 
            'PHL': 'Philippines', 'VNM': 'Vietnam', 'BGD': 'Bangladesh', 
            'PAK': 'Pakistan', 'IRN': 'Iran', 'IRQ': 'Iraq', 'SAU': 'Saudi Arabia',
            'ARE': 'UAE', 'QAT': 'Qatar', 'KWT': 'Kuwait', 'TUR': 'Turkey', 
            'EGY': 'Egypt', 'MAR': 'Morocco', 'DZA': 'Algeria', 'TUN': 'Tunisia', 
            'LBY': 'Libya', 'SDN': 'Sudan', 'ETH': 'Ethiopia', 'KEN': 'Kenya', 
            'TZA': 'Tanzania', 'UGA': 'Uganda', 'GHA': 'Ghana', 'NGA': 'Nigeria', 
            'CIV': 'Ivory Coast', 'SEN': 'Senegal', 'MLI': 'Mali', 'BFA': 'Burkina Faso',
            'NER': 'Niger', 'TCD': 'Chad', 'CMR': 'Cameroon', 'GAB': 'Gabon', 
            'COD': 'DR Congo', 'AGO': 'Angola', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe', 
            'BWA': 'Botswana', 'NAM': 'Namibia', 'MOZ': 'Mozambique', 
            'MDG': 'Madagascar', 'MUS': 'Mauritius',
            'DJI': 'Djibouti', 'ERI': 'Eritrea', 'BEN': 'Benin', 'GNB': 'Guinea-Bissau',
            'GIN': 'Guinea', 'TGO': 'Togo', 'SLE': 'Sierra Leone', 'GNQ': 'Equatorial Guinea',
            'LBR': 'Liberia'
        }

def find_component_sheet(component_data, tech, scenario):
    """Find the appropriate component sheet for a technology and scenario"""
    # Map scenario names to sheet naming convention
    scenario_mapping = {
        'Base_2024': 'Base24',
        'Base_2030': 'Base30', 
        'Base_2050': 'Base50',
        '2 degree_2030': '2deg30',
        '2 degree_2050': '2deg50',
        '1.5 degree_2030': '15deg30',
        '1.5 degree_2050': '15deg50'
    }
    
    scenario_code = scenario_mapping.get(scenario, scenario)
    
    # Try exact match first
    exact_sheet = f'Comp_{scenario_code}_{tech}'
    if exact_sheet in component_data:
        return exact_sheet
    
    # Try with partial match - check first 20 characters to handle Excel truncation
    # This is similar to how aggregatedplot.py handles it
    matching_sheets = [s for s in component_data.keys() 
                      if s.startswith(f'Comp_{scenario_code}_{tech}'[:20])]
    
    if matching_sheets:
        return matching_sheets[0]
    
    # Print debug info about available sheets for this scenario
    scenario_sheets = [s for s in component_data.keys() if f'Comp_{scenario_code}' in s]
    print(f"Available sheets for {scenario_code}: {len(scenario_sheets)} sheets")
    for sheet in scenario_sheets[:5]:  # Print first 5 for debug
        print(f"  {sheet}")
    if len(scenario_sheets) > 5:
        print(f"  ...and {len(scenario_sheets)-5} more")
    
    # Try checking if sheet exists with underscore variations
    if '_' in tech:
        # Try variations without trailing product type
        base_tech = tech.split('_')[0]
        matching_sheets = [s for s in component_data.keys() 
                          if s.startswith(f'Comp_{scenario_code}_{base_tech}'[:20])]
        if matching_sheets:
            print(f"Found sheet with base tech name: {matching_sheets[0]}")
            return matching_sheets[0]
    
    print(f"No component data sheet found for {tech} in {scenario}")
    return None

def create_horizontal_component_plot(results, tech, scenario, output_dir):
    """
    Create horizontal bar chart showing cost components by country, ranked from highest to lowest
    """
    colors, component_names = get_component_styling()
    
    # Get country name mapping
    country_name_map = get_country_names()
    
    sheet_name = find_component_sheet(results['component_data'], tech, scenario)
    if not sheet_name:
        print(f"No component data sheet found for {tech} in {scenario}")
        return
    
    data = results['component_data'][sheet_name]
    
    # Calculate total costs for each country and sort
    country_totals = {}
    for country in data.index:
        if pd.notna(country):
            row_data = data.loc[country]
            
            # Sum only numeric values in the row
            if isinstance(row_data, pd.Series):
                numeric_values = [v for v in row_data.values if isinstance(v, (int, float)) and pd.notna(v)]
                total = sum(numeric_values)
            else:
                numeric_cols = row_data.select_dtypes(include=[np.number])
                total = numeric_cols.sum().sum()
                
            if total > 0:  # Only include countries with valid data
                country_totals[country] = total
    
    if not country_totals:
        print(f"No valid cost data for {tech} in {scenario}")
        return
    
    # Sort countries by total cost (highest to lowest)
    sorted_countries = sorted(country_totals.items(), key=lambda x: x[1], reverse=True)
    
    # Use all countries from the data
    countries = [item[0] for item in sorted_countries]
    
    # Look up full country names
    country_labels = [country_name_map.get(country, country) for country in countries]
    
    # Set up the figure with better spacing
    # Adjust figure height based on number of countries
    height = max(8, len(countries) * 0.25 + 2)  
    fig, ax = plt.subplots(figsize=(10, height))
    
    # Component order priority
    component_order = ['c_capex', 'c_om', 'repex', 'c_upgrade', 'c_elec', 'c_heat', 
                      'c_bio', 'c_ng', 'c_h2', 'c_h2_storage', 'c_co2', 'c_co2_storage',
                      'c_pw', 'c_iw', 'c_shipping']
    
    # Filter to components that exist in the data
    available_components = [comp for comp in component_order if comp in data.columns]
    
    # Initialize left array for horizontal stacking
    lefts = np.zeros(len(countries))
    
    # Store bars for legend
    bar_handles = []
    bar_labels = []
    
    # Plot each component - INCREASE BAR HEIGHT to remove gaps
    for component in available_components:
        values = []
        for country in countries:
            if country in data.index:
                value = data.loc[country, component]
                if pd.isna(value):
                    value = 0
                values.append(value)
            else:
                values.append(0)
        
        # Only plot if there are non-zero values
        if any(v > 0.001 for v in values):
            # Use height=0.9 to reduce gaps between bars
            bar = ax.barh(range(len(countries)), values, left=lefts, height=0.9,
                  label=component_names.get(component, component),
                  color=colors.get(component, '#888888'),
                  edgecolor='white', linewidth=0.5)
            lefts += np.array(values)
            
            bar_handles.append(bar)
            bar_labels.append(component_names.get(component, component))
    
    # Add total cost values at the end of bars
    for i, (country, total) in enumerate(sorted_countries):
        ax.text(total * 1.01, i, format_sigfigs(total, 2), 
               va='center', ha='left', fontsize=8, fontweight='bold')
    
    # Customize the plot
    ax.set_title(f'{tech} - Cost Components by Country ({scenario})', fontweight='bold', pad=15)
    ax.set_xlabel('Cost (EUR/kWh)', fontweight='bold')
    
    # Set y-axis with full country names
    ax.set_yticks(range(len(country_labels)))
    ax.set_yticklabels(country_labels, fontsize=8)
    
    # IMPORTANT: Set y-axis limits to remove the extra space
    ax.set_ylim(-0.5, len(countries)-0.5)
    
    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend with cleaner layout and more columns
    ax.legend(bar_handles, bar_labels, loc='upper center',
             bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(bar_labels)),
             frameon=True, fancybox=True, fontsize=9)
    
    # Adjust layout with tight padding
    plt.tight_layout(pad=2)
    
    # Save plot
    filename = f'ranked_cost_components_{tech}_{scenario.replace(" ", "_")}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ranked component plot: {filename}")
    return filename

def create_monte_carlo_distribution_plot(results, tech, scenario, output_dir):
    """
    Create violin plot showing Monte Carlo distribution by country, ranked by median cost
    """
    if (scenario not in results.get('monte_carlo_results', {}) or
        tech not in results['monte_carlo_results'][scenario]):
        print(f"No Monte Carlo data available for {tech} in {scenario}")
        return
    
    # Get country name mapping
    country_name_map = get_country_names()
    
    mc_data = results['monte_carlo_results'][scenario][tech]
    
    # Use all countries present in the data
    countries_data = []
    country_codes = []
    country_medians = {}
    
    for country, data in mc_data.items():
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            # Remove any problematic values
            clean_data = np.array(data)
            clean_data = clean_data[np.isfinite(clean_data)]
            
            if len(clean_data) > 0:
                countries_data.append(clean_data)
                country_codes.append(country)
                country_medians[country] = np.median(clean_data)
    
    if not countries_data:
        print(f"No valid Monte Carlo data for {tech} in {scenario}")
        return
    
    # Sort countries by median cost (highest to lowest)
    sorted_indices = [i for i, code in sorted(enumerate(country_codes), 
                                             key=lambda x: country_medians[x[1]], 
                                             reverse=True)]
    
    # Reorder data based on ranking
    countries_data = [countries_data[i] for i in sorted_indices]
    country_codes = [country_codes[i] for i in sorted_indices]
    
    # Look up FULL country names
    country_labels = [country_name_map.get(country, country) for country in country_codes]
    
    # Create figure - adjust height based on number of countries
    height = max(8, len(countries_data) * 0.25 + 2)
    fig, ax = plt.subplots(figsize=(10, height))
    
    # Create custom filled violin plot with color gradient
    cmap = LinearSegmentedColormap.from_list("custom_red", ["#ffcccc", "#990000"])
    violin_colors = cmap(np.linspace(0, 1, len(countries_data)))
    
    # Add reference line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Plot violins sideways - ADJUST WIDTHS to reduce space between violins
    parts = ax.violinplot(countries_data, positions=range(len(countries_data)),
                        vert=False, showmedians=True, showextrema=True,
                        widths=0.9)  # Increased width to reduce gaps
    
    # Customize violin appearance
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1)
    
    # Customize median lines
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    
    # Customize extrema lines
    for partname in ['cbars', 'cmins', 'cmaxes']:
        parts[partname].set_color('black')
        parts[partname].set_linewidth(1)
    
    # Customize the plot
    ax.set_title(f'{tech} - Cost Distribution by Country ({scenario})', fontweight='bold', pad=15)
    ax.set_xlabel('Cost (EUR/kWh)', fontweight='bold')
    ax.set_ylabel('Countries (ranked by median cost)', fontweight='bold')
    
    # Set y-axis with full country names
    ax.set_yticks(range(len(country_labels)))
    ax.set_yticklabels(country_labels, fontsize=8)
    
    # IMPORTANT: Set y-axis limits to remove extra space
    ax.set_ylim(-0.5, len(countries_data)-0.5)
    
    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add statistics for top countries
    stats_text = []
    for i, (data, code) in enumerate(zip(countries_data[:5], country_codes[:5])):
        median = np.median(data)
        mean = np.mean(data)
        std = np.std(data)
        stats_text.append(f"{code}: median={median:.3f}, mean={mean:.3f}, σ={std:.3f}")
    
    # Position statistics text better
    ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='left', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout with tight padding
    plt.tight_layout(pad=2)
    
    # Save plot
    filename = f'ranked_monte_carlo_{tech}_{scenario.replace(" ", "_")}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ranked Monte Carlo plot: {filename}")
    return filename

def create_multi_technology_figure(results, fuel_category, techs, scenario, output_dir):
    """
    Create a multi-panel figure showing all technologies in a fuel category for a specific scenario
    """
    tech_count = len(techs)
    if tech_count == 0:
        print(f"No technologies specified for {fuel_category}")
        return
    
    tech_labels = {
        'HTL': 'HTL', 'SR_FT_kerosene': 'SR-FT', 'ST_FT_kerosene': 'ST-FT',
        'TG_FT_kerosene': 'TG-FT', 'HVO_kerosene': 'HVO', 'B_PYR_kerosene': 'B-PYR',
        'RWGS_FT_kerosene': 'RWGS-FT', 'RWGS_MeOH_kerosene': 'MeOH-SAF',
        'SR_FT_diesel': 'SR-FT', 'ST_FT_diesel': 'ST-FT', 'TG_FT_diesel': 'TG-FT',
        'HVO_diesel': 'HVO', 'FAME': 'FAME', 'RWGS_FT_diesel': 'RWGS-FT',
        'PEM': 'PEM', 'AE': 'AE', 'SOEC': 'SOEC', 'HTL': 'HTL', 'HTSE': 'HTSE',
        'CuCl': 'CuCl', 'SMR_CCS': 'SMR-CCS', 'ATR_CCS': 'ATR-CCS', 'CLR': 'CLR',
        'M_PYR': 'M-PYR', 'TG_CCS': 'TG-CCS', 'HB': 'HB', 'RWGS_MeOH_methanol': 'MeOH',
        'RWGS_MeOH_DME': 'DME', 'PTM': 'PTM', 'AD': 'AD'
    }
    
    # Create subfigures for both visualization types
    output_files = {
        'components': [],
        'monte_carlo': []
    }
    
    # Create individual plots first
    for tech in techs:
        # Create component plot
        comp_filename = create_horizontal_component_plot(results, tech, scenario, output_dir)
        if comp_filename:
            output_files['components'].append(comp_filename)
        
        # Create Monte Carlo plot
        mc_filename = create_monte_carlo_distribution_plot(results, tech, scenario, output_dir)
        if mc_filename:
            output_files['monte_carlo'].append(mc_filename)
    
    # Return the generated filenames
    return output_files

def main():
    """Main function to create all country cost visualization plots"""
    set_plot_style()
    
    # Load results
    print("Loading data...")
    results = load_results()
    if not results:
        print("No data available for analysis!")
        return
    
    # Create output directory
    output_dir = Path('figures/ranked_cost_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define fuel categories and their technologies
    fuel_categories = {
        'Kerosene (SAF)': [
            'HTL', 'SR_FT_kerosene', 'ST_FT_kerosene', 'TG_FT_kerosene',
            'HVO_kerosene', 'B_PYR_kerosene', 'RWGS_FT_kerosene'
        ],
        'Diesel': [
            'SR_FT_diesel', 'ST_FT_diesel', 'TG_FT_diesel',
            'HVO_diesel', 'FAME', 'RWGS_FT_diesel'
        ],
        'Hydrogen': [
            'PEM', 'AE', 'SOEC', 'HTSE', 'SMR_CCS', 'ATR_CCS'
        ],
        'Other Fuels': [
            'HB', 'RWGS_MeOH_methanol', 'RWGS_MeOH_DME', 'PTM', 'AD'
        ]
    }
    
    # Define scenarios
    scenarios = ['Base_2024', 'Base_2030', 'Base_2050', 
                '2 degree_2030', '2 degree_2050', 
                '1.5 degree_2030', '1.5 degree_2050']
    
    print(f"\nCreating ranked cost analysis plots...")
    print(f"Fuel categories: {list(fuel_categories.keys())}")
    print(f"Scenarios: {scenarios}")
    
    # Create all figures for each category and scenario
    for category_name, techs in fuel_categories.items():
        print(f"\nProcessing {category_name}...")
        
        for scenario in scenarios:
            print(f"  Scenario: {scenario}")
            create_multi_technology_figure(results, category_name, techs, scenario, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nAnalysis complete! Generated two types of plots for each technology and scenario:")
    print("1. Ranked cost component bar charts (horizontal)")
    print("2. Ranked Monte Carlo distribution plots")

if __name__ == "__main__":
    main()