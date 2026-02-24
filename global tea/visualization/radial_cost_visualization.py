"""
Radial Stacked Bar Chart Visualization for Fuel Production Technologies

This script creates spider/radial charts showing cost components for:
1. Hydrogen production technologies across selected countries  
2. Kerosene production technologies across selected countries

Each country gets two charts - one for hydrogen, one for kerosene.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import seaborn as sns
from math import pi

def set_plot_style():
    """Set professional plot style for publication-quality figures"""
    plt.style.use('default')
    
    # Professional EES journal style with precise typography
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 12          # 11-12pt for labels
    mpl.rcParams['axes.titlesize'] = 15          # 14-16pt for titles
    mpl.rcParams['xtick.labelsize'] = 11         # 11-12pt for labels
    mpl.rcParams['ytick.labelsize'] = 10         # Slightly smaller for radial ticks
    mpl.rcParams['legend.fontsize'] = 10         # 10-11pt for legend
    mpl.rcParams['figure.titlesize'] = 15        # Consistent with axes titles
    
    # Professional EES journal appearance - minimal clutter
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['grid.linewidth'] = 0.3
    mpl.rcParams['grid.alpha'] = 0.2
    mpl.rcParams['grid.color'] = '#cccccc'
    mpl.rcParams['lines.linewidth'] = 1.2
    mpl.rcParams['patch.linewidth'] = 0.3
    mpl.rcParams['axes.linewidth'] = 0.6
    mpl.rcParams['axes.edgecolor'] = '#666666'
    mpl.rcParams['figure.dpi'] = 600  # High resolution for publication
    mpl.rcParams['figure.facecolor'] = 'none'  # Transparent figure background
    mpl.rcParams['axes.facecolor'] = 'none'  # Transparent axes background
    mpl.rcParams['savefig.facecolor'] = 'none'  # Transparent save background
    mpl.rcParams['savefig.edgecolor'] = 'none'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1
    mpl.rcParams['savefig.transparent'] = True  # Enable transparency by default
    
    # Remove axes spines for cleaner radial plots
    mpl.rcParams['axes.spines.left'] = False
    mpl.rcParams['axes.spines.bottom'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

def load_component_data(excel_file='output/lcox_results.xlsx'):
    """Load cost component data from Excel file"""
    print(f"Loading component data from {excel_file}")
    
    if not Path(excel_file).exists():
        raise FileNotFoundError(f"Excel file {excel_file} not found!")
    
    xlsx = pd.ExcelFile(excel_file)
    comp_sheets = [s for s in xlsx.sheet_names if s.startswith('Comp_')]
    
    if not comp_sheets:
        raise ValueError("No component sheets found in Excel file")
    
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

def get_country_names():
    """Return mapping from country codes to full names"""
    return {
        'AUS': 'Australia',
        'BRA': 'Brazil', 
        'CAN': 'Canada',
        'CHL': 'Chile',
        'CHN': 'China',
        'DEU': 'Germany',
        'DNK': 'Denmark',
        'EGY': 'Egypt',
        'ESP': 'Spain',
        'FRA': 'France',
        'GBR': 'United Kingdom',
        'IND': 'India',
        'IRN': 'Iran',
        'JPN': 'Japan',
        'KEN': 'Kenya',
        'MAR': 'Morocco',
        'MEX': 'Mexico',
        'NOR': 'Norway',
        'OMN': 'Oman',
        'PER': 'Peru',
        'RUS': 'Russia',
        'SAU': 'Saudi Arabia',
        'TUR': 'Turkey',
        'UAE': 'United Arab Emirates',
        'USA': 'United States',
        'ZAF': 'South Africa'
    }

def get_color_palettes():
    """Define colorblind-friendly, muted color palettes for EES journal standards"""
    # Colorblind-friendly palette with muted tones suitable for scientific publication
    component_colors = {
        'c_capex': '#1f77b4',         # Muted blue for CAPEX (primary cost)
        'c_om': '#aec7e8',            # Light blue for O&M/FOC
        'repex': '#ff7f0e',           # Muted orange for replacement costs
        'c_elec': '#ffbb78',          # Light orange for electricity
        'c_heat': '#d62728',          # Muted red for heat
        'c_ng': '#8c8c8c',            # Neutral gray for natural gas
        'c_bio': '#2ca02c',           # Muted green for biomass
        'c_pw': '#98df8a',            # Light green for pure water
        'c_iw': '#c5dbcb',            # Very light green for industrial water
        'c_h2': '#17becf',            # Teal for hydrogen
        'c_co2': '#dbdb8d',           # Muted yellow-green for CO2
        'c_h2_storage': '#9edae5',    # Light teal for H2 storage
        'c_co2_storage': '#f7b6d3',   # Light pink for CO2 storage
        'c_upgrade': '#9467bd',       # Muted purple for upgrading costs
        'c_other': '#e6e6e6'          # Light gray for aggregated minor components
    }
    
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
        'c_co2': r'CO$_2$',
        'c_h2_storage': r'H$_2$ Storage',
        'c_co2_storage': r'CO$_2$ Storage',
        'c_upgrade': 'Upgrading',
        'c_other': 'Other'
    }
    
    return component_colors, component_names

def define_technologies():
    """Define hydrogen and kerosene technologies"""
    hydrogen_techs = ['PEM', 'AE', 'SOEC', 'HTSE', 'CuCl', 'SMR_CCS', 'ATR_CCS', 'CLR', 'M_PYR', 'TG_CCS']
    kerosene_techs = ['SR_FT_kerosene', 'ST_FT_kerosene', 'TG_FT_kerosene', 'HTL', 'HVO_kerosene', 
                     'B_PYR_kerosene', 'RWGS_FT_kerosene', 'RWGS_MeOH_kerosene']
    
    # Technology display names - abbreviated to match aggregated plot
    tech_display_names = {
        'PEM': 'PEM',
        'AE': 'AE',
        'SOEC': 'SOEC',
        'HTSE': 'HTSE',
        'CuCl': 'CuCl',
        'SMR_CCS': 'SMR+CCS',
        'ATR_CCS': 'ATR+CCS',
        'CLR': 'CLR',
        'M_PYR': 'M_PYR',
        'TG_CCS': 'TG+CCS',
        'SR_FT_kerosene': 'SR-FT',
        'ST_FT_kerosene': 'ST-FT',
        'TG_FT_kerosene': 'TG-FT',
        'HTL': 'HTL',
        'HVO_kerosene': 'HVO',
        'B_PYR_kerosene': 'B_PYR',
        'RWGS_FT_kerosene': 'RWGS-FT',
        'RWGS_MeOH_kerosene': 'RWGS-MeOH'
    }
    
    return hydrogen_techs, kerosene_techs, tech_display_names

def find_component_sheet(component_data, tech, scenario='Base50'):
    """Find the appropriate component sheet for a technology and scenario"""
    # Try exact match first
    exact_sheet = f'Comp_{scenario}_{tech}'
    if exact_sheet in component_data:
        return exact_sheet
    
    # Try partial match for Excel sheet name truncation
    matching_sheets = [s for s in component_data.keys() 
                      if s.startswith(f'Comp_{scenario}_{tech}'[:20])]
    
    if matching_sheets:
        return matching_sheets[0]
    
    return None

def extract_country_data(component_data, country, tech, scenario='Base50'):
    """Extract cost component data for a specific country and technology"""
    sheet_name = find_component_sheet(component_data, tech, scenario)
    
    if not sheet_name:
        return None
    
    data = component_data[sheet_name]
    
    if country not in data.index:
        return None
    
    # Extract cost components for this country
    country_data = data.loc[country]
    
    # Filter out zero/negligible values and invalid data
    components = {}
    for comp, value in country_data.items():
        if pd.notna(value) and value > 0.001:  # Threshold for negligible values
            components[comp] = value
    
    return components if components else None

def aggregate_minor_components(components, threshold=0.01):
    """
    Aggregate minor cost components into 'Other' category to reduce visual clutter.
    
    Args:
        components: Dictionary of component names and values
        threshold: Minimum percentage of total cost to keep as separate component
    
    Returns:
        Dictionary with minor components aggregated into 'Other'
    """
    if not components:
        return components
    
    total_cost = sum(components.values())
    if total_cost == 0:
        return components
    
    # Calculate threshold value
    threshold_value = total_cost * threshold
    
    # Separate major and minor components
    major_components = {}
    minor_total = 0
    
    for comp, value in components.items():
        if value >= threshold_value:
            major_components[comp] = value
        else:
            minor_total += value
    
    # Add aggregated minor components if significant
    if minor_total > 0:
        major_components['c_other'] = minor_total
    
    return major_components

def create_radial_chart(data_dict, title, colors, component_names, output_path, max_radius=None):
    """
    Create a professional radial stacked bar chart optimized for EES journal subfigures.
    
    Features:
    - Colorblind-friendly palette with muted tones
    - Professional typography (Arial/Helvetica)
    - Minimal visual clutter and subtle gridlines
    - Component aggregation for clarity
    - Optimized for publication as subfigures
    """
    if not data_dict:
        print(f"No data available for {title}")
        return
    
    # Aggregate minor components to reduce visual clutter
    aggregated_data = {}
    for tech, components in data_dict.items():
        aggregated_data[tech] = aggregate_minor_components(components, threshold=0.03)
    
    data_dict = aggregated_data
    
    # Set up figure optimized for subfigures (smaller, publication-ready) with transparency
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    fig.patch.set_alpha(0.0)  # Make figure background transparent
    
    # Calculate technology positions
    techs = list(data_dict.keys())
    N = len(techs)
    
    if N == 0:
        print(f"No technologies available for {title}")
        return
    
    # Evenly distribute technologies around the circle with better spacing
    theta = np.linspace(0, 2 * pi, N, endpoint=False)
    
    # Calculate appropriate radius scaling
    if max_radius is None:
        all_totals = [sum(tech_data.values()) for tech_data in data_dict.values() if tech_data]
        max_radius = max(all_totals) * 1.05 if all_totals else 1.0
    
    # Track components for legend (only show used components)
    used_components = set()
    
    # Professional component ordering (most important first)
    component_order = ['c_capex', 'c_om', 'repex', 'c_elec', 'c_heat', 'c_bio', 
                      'c_ng', 'c_h2', 'c_co2', 'c_upgrade', 'c_h2_storage', 
                      'c_co2_storage', 'c_pw', 'c_iw', 'c_other']
    
    # Create radial bars with professional styling
    for i, (tech, components) in enumerate(data_dict.items()):
        if not components:
            continue
        
        # Order components consistently
        ordered_components = [(comp, components[comp]) for comp in component_order 
                            if comp in components and components[comp] > 0]
        
        # Create stacked segments
        bottom = 0
        bar_width = 2 * pi / N * 0.75  # Slightly narrower for cleaner appearance
        
        for comp, value in ordered_components:
            used_components.add(comp)
            color = colors.get(comp, '#cccccc')
            
            # Professional bar styling - minimal borders, clean appearance
            ax.bar(theta[i], value, width=bar_width, bottom=bottom,
                  color=color, alpha=0.9, edgecolor='white', linewidth=0.3)
            bottom += value
    
    # Professional chart styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Clean, readable technology labels positioned outside
    tech_display_names = define_technologies()[2]
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    
    # Position technology labels with optimal spacing
    label_radius = max_radius * 1.1  # Closer positioning since ylim ends at max_radius
    for i, tech in enumerate(techs):
        label = tech_display_names.get(tech, tech)
        
        # Clean label positioning without boxes (minimal clutter)
        ax.text(theta[i], label_radius, label,
                ha='center', va='center',
                fontsize=11, fontweight='normal',
                color='#333333')
    
    # Enhanced grid and axis styling - limit to accommodate labels
    ax.set_ylim(0, max_radius * 1.15)  # Minimal space for labels
    
    # Enhanced radial ticks for clarity
    n_ticks = 4 if max_radius < 0.5 else 5
    radial_ticks = np.linspace(0, max_radius, n_ticks + 1)[1:]
    ax.set_rticks(radial_ticks)
    ax.set_rlabel_position(135)  # Rotate labels to avoid overlap with bars
    ax.tick_params(axis='y', labelsize=9, colors='#666666')
    
    # Move EUR/kWh label to lower middle (270 degrees / bottom)
    ax.text(3*np.pi/2, max_radius * 0.3, 'EUR/kWh', 
           ha='center', va='center', 
           fontsize=10, fontweight='bold',
           color='#333333',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Enhanced grid styling with more visible inner lines
    ax.grid(True, alpha=0.4, linewidth=0.6, linestyle='-', color='#888888')
    ax.set_facecolor('none')  # Transparent background
    
    # Format radial labels professionally
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # Create clean, professional legend
    legend_elements = []
    for comp in component_order:
        if comp in used_components:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                                               facecolor=colors[comp], 
                                               alpha=0.9,
                                               edgecolor='none',
                                               label=component_names[comp]))
    
    # Position legend professionally for subfigures
    if legend_elements:
        legend = ax.legend(handles=legend_elements, 
                          loc='center left', 
                          bbox_to_anchor=(1.15, 0.5),
                          frameon=False,  # No frame for cleaner look
                          fontsize=10,
                          title='Cost Components',
                          title_fontsize=10,
                          handletextpad=0.4,
                          borderaxespad=0)
        
        # Style legend title
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_color('#333333')
    
    # Publication-quality output settings with transparency
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', 
               facecolor='none', edgecolor='none',
               format='png', transparent=True,
               pad_inches=0.1)
    plt.close()
    
    print(f"Saved professional radial chart: {output_path}")

def select_interesting_countries():
    """Select a diverse set of interesting countries for analysis"""
    # Countries selected for geographic, economic, and energy diversity
    countries = {
        'AUS': 'High renewable potential, mining economy',
        'SAU': 'Oil-rich, sunny climate, H2 ambitions', 
        'DEU': 'Industrial economy, renewable transition',
        'BRA': 'Large biomass resources, diverse energy mix',
        'JPN': 'Advanced technology, limited resources',
        'KEN': 'Developing economy, good solar/geothermal',
        'CAN': 'Cold climate, hydroelectric resources',
        'UAE': 'Desert climate, oil-rich, solar ambitions'
    }
    
    return countries

def main():
    """
    Main execution function for professional EES journal-quality radial cost visualization.
    
    Generates publication-ready radial charts optimized for subfigures with:
    - Colorblind-friendly muted color palette
    - Professional typography (Arial/Helvetica)
    - Minimal visual clutter and subtle gridlines
    - Component aggregation for clarity
    - High-resolution output (600 DPI)
    """
    set_plot_style()
    
    print("=== Professional Radial Cost Visualization (EES Journal Quality) ===")
    print("Features: Colorblind-friendly palette, minimal clutter, optimized for subfigures")
    
    # Load data
    print("\n1. Loading component data...")
    component_data = load_component_data()
    
    # Get professional styling
    colors, component_names = get_color_palettes()
    
    # Define technologies with abbreviated names
    hydrogen_techs, kerosene_techs, tech_display_names = define_technologies()
    
    # Select countries
    selected_countries = select_interesting_countries()
    country_names = get_country_names()
    
    # Create output directory
    output_dir = Path('figures/radial_charts')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Scenarios to analyze (BAU 2050 and 1.5°C 2050)
    scenarios = {
        'Base50': 'BAU 2050',
        '15deg50': '1.5°C 2050'
    }
    
    print(f"\n2. Creating radial charts for scenarios: {list(scenarios.values())}")
    print(f"Selected countries: {list(selected_countries.keys())}")
    
    # Process each scenario
    for scenario_code, scenario_name in scenarios.items():
        print(f"\n=== Processing Scenario: {scenario_name} ({scenario_code}) ===")
        
        # Process each country
        for country_code, description in selected_countries.items():
            country_name = country_names.get(country_code, country_code)
            print(f"\n--- Processing {country_name} ({country_code}) for {scenario_name} ---")
            print(f"Description: {description}")
            
            # Collect hydrogen technology data
            hydrogen_data = {}
            for tech in hydrogen_techs:
                components = extract_country_data(component_data, country_code, tech, scenario_code)
                if components:
                    hydrogen_data[tech] = components
            
            # Collect kerosene technology data  
            kerosene_data = {}
            for tech in kerosene_techs:
                components = extract_country_data(component_data, country_code, tech, scenario_code)
                if components:
                    kerosene_data[tech] = components
            
            print(f"Found data for {len(hydrogen_data)} hydrogen technologies")
            print(f"Found data for {len(kerosene_data)} kerosene technologies")
            
            # Calculate common max radius for both charts (for comparison)
            all_totals = []
            for data_dict in [hydrogen_data, kerosene_data]:
                for tech_data in data_dict.values():
                    if tech_data:
                        total = sum(tech_data.values())
                        all_totals.append(total)
            
            max_radius = max(all_totals) * 1.1 if all_totals else 1.0
            
            # Override max_radius for 2050 scenarios with specific limits
            if scenario_code in ['Base50', '15deg50']:  # 2050 scenarios
                h2_max_radius = 0.15  # Set hydrogen technologies to 0-0.15 for 2050
                kerosene_max_radius = 0.35  # Set kerosene technologies to 0-0.35 for 2050
                print(f"Applying 2050 axis limits - Hydrogen: 0-{h2_max_radius}, Kerosene: 0-{kerosene_max_radius}")
            else:
                h2_max_radius = max_radius
                kerosene_max_radius = max_radius
            
            # Create hydrogen chart
            if hydrogen_data:
                h2_title = f'Hydrogen Production Technologies\n{country_name} ({scenario_name})'
                h2_output = output_dir / f'hydrogen_radial_{country_code}_{scenario_code}.png'
                create_radial_chart(hydrogen_data, h2_title, colors, component_names, 
                                  h2_output, h2_max_radius)
            else:
                print(f"No hydrogen data available for {country_name} in {scenario_name}")
            
            # Create kerosene chart
            if kerosene_data:
                kerosene_title = f'Kerosene Production Technologies\n{country_name} ({scenario_name})'
                kerosene_output = output_dir / f'kerosene_radial_{country_code}_{scenario_code}.png'
                create_radial_chart(kerosene_data, kerosene_title, colors, component_names, 
                                   kerosene_output, kerosene_max_radius)
            else:
                print(f"No kerosene data available for {country_name} in {scenario_name}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"All charts saved to: {output_dir}")
    print("\nGenerated files:")
    for scenario_code in scenarios.keys():
        for country_code in selected_countries.keys():
            print(f"- hydrogen_radial_{country_code}_{scenario_code}.png")
            print(f"- kerosene_radial_{country_code}_{scenario_code}.png")

if __name__ == '__main__':
    main() 