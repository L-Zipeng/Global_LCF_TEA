#!/usr/bin/env python3
"""
The methodology uses existing LCOX results from Excel output files and performs
sensitivity analysis by decreasing component costs by -20% and WACC by -20%.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def set_plot_style():
    """Set publication-quality plot style"""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

def load_data(excel_file='output/lcox_results.xlsx'):
    """
    Load data from the Excel file containing LCOX results and component data
    (Using the same approach as the working sensitivity analysis script)
    
    Returns:
        dict: Dictionary containing the loaded data
    """
    print(f"Loading data from {excel_file}")
    
    # Read list of sheets
    xlsx = pd.ExcelFile(excel_file)
    sheets = xlsx.sheet_names
    
    # Identify LCOX and component sheets
    lcox_sheets = [s for s in sheets if s.startswith('LCOX_')]
    comp_sheets = [s for s in sheets if s.startswith('Comp_')]
    
    if not lcox_sheets:
        raise ValueError("No LCOX sheets found in the Excel file")
    
    print(f"Found {len(comp_sheets)} component sheets:")
    for sheet in comp_sheets:
        print(f"  {sheet}")
    
    # Load basic data from LCOX sheets
    scenarios = {}
    for sheet in lcox_sheets:
        scenario = sheet.replace('LCOX_', '')
        df = pd.read_excel(excel_file, sheet_name=sheet)
        df.set_index(df.columns[0], inplace=True)  # Set first column as index (countries)
        scenarios[scenario] = df
    
    # Process component data by technology and scenario
    component_data = {}
    
    for sheet in comp_sheets:
        parts = sheet.split('_', 2)
        if len(parts) < 3:
            continue
            
        scenario = parts[1]
        tech_truncated = parts[2]
        
        print(f"\nProcessing sheet: {sheet}")
        print(f"  Scenario: {scenario}, Truncated tech: {tech_truncated}")
        
        # Handle Excel sheet name truncation - try to match with full technology names
        tech = tech_truncated  # Default to truncated name
        
        # Try to match truncated names with full technology names from LCOX data
        if scenarios:  # If we have LCOX data loaded
            first_scenario_data = list(scenarios.values())[0]
            full_tech_names = first_scenario_data.columns.tolist()
            
            # Special debug for RWGS_MeOH technologies
            if 'RWGS_MeOH' in tech_truncated:
                print(f"  Debugging RWGS_MeOH matching:")
                print(f"    Truncated name: {tech_truncated}")
                print(f"    Available full names containing 'RWGS_MeOH': {[name for name in full_tech_names if 'RWGS_MeOH' in name]}")
            
            # Find the best match for the truncated name
            best_match = None
            best_score = 0
            
            # Special handling for RWGS_MeOH technologies to handle Excel truncation
            if tech_truncated.startswith('RWGS_MeOH'):
                # Map common truncations to full names
                rwgs_meoh_mappings = {
                    'RWGS_MeOH_metha': 'RWGS_MeOH_methanol',
                    'RWGS_MeOH_keros': 'RWGS_MeOH_kerosene',
                    'RWGS_MeOH_methano': 'RWGS_MeOH_methanol',
                    'RWGS_MeOH_kerosen': 'RWGS_MeOH_kerosene',
                    'RWGS_MeOH_DME': 'RWGS_MeOH_DME',
                    'RWGS_MeOH': 'RWGS_MeOH_methanol'  # Default to methanol if ambiguous
                }
                
                if tech_truncated in rwgs_meoh_mappings:
                    mapped_name = rwgs_meoh_mappings[tech_truncated]
                    if mapped_name in full_tech_names:
                        best_match = mapped_name
                        best_score = 1.0
                        print(f"  Applied RWGS_MeOH mapping: '{tech_truncated}' -> '{best_match}'")
            
            # If no special mapping found, use general matching algorithm
            if best_score == 0:
                for full_name in full_tech_names:
                    # Calculate similarity score
                    if full_name.startswith(tech_truncated):
                        score = len(tech_truncated) / len(full_name)
                        if score > best_score:
                            best_score = score
                            best_match = full_name
                    elif tech_truncated in full_name:
                        score = len(tech_truncated) / len(full_name) * 0.8  # Slightly lower score for partial matches
                        if score > best_score:
                            best_score = score
                            best_match = full_name
            
            if best_match and best_score > 0.5:  # Only use if we have a good match
                tech = best_match
                print(f"  Matched truncated '{tech_truncated}' to full name '{tech}' (score: {best_score:.2f})")
            else:
                print(f"  Warning: Could not find good match for truncated name '{tech_truncated}' (best score: {best_score:.2f})")
                if best_match:
                    print(f"    Best candidate was: '{best_match}'")
        
        # Load raw component data
        df = pd.read_excel(excel_file, sheet_name=sheet)
        df.set_index(df.columns[0], inplace=True)  # Set first column as index (countries)
        
        # Initialize nested dictionaries if they don't exist
        if tech not in component_data:
            component_data[tech] = {}
        if scenario not in component_data[tech]:
            component_data[tech][scenario] = df
        
        print(f"  Stored component data for technology: '{tech}'")
        
        # Special handling: if this is a RWGS_MeOH_ truncated sheet, also create entries for 
        # the missing RWGS_MeOH technologies to ensure they have component data
        if tech_truncated == 'RWGS_MeOH_' and tech == 'RWGS_MeOH_DME':
            # Create component data for RWGS_MeOH_methanol and RWGS_MeOH_kerosene
            # using the same data as RWGS_MeOH_DME (since they're all RWGS_MeOH variants)
            for missing_tech in ['RWGS_MeOH_methanol', 'RWGS_MeOH_kerosene']:
                if missing_tech not in component_data:
                    component_data[missing_tech] = {}
                if scenario not in component_data[missing_tech]:
                    component_data[missing_tech][scenario] = df.copy()  # Use copy to avoid shared reference
                    print(f"  Also stored component data for missing technology: '{missing_tech}'")
    
    # Get countries and technologies from the first LCOX sheet
    first_lcox = scenarios[list(scenarios.keys())[0]]
    countries = first_lcox.index.tolist()
    technologies = first_lcox.columns.tolist()
    
    print(f"\nFinal component data summary:")
    for tech, scenarios_dict in component_data.items():
        print(f"  {tech}: {list(scenarios_dict.keys())}")
    
    # Specifically check for RWGS_MeOH technologies
    print(f"\nRWGS_MeOH technology check:")
    rwgs_meoh_techs = [tech for tech in component_data.keys() if 'RWGS_MeOH' in tech]
    print(f"  Found RWGS_MeOH technologies in component data: {rwgs_meoh_techs}")
    
    # Try to load WACC data if available
    wacc_data = {}
    
    # Try to load country-specific WACC factors
    try:
        if 'f_wacc_c' in sheets:
            wacc_data['f_wacc_c'] = pd.read_excel(excel_file, sheet_name='f_wacc_c')
            print("Found f_wacc_c in output file")
    except Exception as e:
        print(f"Error loading country WACC factors from output file: {e}")
    
    # Try to load technology-specific WACC factors
    try:
        if 'f_wacc_t' in sheets:
            wacc_data['f_wacc_t'] = pd.read_excel(excel_file, sheet_name='f_wacc_t', index_col=0)
            print("Found f_wacc_t in output file")
    except Exception as e:
        print(f"Error loading technology WACC factors from output file: {e}")
    
    # If WACC data not found in output file, try to load from TEA input files
    if 'f_wacc_c' not in wacc_data:
        wacc_data.update(load_wacc_from_input_files())
    
    data = {
        'scenarios': scenarios,
        'component_data': component_data,
        'countries': countries,
        'technologies': technologies
    }
    
    # Add WACC data if available
    for key, value in wacc_data.items():
        data[key] = value
    
    return data

def load_wacc_from_input_files():
    """
    Load WACC data from TEA input files if not found in output file
    """
    wacc_data = {}
    
    # Try different possible locations for TEA input file
    input_files = [
        'input/TEA input.xlsx',
        'data/TEA input.xlsx', 
        'TEA input.xlsx',
        'input/tea_input.xlsx',
        'data/tea_input.xlsx'
    ]
    
    for input_file in input_files:
        try:
            if os.path.exists(input_file):
                print(f"Trying to load WACC data from {input_file}")
                xlsx = pd.ExcelFile(input_file)
                sheets = xlsx.sheet_names
                
                # Load f_wacc_c (country WACC factors)
                if 'f_wacc_c' in sheets:
                    wacc_data['f_wacc_c'] = pd.read_excel(input_file, sheet_name='f_wacc_c')
                    print(f"Loaded f_wacc_c from {input_file}")
                    print(f"f_wacc_c shape: {wacc_data['f_wacc_c'].shape}")
                    print(f"f_wacc_c columns: {wacc_data['f_wacc_c'].columns.tolist()}")
                    print(f"f_wacc_c first few rows:")
                    print(wacc_data['f_wacc_c'].head())
                
                # Load f_wacc_t (technology WACC factors)
                if 'f_wacc_t' in sheets:
                    wacc_data['f_wacc_t'] = pd.read_excel(input_file, sheet_name='f_wacc_t', index_col=0)
                    print(f"Loaded f_wacc_t from {input_file}")
                
                if wacc_data:  # If we found any WACC data, stop searching
                    break
                    
        except Exception as e:
            print(f"Error loading from {input_file}: {e}")
            continue
    
    if not wacc_data:
        print("Warning: Could not find WACC data in any input files")
        print("Searched in:", input_files)
    
    return wacc_data

def get_ordered_technologies(available_techs):
    """
    Orders technologies according to the complete list used in aggregatedplot.py
    Organized by technology groups: Hydrogen, Kerosene, Diesel, Other Fuels
    
    Note: Base technology names (e.g., ST_FT) are excluded if fuel-specific variants exist
    (e.g., ST_FT_kerosene, ST_FT_diesel) to avoid duplication
    """
    # Complete technology order - EXCLUDES base technology names when fuel-specific variants exist
    complete_tech_order = [
        # Hydrogen technologies (Green, Pink, Blue, Turquoise, Bio)
        'AE', 'PEM', 'SOEC',           # Green H2
        'HTSE', 'CuCl',                # Pink H2  
        'SMR_CCS', 'ATR_CCS', 'CLR',   # Blue H2
        'M_PYR',                       # Turquoise H2
        'TG_CCS',                      # Bio H2
        
        # Single-fuel technologies (no fuel variants)
        'HTL',                         # Hydrothermal liquefaction
        
        # Fuel-specific technologies (ONLY fuel-specific, NOT base names)
        'SR_FT_kerosene', 'ST_FT_kerosene',              # Solar kerosene
        'TG_FT_kerosene', 'HVO_kerosene', 'B_PYR_kerosene',  # Bio kerosene
        'RWGS_FT_kerosene', 'RWGS_MeOH_kerosene',        # Power-to-Liquid kerosene
        'SR_FT_diesel', 'ST_FT_diesel',                  # Solar Diesel
        'TG_FT_diesel', 'HVO_diesel',                    # Bio Diesel
        'RWGS_FT_diesel',                                # Power-to-Liquid Diesel
        'RWGS_MeOH_methanol', 'RWGS_MeOH_DME',           # Other methanol products
        
        # Other fuels (Ammonia, Methane)
        'HB',                          # Ammonia
        'PTM', 'AD',                   # Methane
        
        # Additional technologies that might appear in data
        'DAC'                          # Direct Air Capture
    ]
    
    # Filter to only include technologies available in the data
    ordered_techs = [tech for tech in complete_tech_order if tech in available_techs]
    
    # Add any technologies that might be in results but not in the predefined order
    for tech in available_techs:
        if tech not in ordered_techs:
            ordered_techs.append(tech)
            
    return ordered_techs

def get_technology_categories():
    """Define technology categories with fuel-specific technologies (no base aggregated names)"""
    categories = {
        'Hydrogen': ['AE', 'PEM', 'SOEC', 'HTSE', 'CuCl', 'SMR_CCS', 'ATR_CCS', 'CLR', 'M_PYR', 'TG_CCS'],
        'Single-Fuel': ['HTL'],
        'Solar Kerosene': ['SR_FT_kerosene', 'ST_FT_kerosene'],
        'Solar Diesel': ['SR_FT_diesel', 'ST_FT_diesel'],
        'Bio Kerosene': ['TG_FT_kerosene', 'HVO_kerosene', 'B_PYR_kerosene'],
        'Bio Diesel': ['TG_FT_diesel', 'HVO_diesel'],
        'Power-to-X Kerosene': ['RWGS_FT_kerosene', 'RWGS_MeOH_kerosene'],
        'Power-to-X Diesel': ['RWGS_FT_diesel'],
        'Power-to-X Other': ['RWGS_MeOH_methanol', 'RWGS_MeOH_DME'],
        'Other Fuels': ['HB', 'PTM', 'AD'],
        'DAC': ['DAC']
    }
    return categories

def get_component_colors_and_names():
    """Get consistent component colors and names (matches aggregatedplot.py)"""
    colors = {
        'c_capex': '#3182bd',         # Blue for CAPEX
        'c_om': '#6baed6',            # Light blue for O&M
        'repex': '#fd8d3c',           # Orange for replacement costs
        'c_elec': '#f0eebb',          # Light yellow for electricity
        'c_heat': '#9b3a4d',          # Dark red for heat
        'c_ng': '#636363',            # Dark gray for natural gas
        'c_bio': '#31a354',           # Green for biomass
        'c_pw': '#74c476',            # Light green for process water
        'c_iw': '#a1d99b',            # Pale green for industrial water
        'c_h2': '#9ecae1',            # Pale blue for hydrogen
        'c_co2': '#f4c28f',           # Orange-tan for CO2
        'c_h2_storage': '#c6dbef',    # Very light blue for H2 storage
        'c_co2_storage': '#fb6a4a',   # Light red for CO2 storage
        'c_upgrade': '#9467bd'        # Purple for upgrading costs
    }
    
    component_names = {
        'c_capex': 'CAPEX',
        'c_om': 'O&M',
        'repex': 'Replacement',
        'c_elec': 'Electricity',
        'c_heat': 'Heat',
        'c_ng': 'Natural Gas',
        'c_bio': 'Biomass',
        'c_pw': 'Process Water',
        'c_iw': 'Industrial Water',
        'c_h2': 'Hydrogen',
        'c_co2': 'CO$_2$',
        'c_h2_storage': 'H2 Storage',
        'c_co2_storage': 'CO$_2$ Storage',
        'c_upgrade': 'Upgrading'
    }
    
    return colors, component_names

def calculate_component_sensitivity(data, scenario='Base24', countries=None, variation=0.2):
    """
    Calculate sensitivity of technologies to different cost components
    
    Args:
        data: Loaded data dictionary
        scenario: Scenario to analyze
        countries: List of countries to analyze (None for all)
        variation: Variation percentage (0.2 = -20% decrease)
    
    Returns:
        DataFrame: Technologies vs Components sensitivity matrix
    """
    print(f"Calculating component sensitivity for {scenario}")
    
    if scenario not in data['scenarios']:
        print(f"Scenario {scenario} not found in data")
        return pd.DataFrame()
    
    lcox_data = data['scenarios'][scenario]
    available_techs = lcox_data.columns.tolist()
    
    # Order technologies consistently
    ordered_techs = get_ordered_technologies(available_techs)
    
    if countries is None:
        countries = lcox_data.index.tolist()
    
    # Get all possible components from component data
    all_components = set()
    for tech in ordered_techs:
        if tech in data['component_data'] and scenario in data['component_data'][tech]:
            comp_df = data['component_data'][tech][scenario]
            all_components.update(comp_df.columns.tolist())
    
    # Filter to key components
    key_components = ['c_capex', 'c_om', 'repex', 'c_elec', 'c_heat', 'c_h2', 'c_ng', 'c_bio', 'c_co2']
    key_components = [comp for comp in key_components if comp in all_components]
    
    # Initialize sensitivity matrix
    sensitivity_matrix = np.zeros((len(ordered_techs), len(key_components)))
    
    for i, tech in enumerate(ordered_techs):
        print(f"  Analyzing {tech}...")
        
        # Calculate average sensitivity across countries
        tech_sensitivities = {comp: [] for comp in key_components}
        
        for country in countries:
            try:
                # Get base LCOX
                if country not in lcox_data.index or tech not in lcox_data.columns:
                    continue
                    
                base_lcox = lcox_data.loc[country, tech]
                if pd.isna(base_lcox) or base_lcox <= 0:
                    continue
                
                # Get component data
                if tech not in data['component_data'] or scenario not in data['component_data'][tech]:
                    continue
                
                comp_df = data['component_data'][tech][scenario]
                if country not in comp_df.index:
                    continue
                
                # Calculate sensitivity for each component
                for j, component in enumerate(key_components):
                    if component not in comp_df.columns:
                        continue
                    
                    comp_value = comp_df.loc[country, component]
                    if pd.isna(comp_value) or comp_value <= 0:
                        continue
                    
                    # Calculate impact of -variation% decrease in this component
                    negative_impact = -(comp_value * variation) / base_lcox * 100  # % impact on total LCOX (negative = decrease)
                    tech_sensitivities[component].append(abs(negative_impact))  # Store absolute value for visualization
                    
            except Exception as e:
                continue
        
        # Calculate median sensitivity for each component (consistent with global cost plots)
        for j, component in enumerate(key_components):
            if tech_sensitivities[component]:
                sensitivity_matrix[i, j] = np.median(tech_sensitivities[component])
    
    return pd.DataFrame(sensitivity_matrix, index=ordered_techs, columns=key_components)

def get_dominant_components_by_scenario(data, scenarios=None, countries=None):
    """
    Get dominant cost component for each technology across scenarios
    
    Returns:
        dict: Technologies vs Scenarios with dominant components and impacts
    """
    print("Analyzing dominant components across scenarios")
    
    if scenarios is None:
        scenarios = list(data['scenarios'].keys())
    
    if countries is None:
        # Use representative countries for analysis
        countries = ['USA', 'DEU', 'CHN', 'BRA', 'AUS']
        # Filter to countries that exist in data
        available_countries = data['countries']
        countries = [c for c in countries if c in available_countries]
    
    all_technologies = set()
    for scenario in scenarios:
        if scenario in data['scenarios']:
            all_technologies.update(data['scenarios'][scenario].columns.tolist())
    
    ordered_techs = get_ordered_technologies(list(all_technologies))
    dominant_data = {}
    
    for scenario in scenarios:
        print(f"  Processing {scenario}...")
        scenario_dominants = {}
        
        if scenario not in data['scenarios']:
            continue
            
        lcox_data = data['scenarios'][scenario]
        
        for tech in ordered_techs:
            if tech not in lcox_data.columns:
                continue
            
            tech_results = []
            
            # Calculate across multiple countries and average
            for country in countries:
                if country not in lcox_data.index:
                    continue
                    
                try:
                    base_lcox = lcox_data.loc[country, tech]
                    if pd.isna(base_lcox) or base_lcox <= 0:
                        continue
                    
                    # Get component data
                    if tech not in data['component_data'] or scenario not in data['component_data'][tech]:
                        continue
                    
                    comp_df = data['component_data'][tech][scenario]
                    if country not in comp_df.index:
                        continue
                    
                    # Find component with highest contribution
                    components = comp_df.loc[country]
                    components = components[components > 0]  # Only positive components
                    
                    if len(components) == 0:
                        continue
                    
                    # Get dominant component and its impact
                    dominant_comp = components.idxmax()
                    max_value = components.max()
                    impact_percent = (max_value / base_lcox) * 100
                    
                    tech_results.append((dominant_comp, impact_percent))
                    
                except Exception as e:
                    continue
            
            if tech_results:
                # Use most common dominant component, or highest impact if tied
                comp_impacts = {}
                for comp, impact in tech_results:
                    if comp not in comp_impacts:
                        comp_impacts[comp] = []
                    comp_impacts[comp].append(impact)
                
                # Get component with highest median impact (consistent with global cost plots)
                median_impacts = {comp: np.median(impacts) for comp, impacts in comp_impacts.items()}
                dominant_comp = max(median_impacts.keys(), key=lambda x: median_impacts[x])
                median_impact = median_impacts[dominant_comp]
                
                scenario_dominants[tech] = {
                    'component': dominant_comp,
                    'impact': median_impact
                }
        
        dominant_data[scenario] = scenario_dominants
    
    return dominant_data

def get_country_risk_groups_from_wacc(data, base_wacc=0.08):
    """Define country risk groups based on actual country WACCs calculated from f_wacc_c factors"""
    
    if 'f_wacc_c' not in data:
        print("Warning: f_wacc_c data not found, using fallback arbitrary groups")
        return {
            'Low Risk': ['USA', 'DEU', 'GBR', 'FRA', 'JPN', 'CHE', 'NLD', 'CAN', 'AUS', 'NOR', 'SWE', 'DNK'],
            'Medium Risk': ['ESP', 'ITA', 'KOR', 'SGP', 'PRT', 'CZE', 'POL', 'HUN', 'CHL', 'URY'],
            'High Risk': ['CHN', 'RUS', 'BRA', 'MEX', 'TUR', 'ZAF', 'ARG', 'COL', 'PER', 'THA', 'MYS'],
            'Very High Risk': ['IND', 'IDN', 'VNM', 'PHL', 'EGY', 'MAR', 'NGA', 'PAK', 'BGD', 'IRN']
        }
    
    # Load WACC factor data
    wacc_df = data['f_wacc_c']
    
    # Use ISO codes as index instead of country names to match LCOX data format
    if wacc_df.shape[1] > 1:
        # Check if we have ISO codes column for proper matching with LCOX data
        if 'ISO_A3_EH' in wacc_df.columns and 'f_wacc_c' in wacc_df.columns:
            # Set ISO codes as index and use f_wacc_c values
            wacc_df = wacc_df.set_index('ISO_A3_EH')
            wacc_factors = wacc_df['f_wacc_c']
        else:
            # Fallback: use first column as index
            wacc_df = wacc_df.set_index(wacc_df.columns[0])
            
            # Look for the actual WACC factor column (should be named 'f_wacc_c')
            if 'f_wacc_c' in wacc_df.columns:
                wacc_factors = wacc_df['f_wacc_c']  # Take the f_wacc_c column specifically
            else:
                # Fallback: take the last column (assuming it's the numeric data)
                wacc_factors = wacc_df.iloc[:, -1]
    else:
        # If already indexed properly
        wacc_factors = wacc_df.iloc[:, 0] if wacc_df.shape[1] == 1 else wacc_df
    
    # Debug country names and values  
    print(f"WACC factors index (countries): {wacc_factors.index[:10].tolist()}...")
    print(f"WACC factors values sample: {wacc_factors.head().tolist()}")
    print(f"Using {'ISO codes' if 'ISO_A3_EH' in data['f_wacc_c'].columns else 'country names'} for matching with LCOX data")
    
    # Convert to numeric values (in case they were loaded as strings)
    print(f"WACC factors data type before conversion: {wacc_factors.dtype}")
    wacc_factors = pd.to_numeric(wacc_factors, errors='coerce')
    print(f"WACC factors data type after conversion: {wacc_factors.dtype}")
    
    # Check how many values became NaN after conversion
    nan_count = wacc_factors.isna().sum()
    print(f"Number of NaN values after numeric conversion: {nan_count} out of {len(wacc_factors)}")
    
    # Remove any NaN values (including those that couldn't be converted to numeric)
    wacc_factors_before_dropna = len(wacc_factors)
    wacc_factors = wacc_factors.dropna()
    wacc_factors_after_dropna = len(wacc_factors)
    
    print(f"WACC factors count: {wacc_factors_before_dropna} -> {wacc_factors_after_dropna} after dropping NaN")
    
    if len(wacc_factors) == 0:
        print("Warning: No valid WACC factor data found, using fallback groups")
        print("This usually means country names don't match between f_wacc_c and LCOX data")
        return get_country_risk_groups()
    
    # Show available countries in LCOX data for comparison
    lcox_countries = data['countries'][:10]  # First 10 countries from LCOX data
    print(f"Sample LCOX countries: {lcox_countries}")
    print(f"Sample WACC countries: {wacc_factors.index[:10].tolist()}")
    
    # Show some sample values for debugging
    print(f"Sample WACC factors:")
    sample_factors = wacc_factors.head(5)
    for country, factor in sample_factors.items():
        print(f"  {country}: {factor} (type: {type(factor)})")
    
    # Calculate actual country WACCs: base_wacc * f_wacc_c
    print(f"Calculating country WACCs using base WACC of {base_wacc*100:.1f}%")
    country_waccs = wacc_factors * base_wacc
    
    print(f"Country WACC range: {country_waccs.min()*100:.1f}% to {country_waccs.max()*100:.1f}%")
    
    # Create quartile-based risk groups based on actual WACC values
    # Higher WACC = Higher Risk
    q1 = country_waccs.quantile(0.25)
    q2 = country_waccs.quantile(0.50)
    q3 = country_waccs.quantile(0.75)
    
    risk_groups = {
        'Low Risk (WACC < {:.1f}%)'.format(q1*100): [],
        'Medium-Low Risk ({:.1f}%-{:.1f}%)'.format(q1*100, q2*100): [],
        'Medium-High Risk ({:.1f}%-{:.1f}%)'.format(q2*100, q3*100): [],
        'High Risk (WACC > {:.1f}%)'.format(q3*100): []
    }
    
    for country, country_wacc in country_waccs.items():
        if country_wacc <= q1:
            risk_groups['Low Risk (WACC < {:.1f}%)'.format(q1*100)].append(country)
        elif country_wacc <= q2:
            risk_groups['Medium-Low Risk ({:.1f}%-{:.1f}%)'.format(q1*100, q2*100)].append(country)
        elif country_wacc <= q3:
            risk_groups['Medium-High Risk ({:.1f}%-{:.1f}%)'.format(q2*100, q3*100)].append(country)
        else:
            risk_groups['High Risk (WACC > {:.1f}%)'.format(q3*100)].append(country)
    
    # Filter out empty groups
    risk_groups = {k: v for k, v in risk_groups.items() if len(v) > 0}
    
    print(f"Created WACC-based risk groups:")
    for group, countries in risk_groups.items():
        print(f"  {group}: {len(countries)} countries - {countries[:5]}{'...' if len(countries) > 5 else ''}")
    
    # Also show some example country WACCs for verification
    print(f"Example country WACCs:")
    sample_countries = country_waccs.head(5)
    for country, wacc in sample_countries.items():
        factor = wacc_factors[country]
        print(f"  {country}: factor={factor:.2f} → WACC={wacc*100:.1f}%")
    
    return risk_groups

def get_country_risk_groups():
    """Fallback function for arbitrary risk groups (kept for compatibility)"""
    return {
        'Low Risk': ['USA', 'DEU', 'GBR', 'FRA', 'JPN', 'CHE', 'NLD', 'CAN', 'AUS', 'NOR', 'SWE', 'DNK'],
        'Medium Risk': ['ESP', 'ITA', 'KOR', 'SGP', 'PRT', 'CZE', 'POL', 'HUN', 'CHL', 'URY'],
        'High Risk': ['CHN', 'RUS', 'BRA', 'MEX', 'TUR', 'ZAF', 'ARG', 'COL', 'PER', 'THA', 'MYS'],
        'Very High Risk': ['IND', 'IDN', 'VNM', 'PHL', 'EGY', 'MAR', 'NGA', 'PAK', 'BGD', 'IRN']
    }

def analyze_country_risk_impact(data, scenario='Base24'):
    """
    Analyze impact of country WACC groups (calculated from f_wacc_c factors × 8% base WACC) on technology LCOX
    
    Returns:
        DataFrame: Technologies vs WACC Groups impact matrix
    """
    print(f"Analyzing country risk impact for {scenario}")
    
    if scenario not in data['scenarios']:
        return pd.DataFrame()
    
    lcox_data = data['scenarios'][scenario]
    available_techs = lcox_data.columns.tolist()
    ordered_techs = get_ordered_technologies(available_techs)
    
    risk_groups = get_country_risk_groups_from_wacc(data)
    
    # Calculate LCOX for all countries first
    risk_impact_matrix = np.zeros((len(ordered_techs), len(risk_groups)))
    group_names = list(risk_groups.keys())
    
    for i, tech in enumerate(ordered_techs):
        if tech not in lcox_data.columns:
            continue
            
        # Get LCOX values for this technology across all countries
        tech_data = lcox_data[tech].dropna()
        if len(tech_data) == 0:
            continue
            
        # Calculate global median for reference (consistent with global cost plots)
        global_median = tech_data.median()
        
        for j, (group_name, countries) in enumerate(risk_groups.items()):
            # Get LCOX values for countries in this risk group
            group_countries = [c for c in countries if c in tech_data.index]
            
            if group_countries:
                group_values = tech_data[group_countries]
                group_median = group_values.median()
                
                # Calculate percentage difference from global median
                if global_median > 0:
                    risk_impact_matrix[i, j] = ((group_median / global_median) - 1) * 100
    
    return pd.DataFrame(risk_impact_matrix, index=ordered_techs, columns=group_names)

def get_regional_mapping():
    """Define comprehensive regional mapping for WACC analysis with all available countries"""
    return {
        # North America (16 countries, added GRL here)
        'USA': 'North America', 'CAN': 'North America', 'MEX': 'North America',
        'GTM': 'North America', 'HND': 'North America', 'SLV': 'North America', 
        'NIC': 'North America', 'CRI': 'North America', 'PAN': 'North America',
        'BLZ': 'North America', 'JAM': 'North America', 'DOM': 'North America',
        'HTI': 'North America', 'CUB': 'North America', 'BHS': 'North America',
        'GRL': 'North America',
        
        # Europe (40 countries, added MLT)
        'DEU': 'Europe', 'FRA': 'Europe', 'GBR': 'Europe', 'ITA': 'Europe', 'ESP': 'Europe',
        'NLD': 'Europe', 'BEL': 'Europe', 'SWE': 'Europe', 'NOR': 'Europe', 'DNK': 'Europe',
        'CHE': 'Europe', 'AUT': 'Europe', 'POL': 'Europe', 'PRT': 'Europe', 'IRL': 'Europe',
        'FIN': 'Europe', 'EST': 'Europe', 'LVA': 'Europe', 'LTU': 'Europe', 'CZE': 'Europe',
        'SVK': 'Europe', 'HUN': 'Europe', 'SVN': 'Europe', 'HRV': 'Europe', 'BIH': 'Europe',
        'SRB': 'Europe', 'MNE': 'Europe', 'MKD': 'Europe', 'ALB': 'Europe', 'BGR': 'Europe',
        'ROU': 'Europe', 'MDA': 'Europe', 'UKR': 'Europe', 'BLR': 'Europe', 'RUS': 'Europe',
        'GRC': 'Europe', 'CYP': 'Europe', 'LUX': 'Europe', 'ISL': 'Europe', 'MLT': 'Europe',
        
        # Asia Developed (4 countries, added SGP)
        'JPN': 'Asia Developed', 'KOR': 'Asia Developed', 'BRN': 'Asia Developed', 'SGP': 'Asia Developed',
        
        # Asia Developing (28 countries, added AFG, BTN, MDV, TKM, TLS)
        'CHN': 'Asia Developing', 'IND': 'Asia Developing', 'IDN': 'Asia Developing', 'THA': 'Asia Developing',
        'MYS': 'Asia Developing', 'PHL': 'Asia Developing', 'VNM': 'Asia Developing', 'KHM': 'Asia Developing',
        'LAO': 'Asia Developing', 'MMR': 'Asia Developing', 'BGD': 'Asia Developing', 'LKA': 'Asia Developing',
        'NPL': 'Asia Developing', 'PAK': 'Asia Developing', 'KAZ': 'Asia Developing', 'KGZ': 'Asia Developing',
        'UZB': 'Asia Developing', 'TJK': 'Asia Developing', 'MNG': 'Asia Developing', 'ARM': 'Asia Developing',
        'AZE': 'Asia Developing', 'GEO': 'Asia Developing', 'PRK': 'Asia Developing',
        'AFG': 'Asia Developing', 'BTN': 'Asia Developing', 'MDV': 'Asia Developing',
        'TKM': 'Asia Developing', 'TLS': 'Asia Developing',
        
        # Middle East (15 countries, added BHR)
        'SAU': 'Middle East', 'ARE': 'Middle East', 'QAT': 'Middle East', 'IRN': 'Middle East',
        'KWT': 'Middle East', 'OMN': 'Middle East', 'YEM': 'Middle East', 'JOR': 'Middle East',
        'LBN': 'Middle East', 'SYR': 'Middle East', 'IRQ': 'Middle East', 'ISR': 'Middle East',
        'PSE': 'Middle East', 'TUR': 'Middle East', 'BHR': 'Middle East',
        
        # South America (12 sovereign countries, removed FLK)
        'BRA': 'South America', 'ARG': 'South America', 'CHL': 'South America', 'COL': 'South America',
        'PER': 'South America', 'VEN': 'South America', 'ECU': 'South America', 'BOL': 'South America',
        'PRY': 'South America', 'URY': 'South America', 'GUY': 'South America', 'SUR': 'South America',
        
        # Africa (38 countries, unchanged here but still incomplete for minor states)
        'ZAF': 'Africa', 'EGY': 'Africa', 'NGA': 'Africa', 'MAR': 'Africa',
        'DZA': 'Africa', 'TUN': 'Africa', 'LBY': 'Africa', 'SDN': 'Africa',
        'ETH': 'Africa', 'KEN': 'Africa', 'TZA': 'Africa', 'UGA': 'Africa',
        'RWA': 'Africa', 'GHA': 'Africa', 'CIV': 'Africa', 'SEN': 'Africa',
        'MLI': 'Africa', 'BFA': 'Africa', 'NER': 'Africa', 'BEN': 'Africa',
        'CMR': 'Africa', 'GAB': 'Africa', 'COG': 'Africa', 'COD': 'Africa',
        'AGO': 'Africa', 'ZMB': 'Africa', 'ZWE': 'Africa', 'BWA': 'Africa',
        'NAM': 'Africa', 'SWZ': 'Africa', 'MOZ': 'Africa', 'MDG': 'Africa',
        'MWI': 'Africa', 'LBR': 'Africa', 'SLE': 'Africa', 'GIN': 'Africa',
        'GMB': 'Africa', 'SOM': 'Africa',
        
        # Oceania (5 countries, removed GRL)
        'AUS': 'Oceania', 'NZL': 'Oceania', 'PNG': 'Oceania', 'FJI': 'Oceania',
        'SLB': 'Oceania'
    }


def analyze_wacc_regional_impact(data, scenario='Base24', wacc_variation=0.2):
    """
    Analyze WACC sensitivity impact by region for all technologies
    
    Args:
        wacc_variation: WACC decrease (0.2 = -20% WACC decrease)
    
    Returns:
        DataFrame: Technologies vs Regions WACC impact matrix (% change in LCOF from WACC increase)
    """
    print(f"Analyzing WACC regional impact for {scenario} with -{wacc_variation*100:.0f}% WACC decrease")
    
    if scenario not in data['scenarios']:
        return pd.DataFrame()
    
    lcox_data = data['scenarios'][scenario]
    available_techs = lcox_data.columns.tolist()
    ordered_techs = get_ordered_technologies(available_techs)
    
    regional_mapping = get_regional_mapping()
    
    # Group countries by region
    regions = {}
    for country, region in regional_mapping.items():
        if region not in regions:
            regions[region] = []
        regions[region].append(country)
    
    # Calculate realistic WACC impact: LCOF sensitivity to WACC changes
    # WACC mainly affects CAPEX components through financing costs
    wacc_impact_matrix = np.zeros((len(ordered_techs), len(regions)))
    region_names = list(regions.keys())
    
    for i, tech in enumerate(ordered_techs):
        print(f"  Analyzing WACC impact for {tech}...")
        
        for j, (region_name, countries) in enumerate(regions.items()):
            region_impacts = []
            
            for country in countries:
                if country not in lcox_data.index or tech not in lcox_data.columns:
                    continue
                
                try:
                    base_lcox = lcox_data.loc[country, tech]
                    if pd.isna(base_lcox) or base_lcox <= 0:
                        continue
                    
                    # Get component data to calculate capital-intensive components
                    if tech not in data['component_data'] or scenario not in data['component_data'][tech]:
                        continue
                    
                    comp_df = data['component_data'][tech][scenario]
                    if country not in comp_df.index:
                        continue
                    
                    # Calculate capital-intensive components (CAPEX + replacement costs)
                    capital_components = 0
                    for cap_comp in ['c_capex', 'repex', 'c_upgrade']:
                        if cap_comp in comp_df.columns:
                            comp_value = comp_df.loc[country, cap_comp]
                            if not pd.isna(comp_value) and comp_value > 0:
                                capital_components += comp_value
                    
                    if capital_components > 0:
                        # WACC impact: capital components are sensitive to financing costs
                        # Direct impact: -20% WACC decrease → -20% decrease in capital costs
                        capital_share = capital_components / base_lcox
                        
                        # Direct relationship: WACC change affects capital costs proportionally
                        lcof_impact = capital_share * wacc_variation * 100  # Positive value represents benefit from WACC decrease
                        
                        region_impacts.append(lcof_impact)
                    
                except Exception as e:
                    continue
            
            if region_impacts:
                # Use median impact for this region (consistent with global cost plots)
                wacc_impact_matrix[i, j] = np.median(region_impacts)
    
    return pd.DataFrame(wacc_impact_matrix, index=ordered_techs, columns=region_names)

def create_scenario_sensitivity_figure(data, scenario, output_file):
    """
    Create a 3-panel sensitivity analysis figure for a specific scenario
    Panels: (a) Component Sensitivity, (b) Country Risk Impact, (c) WACC Regional Impact
    """
    print(f"Creating sensitivity analysis figure for {scenario}...")
    
    # Set plot style
    set_plot_style()
    
    # Get component colors and names
    colors, component_names = get_component_colors_and_names()
    
    # Create figure with 3 subplots (1 row, 3 columns)
    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.15, wspace=0.15, 
                          left=0.06, right=0.98, top=0.90, bottom=0.15)
    
    # Get technology categories for consistent ordering
    tech_categories = get_technology_categories()
    
    # Determine consistent technology ordering across all panels
    # Get all technologies that appear in any analysis
    all_techs_in_data = set()
    
    # From LCOX data
    for scenario_data in data['scenarios'].values():
        all_techs_in_data.update(scenario_data.columns.tolist())
    
    # Get ordered list of technologies for consistent y-axis across all panels
    master_tech_order = get_ordered_technologies(list(all_techs_in_data))
    
    print(f"Using complete technology list across all panels: {len(master_tech_order)} technologies")
    print(f"Technology order: {master_tech_order}")
    
    # Panel (a): Component Sensitivity Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sensitivity_df = calculate_component_sensitivity(data, scenario)
    
    if not sensitivity_df.empty:
        # Use master technology order and reindex to ensure consistent ordering
        plot_data = sensitivity_df.reindex(master_tech_order, fill_value=0)
        
        # Keep ALL technologies in master order for consistency across panels
        # Even if some have zero values, they will show as empty rows
        
        if not plot_data.empty:
            # Replace component codes with real names for display
            display_columns = [component_names.get(col, col) for col in plot_data.columns]
            
            vmax = plot_data.values.max()
            im1 = ax1.imshow(plot_data.values, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
            ax1.set_xticks(range(len(plot_data.columns)))
            ax1.set_xticklabels(display_columns, rotation=45, ha='right')
            ax1.set_yticks(range(len(plot_data.index)))
            ax1.set_yticklabels(plot_data.index, fontsize=6)
            ax1.set_title(f'(a) LCOX Benefit from -20% Cost Component Decrease - {scenario}', fontweight='bold')
            ax1.set_xlabel('Cost Components')
            ax1.set_ylabel('Technologies')
            
            # Add numerical annotations on the heatmap
            for i in range(len(plot_data.index)):
                for j in range(len(plot_data.columns)):
                    value = plot_data.iloc[i, j]
                    if value > 0.1:  # Only show significant values
                        text_color = 'white' if value > vmax * 0.5 else 'black'
                        ax1.text(j, i, f'{value:.1f}', ha='center', va='center', 
                               color=text_color, fontsize=7, fontweight='bold')
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Absolute LCOF Impact (%)')
    
    # Panel (b): Country Risk Groups Impact
    ax2 = fig.add_subplot(gs[0, 1])
    risk_impact_df = analyze_country_risk_impact(data, scenario)
    
    if not risk_impact_df.empty:
        # Use master technology order and reindex for consistency
        plot_data = risk_impact_df.reindex(master_tech_order, fill_value=0)
        
        # Keep ALL technologies in master order for consistency across panels
        
        if not plot_data.empty:
            # Set color scale based on data range
            vmax = max(abs(plot_data.values.min()), abs(plot_data.values.max()))
            
            im2 = ax2.imshow(plot_data.values, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
            ax2.set_xticks(range(len(plot_data.columns)))
            ax2.set_xticklabels(plot_data.columns, rotation=45, ha='right')
            ax2.set_yticks(range(len(plot_data.index)))
            ax2.set_yticklabels(plot_data.index, fontsize=6)
            ax2.set_title(f'(b) Impact of Country WACC Groups on Technology LCOX - {scenario}', fontweight='bold')
            ax2.set_xlabel('Country WACC Groups')
            ax2.set_ylabel('Technologies')
            
            # Add numerical annotations on the heatmap
            for i in range(len(plot_data.index)):
                for j in range(len(plot_data.columns)):
                    value = plot_data.iloc[i, j]
                    if abs(value) > 1:  # Only show significant deviations
                        text_color = 'white' if abs(value) > vmax * 0.6 else 'black'
                        ax2.text(j, i, f'{value:+.0f}', ha='center', va='center', 
                               color=text_color, fontsize=7, fontweight='bold')
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('LCOX Deviation from Global Median (%)')
    
    # Panel (c): WACC Regional Sensitivity
    ax3 = fig.add_subplot(gs[0, 2])
    wacc_impact_df = analyze_wacc_regional_impact(data, scenario)
    
    if not wacc_impact_df.empty:
        # Use master technology order and reindex for consistency
        plot_data = wacc_impact_df.reindex(master_tech_order, fill_value=0)
        
        # Keep ALL technologies in master order for consistency across panels
        # Only filter regions with data
        region_mask = plot_data.sum(axis=0) > 0
        plot_data = plot_data.loc[:, region_mask]
        
        if not plot_data.empty:
            vmax = plot_data.values.max()
            im3 = ax3.imshow(plot_data.values, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
            ax3.set_xticks(range(len(plot_data.columns)))
            ax3.set_xticklabels(plot_data.columns, rotation=45, ha='right')
            ax3.set_yticks(range(len(plot_data.index)))
            ax3.set_yticklabels(plot_data.index, fontsize=6)
            ax3.set_title(f'(c) LCOF Benefit from -20% WACC Decrease by Region - {scenario}', fontweight='bold')
            ax3.set_xlabel('Regions')
            ax3.set_ylabel('Technologies')
            
            # Add numerical annotations and country counts on the heatmap
            regional_mapping = get_regional_mapping()
            region_country_counts = {}
            for country, region in regional_mapping.items():
                if region not in region_country_counts:
                    region_country_counts[region] = 0
                region_country_counts[region] += 1
            
            for i in range(len(plot_data.index)):
                for j in range(len(plot_data.columns)):
                    value = plot_data.iloc[i, j]
                    region_name = plot_data.columns[j]
                    country_count = region_country_counts.get(region_name, 0)
                    
                    if value > 0.1:  # Show values above 0.1% LCOF impact
                        # Show both the impact value and country count
                        ax3.text(j, i, f'{value:.1f}%\n({country_count})', ha='center', va='center', 
                               color='white', fontsize=6, fontweight='bold')
            
            # Add colorbar
            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
            cbar3.set_label('Absolute LCOF Impact (%)')
    
    # Save figure
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sensitivity analysis figure for {scenario} saved to {output_file}")

def create_all_scenario_figures(data):
    """
    Create sensitivity analysis figures for all key scenarios
    """
    print("Creating sensitivity analysis figures for all scenarios...")
    
    # Define scenarios to analyze
    scenarios = {
        'Base24': 'figures/sensitivity_analysis_base24.png',
        'Base50': 'figures/sensitivity_analysis_bau_2050.png', 
        '2deg50': 'figures/sensitivity_analysis_2deg_2050.png',
        '15deg50': 'figures/sensitivity_analysis_15deg_2050.png'
    }
    
    # Create a figure for each scenario
    for scenario, output_file in scenarios.items():
        if scenario in data['scenarios']:
            create_scenario_sensitivity_figure(data, scenario, output_file)
        else:
            print(f"Warning: Scenario {scenario} not found in data")
    
    print("All sensitivity analysis figures completed!")

def debug_available_data(data):
    """Debug function to see what data is actually available"""
    print("\n" + "="*60)
    print("DEBUGGING AVAILABLE DATA")
    print("="*60)
    
    # Print available component sheets
    print(f"\nAvailable component data for technologies:")
    for tech, scenarios in data['component_data'].items():
        print(f"  {tech}: {list(scenarios.keys())}")
    
    # Print technologies in LCOX data vs component data
    if data['scenarios']:
        first_scenario = list(data['scenarios'].keys())[0]
        lcox_techs = set(data['scenarios'][first_scenario].columns)
        comp_techs = set(data['component_data'].keys())
        
        print(f"\nTechnologies in LCOX data ({len(lcox_techs)}): {sorted(lcox_techs)}")
        print(f"\nTechnologies in component data ({len(comp_techs)}): {sorted(comp_techs)}")
        
        missing_in_comp = lcox_techs - comp_techs
        if missing_in_comp:
            print(f"\nTechnologies in LCOX but missing in component data: {sorted(missing_in_comp)}")
        
        missing_in_lcox = comp_techs - lcox_techs
        if missing_in_lcox:
            print(f"\nTechnologies in component data but missing in LCOX: {sorted(missing_in_lcox)}")
    
    print("="*60)

def aggregate_fuel_technologies(data):
    """
    [DEPRECATED/DISABLED] Aggregate fuel-specific technologies from the same base technology
    E.g., combine RWGS_FT_diesel and RWGS_FT_kerosene into RWGS_FT
    
    NOTE: This function is no longer used because we want to keep fuel-specific 
    technologies separate (ST_FT_kerosene vs ST_FT_diesel) rather than aggregating
    them into base technology names (ST_FT).
    """
    print("\nAggregating fuel-specific technologies...")
    
    # Define aggregation mapping: base_tech -> [fuel_specific_techs]
    aggregation_mapping = {
        'RWGS_FT': ['RWGS_FT_diesel', 'RWGS_FT_kerosene'],
        'RWGS_MeOH': ['RWGS_MeOH_methanol', 'RWGS_MeOH_kerosene', 'RWGS_MeOH_DME'],
        'SR_FT': ['SR_FT_diesel', 'SR_FT_kerosene'],
        'ST_FT': ['ST_FT_diesel', 'ST_FT_kerosene'],
        'TG_FT': ['TG_FT_diesel', 'TG_FT_kerosene'],
        'HVO': ['HVO_diesel', 'HVO_kerosene'],
        'B_PYR': ['B_PYR_kerosene']  # Only kerosene version exists
    }
    
    # Aggregate component data
    for base_tech, fuel_techs in aggregation_mapping.items():
        # Check which fuel-specific versions exist in the data
        existing_fuel_techs = [tech for tech in fuel_techs if tech in data['component_data']]
        
        if len(existing_fuel_techs) > 0:
            print(f"  Aggregating {base_tech} from: {existing_fuel_techs}")
            
            # Initialize base tech data structure
            data['component_data'][base_tech] = {}
            
            # For each scenario, aggregate the fuel-specific data
            for scenario in data['scenarios'].keys():
                # Collect data from all fuel-specific versions for this scenario
                scenario_data_list = []
                
                for fuel_tech in existing_fuel_techs:
                    if scenario in data['component_data'][fuel_tech]:
                        scenario_data_list.append(data['component_data'][fuel_tech][scenario])
                
                if scenario_data_list:
                    # Average the component data across fuel types
                    # This gives us the "typical" cost structure for this base technology
                    aggregated_data = pd.concat(scenario_data_list, axis=0).groupby(level=0).mean()
                    data['component_data'][base_tech][scenario] = aggregated_data
            
            # Also aggregate LCOX data if both versions exist
            for scenario_name, scenario_data in data['scenarios'].items():
                existing_cols = [tech for tech in fuel_techs if tech in scenario_data.columns]
                if len(existing_cols) > 1:
                    # Take average of fuel-specific LCOX values
                    aggregated_lcox = scenario_data[existing_cols].mean(axis=1)
                    data['scenarios'][scenario_name][base_tech] = aggregated_lcox
                elif len(existing_cols) == 1:
                    # Just rename the single existing technology
                    data['scenarios'][scenario_name][base_tech] = scenario_data[existing_cols[0]]
    
    # Update the technologies list to include aggregated technologies
    if data['scenarios']:
        first_scenario = list(data['scenarios'].keys())[0]
        data['technologies'] = list(data['scenarios'][first_scenario].columns)
    
    print(f"After aggregation, found {len(data['technologies'])} technologies")
    return data

def main():
    """Main function to run the comprehensive sensitivity analysis"""
    try:
        print("Starting comprehensive sensitivity analysis...")
        # Load data from Excel output file (same approach as working sensitivity analysis)
        print("Loading data...")
        data = load_data('output/lcox_results.xlsx')
        
        print("Data loaded successfully!")
        print(f"Found {len(data['scenarios'])} scenarios: {list(data['scenarios'].keys())}")
        print(f"Found {len(data['countries'])} countries")
        print(f"Found {len(data['technologies'])} technologies")
        
        # Debug available data
        debug_available_data(data)
        
        # Note: Fuel-specific technologies are kept separate (no aggregation)
        # This ensures ST_FT_kerosene and ST_FT_diesel are treated as distinct technologies
        print("Keeping fuel-specific technologies separate (no aggregation to base names)")
        
        # Debug available data (no change after skipping aggregation)
        print("Final technology list (fuel-specific technologies kept separate):")
        
        # Create figures for all scenarios
        create_all_scenario_figures(data)
        
        print("Comprehensive sensitivity analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in comprehensive sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 