"""
Green Countries Cost Analysis for PEM and RWGS-FT-Kerosene
Focus on countries with low-carbon grid electricity based on year-specific thresholds

This script creates cost component rankings for:
1. PEM electrolysis (green hydrogen production)
2. RWGS-FT-Kerosene (synthetic aviation fuel production)

Uses year-specific carbon intensity thresholds:
- 2024: ≤ 0.077 kg CO₂-eq/kWh
- 2030: ≤ 0.083 kg CO₂-eq/kWh  
- 2050: ≤ 0.099 kg CO₂-eq/kWh

Filtering criteria:
1. Countries meeting carbon intensity thresholds
2. Major economies only (based on 2024 GDP rankings)
3. Limited to top 20 lowest-cost countries for clear visualization
4. Excludes small territories and countries with limited data

Includes renewable energy mix information and cost component breakdowns.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def set_plot_style():
    """Set plot style"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'figure.dpi': 300,
        'figure.constrained_layout.use': True,
        'mathtext.default': 'regular',
        'axes.spines.top': False,
        'axes.spines.right': False
    })

def load_green_countries_data():
    """Load the green hydrogen analysis results from lcoe.py output"""
    try:
        base_dir = Path(__file__).parent.parent
        # Use the correct data file from lcoe.py
        analysis_file = base_dir / "output" / "lcoe_and_grid_prices_electrolyzer_optimized.xlsx"
        
        # Check if the LCOE analysis file exists
        if not analysis_file.exists():
            print("LCOE analysis file not found. Running LCOE analysis...")
            run_lcoe_analysis()
        
        # Load the data from lcoe.py output
        print(f"Loading data from {analysis_file}")
        df = pd.read_excel(analysis_file)
        print(f"Loaded LCOE data with {len(df)} countries")
        
        # Transform the data to match expected format
        # The lcoe.py file has columns like: Carbon_Intensity_Base_2024, Green_H2_Standard_Base_2024, etc.
        
        # Create a list to store transformed records
        records = []
        
        # Get all scenarios from column names
        scenarios = set()
        for col in df.columns:
            if 'Carbon_Intensity_' in col:
                scenario = col.replace('Carbon_Intensity_', '')
                scenarios.add(scenario)
        
        print(f"Found scenarios: {sorted(scenarios)}")
        
        # Transform each country-scenario combination into a separate record
        for _, row in df.iterrows():
            country_code = row['ISO_A3_EH']
            country_name = row.get('NAME_EN', country_code)
            
            for scenario in scenarios:
                # Check if required columns exist for this scenario
                carbon_col = f'Carbon_Intensity_{scenario}'
                green_standard_col = f'Green_H2_Standard_{scenario}'
                green_strict_col = f'Green_H2_Strict_{scenario}'
                solar_share_col = f'Grid_Solar_Share_{scenario}'
                wind_share_col = f'Grid_Wind_Share_{scenario}'
                
                if all(col in df.columns for col in [carbon_col, green_standard_col, green_strict_col]):
                    records.append({
                        'Country': country_code,
                        'Country_Name': country_name,
                        'Scenario': scenario,
                        'Carbon_Intensity': row[carbon_col],
                        'Green_Standard': row[green_standard_col],
                        'Green_Strict': row[green_strict_col],
                        'Solar_Share': row.get(solar_share_col, 0),
                        'Wind_Share': row.get(wind_share_col, 0),
                        'Hydro_Share': 0,  # lcoe.py doesn't break this out separately
                        'Nuclear_Share': 0,  # lcoe.py doesn't break this out separately
                        'Coal_Share': 0,   # lcoe.py doesn't break this out separately
                        'Gas_Share': 0     # lcoe.py doesn't break this out separately
                    })
        
        # Convert to DataFrame
        detailed_df = pd.DataFrame(records)
        
        if len(detailed_df) == 0:
            print("Warning: No valid data found in LCOE file")
            return None
        
        print(f"Transformed into {len(detailed_df)} country-scenario combinations")
        
        # Validate we have the expected data
        green_count = detailed_df['Green_Standard'].sum()
        print(f"Found {green_count} green hydrogen eligible country-scenarios")
        
        return detailed_df
        
    except Exception as e:
        print(f"Error loading green countries data: {e}")
        print("Attempting to run LCOE analysis...")
        try:
            run_lcoe_analysis()
            # Try loading again after running the analysis
            analysis_file = base_dir / "output" / "lcoe_and_grid_prices_electrolyzer_optimized.xlsx"
            df = pd.read_excel(analysis_file)
            print(f"Successfully loaded after running analysis: {len(df)} countries")
            
            # Repeat the transformation process
            records = []
            scenarios = set()
            for col in df.columns:
                if 'Carbon_Intensity_' in col:
                    scenario = col.replace('Carbon_Intensity_', '')
                    scenarios.add(scenario)
            
            for _, row in df.iterrows():
                country_code = row['ISO_A3_EH']
                country_name = row.get('NAME_EN', country_code)
                
                for scenario in scenarios:
                    carbon_col = f'Carbon_Intensity_{scenario}'
                    green_standard_col = f'Green_H2_Standard_{scenario}'
                    green_strict_col = f'Green_H2_Strict_{scenario}'
                    solar_share_col = f'Grid_Solar_Share_{scenario}'
                    wind_share_col = f'Grid_Wind_Share_{scenario}'
                    
                    if all(col in df.columns for col in [carbon_col, green_standard_col, green_strict_col]):
                        records.append({
                            'Country': country_code,
                            'Country_Name': country_name,
                            'Scenario': scenario,
                            'Carbon_Intensity': row[carbon_col],
                            'Green_Standard': row[green_standard_col],
                            'Green_Strict': row[green_strict_col],
                            'Solar_Share': row.get(solar_share_col, 0),
                            'Wind_Share': row.get(wind_share_col, 0),
                            'Hydro_Share': 0,
                            'Nuclear_Share': 0,
                            'Coal_Share': 0,
                            'Gas_Share': 0
                        })
            
            return pd.DataFrame(records)
            
        except Exception as e2:
            print(f"Failed to run LCOE analysis: {e2}")
            return None

def run_lcoe_analysis():
    """Run the LCOE analysis to generate updated data"""
    import sys
    import subprocess
    
    try:
        # Get the path to the LCOE script
        base_dir = Path(__file__).parent.parent
        lcoe_script = base_dir / "scripts" / "lcoe.py"
        
        if not lcoe_script.exists():
            raise FileNotFoundError(f"LCOE script not found at {lcoe_script}")
        
        print(f"Running LCOE analysis from {lcoe_script}")
        
        # Run the LCOE script
        result = subprocess.run([sys.executable, str(lcoe_script)], 
                              capture_output=True, text=True, cwd=str(base_dir))
        
        if result.returncode == 0:
            print("LCOE analysis completed successfully")
            print("Output:", result.stdout[-500:])  # Show last 500 chars of output
        else:
            print(f"LCOE analysis failed with return code {result.returncode}")
            print("Error output:", result.stderr[-500:])  # Show last 500 chars of error
            raise RuntimeError(f"LCOE analysis failed: {result.stderr}")
            
    except Exception as e:
        print(f"Error running LCOE analysis: {e}")
        
        # Try importing and running the module directly
        try:
            print("Attempting to import and run LCOE module directly...")
            sys.path.insert(0, str(base_dir / "scripts"))
            import lcoe
            print("Successfully ran LCOE analysis via direct import")
        except Exception as e2:
            print(f"Direct import also failed: {e2}")
            raise

def load_component_data():
    """Load component cost data from Excel file"""
    try:
        base_dir = Path(__file__).parent.parent
        excel_file = base_dir / "output" / "lcox_results.xlsx"
        
        if not excel_file.exists():
            print(f"Excel file {excel_file} not found!")
            return None
        
        print(f"Loading component data from {excel_file}")
        
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
        print(f"Error loading component data: {e}")
        return None

def get_component_styling():
    """Get consistent styling for cost components with colors matching aggregatedplot.py"""
    # Use exact same color palette as aggregatedplot.py
    colors = {
        'c_capex': '#3182bd',         # Blue for CAPEX
        'c_om': '#6baed6',            # Light blue for O&M
        'repex': '#fd8d3c',           # Orange for replacement costs
        'c_elec': '#f0eebb',          # Light yellow for electricity (total)
        'c_elec_solar': '#ffeda0',    # Pale yellow for solar electricity
        'c_elec_wind': '#c7e9b4',     # Light green for wind electricity  
        'c_elec_conventional': '#fd8d3c',  # Orange for conventional electricity
        'c_heat': '#9b3a4d',          # Dark red for heat
        'c_ng': '#636363',            # Dark gray for natural gas
        'c_bio': '#31a354',           # Green for biomass
        'c_pw': '#74c476',            # Light green for process water
        'c_iw': '#a1d99b',            # Pale green for industrial water
        'c_h2': '#9ecae1',            # Pale blue for hydrogen (default)
        'c_co2': '#f4c28f',           # Orange-tan for CO2
        'c_h2_storage': '#c6dbef',    # Very light blue for H2 storage
        'c_co2_storage': '#fb6a4a',   # Light red for CO2 storage
        'c_upgrade': '#9467bd',       # Purple for upgrading costs
        'c_shipping': '#c5b0d5',      # Light purple for shipping
    }
    
    # Use exact same component names as aggregatedplot.py
    names = {
        'c_capex': 'CAPEX',
        'c_om': 'FOC',
        'repex': 'REPEX',
        'c_upgrade': 'Upgrading CAPEX',
        'c_elec': 'Electricity (Total)',
        'c_elec_solar': 'Solar Electricity',
        'c_elec_wind': 'Wind Electricity',
        'c_elec_conventional': 'Other Electricities',
        'c_heat': 'Heat',
        'c_bio': 'Biomass',
        'c_ng': 'Natural Gas',
        'c_pw': 'Process Water',
        'c_iw': 'Industrial Water',
        'c_h2': r'H$_2$',
        'c_co2': r'CO$_2$',
        'c_h2_storage': r'H$_2$ Storage',
        'c_co2_storage': r'CO$_2$ Storage',
        'c_shipping': 'Shipping'
    }
    
    return colors, names

def get_country_names():
    """Load country code to name mapping from TEA input.xlsx"""
    try:
        base_dir = Path(__file__).parent.parent
        iso_data = pd.read_excel(base_dir / 'data' / 'TEA input.xlsx', sheet_name='ISO A3')
        country_map = {row['ISO A3']: row['Country'] for _, row in iso_data.iterrows() if pd.notna(row['ISO A3'])}
        print(f"Loaded {len(country_map)} country mappings from TEA input.xlsx")
        return country_map
    except Exception as e:
        print(f"Error loading country mappings: {e}")
        return {}

def find_component_sheet(component_data, tech, scenario):
    """Find the appropriate component sheet for a technology and scenario"""
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
    
    # Try with partial match
    matching_sheets = [s for s in component_data.keys() 
                      if s.startswith(f'Comp_{scenario_code}_{tech}'[:20])]
    
    if matching_sheets:
        return matching_sheets[0]
    
    print(f"No component data sheet found for {tech} in {scenario}")
    return None

def get_electricity_breakdown(country, scenario, green_data, total_electricity_cost):
    """
    Break down electricity cost into solar, wind, and conventional components
    
    Args:
        country: Country code
        scenario: Scenario name
        green_data: DataFrame with renewable share data
        total_electricity_cost: Total electricity cost from component data
    
    Returns:
        Dictionary with solar, wind, and conventional electricity costs
    """
    # Find the matching country-scenario in green_data
    country_scenario_data = green_data[
        (green_data['Country'] == country) & 
        (green_data['Scenario'] == scenario)
    ]
    
    if len(country_scenario_data) == 0:
        # No breakdown data available, return total as conventional
        return {
            'c_elec_solar': 0.0,
            'c_elec_wind': 0.0,
            'c_elec_conventional': total_electricity_cost
        }
    
    # Get renewable shares
    row = country_scenario_data.iloc[0]
    solar_share = row.get('Solar_Share', 0)
    wind_share = row.get('Wind_Share', 0)
    
    # Calculate conventional share (remaining after renewables)
    conventional_share = 1.0 - solar_share - wind_share
    conventional_share = max(0.0, conventional_share)  # Ensure non-negative
    
    # Breakdown electricity cost proportionally
    solar_cost = total_electricity_cost * solar_share
    wind_cost = total_electricity_cost * wind_share
    conventional_cost = total_electricity_cost * conventional_share
    
    return {
        'c_elec_solar': solar_cost,
        'c_elec_wind': wind_cost,
        'c_elec_conventional': conventional_cost
    }

def get_excluded_countries():
    """
    Get list of countries to exclude from analysis
    Excludes small territories, islands, countries with limited data, and specific cases
    """
    excluded_countries = {
        # Small territories and dependencies
        'AND', 'ASM', 'ATG', 'BLZ', 'BMU', 'BES', 'BTN', 'CPV', 'CYM', 'COM', 
        'DMA', 'FRO', 'FLK', 'GRD', 'GGY', 'GIB', 'GLP', 'GNB', 'IMN', 'JEY', 
        'KIR', 'LIE', 'MAC', 'MHL', 'MTQ', 'MCO', 'MSR', 'NRU', 'NCL', 'NIU', 
        'PLW', 'PCN', 'PYF', 'REU', 'SMR', 'STP', 'SYC', 'SXM', 'SLB', 'KNA', 
        'LCA', 'VCT', 'WSM', 'TLS', 'TON', 'TUV', 'VUT', 'VAT', 'VIR', 'SPM', 
        'SHN', 'GUM', 'UMI', 'ATA', 'ALA', 'PSE'
        
        # Countries with limited data availability or specific issues
        'AFG', 'ERI', 'LBY', 'SOM', 'SSD', 'SYR', 'YEM',
        
        # Small countries with limited industrial relevance for hydrogen
        'SUR', 'GUY', 'BHS', 'MNG'
    }
        
    return excluded_countries

def get_major_economies():
    """
    Get list of major economies to focus analysis on ~35-40 countries
    Based on 2024 GDP rankings and hydrogen market relevance
    """
    major_economies = {
        # Top economies by nominal GDP (2024) + key hydrogen markets
        'USA',  # United States
        'CHN',  # China
        'DEU',  # Germany  
        'JPN',  # Japan
        'IND',  # India
        'GBR',  # United Kingdom
        'FRA',  # France
        'ITA',  # Italy
        'BRA',  # Brazil
        'CAN',  # Canada
        'RUS',  # Russia
        'KOR',  # South Korea
        'AUS',  # Australia
        'ESP',  # Spain
        'MEX',  # Mexico
        'IDN',  # Indonesia
        'NLD',  # Netherlands
        'SAU',  # Saudi Arabia
        'TUR',  # Turkey
        'CHE',  # Switzerland
        'BEL',  # Belgium
        'ARG',  # Argentina
        'NOR',  # Norway
        'ARE',  # UAE
        'ZAF',  # South Africa
        'CHL',  # Chile
        'DNK',  # Denmark
        'SWE',  # Sweden
        'POL',  # Poland
        'FIN',  # Finland
        'AUT',  # Austria
        'NZL',  # New Zealand
        'SGP',  # Singapore
        'ISR',  # Israel
        'IRL',  # Ireland
        
        # Additional countries for broader coverage
        'THA',  # Thailand
        'EGY',  # Egypt
        'BGD',  # Bangladesh
        'VNM',  # Vietnam
        'PRT',  # Portugal
        'MYS',  # Malaysia
        'CZE',  # Czech Republic
        'HUN',  # Hungary
        'PER',  # Peru
        'COL',  # Colombia
        'ECU',  # Ecuador
        'PHL',  # Philippines
        'MAR',  # Morocco
        'UKR',  # Ukraine
        'SVK',  # Slovakia
        'SVN',  # Slovenia
        'HRV',  # Croatia
        'LTU',  # Lithuania
        'LVA',  # Latvia
        'EST',  # Estonia
    }
    return major_economies

def load_gdp_table(path=None):
    """
    Load a GDP table to select top-N countries by GDP.
    Accepts a CSV with columns including 'Country Code' (ISO3) and '2024' value.
    Returns a DataFrame with columns ISO3 and GDP_USD, or None on failure.
    """
    try:
        from pathlib import Path
        import pandas as pd
        # default to the uploaded file if present
        csv_path = Path(path) if path else Path("../../data/gdp_2024.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Normalize expected columns
            cols = {c.lower(): c for c in df.columns}
            code_col = cols.get('country code', None)
            val_col = cols.get('2024', None)
            if code_col is None or val_col is None:
                # Try alternative naming
                code_col = code_col or 'Country Code'
                val_col = val_col or df.columns[-1]
            out = df[[code_col, val_col]].copy()
            out.rename(columns={code_col: 'ISO3', val_col: 'GDP_USD'}, inplace=True)
            out['ISO3'] = out['ISO3'].astype(str).str.upper().str.strip()
            out['GDP_USD'] = pd.to_numeric(out['GDP_USD'], errors='coerce')
            out = out.dropna(subset=['ISO3', 'GDP_USD'])
            return out
    except Exception as e:
        print(f"Warning: could not load GDP table: {e}")
    return None


def get_year_specific_threshold(scenario):
    """
    Get year-specific carbon intensity threshold based on scenario year
    
    Thresholds:
    - 2024: ≤ 0.077 kg CO₂-eq/kWh
    - 2030: ≤ 0.083 kg CO₂-eq/kWh
    - 2050: ≤ 0.099 kg CO₂-eq/kWh
    """
    # Extract year from scenario name
    if '2024' in scenario:
        return 0.077  # Use 2024 threshold for 2024 data
    elif '2030' in scenario:
        return 0.083
    elif '2050' in scenario:
        return 0.099
    else:
        # Default to 2030 threshold if year not clear
        return 0.083

def filter_green_countries(green_data, scenario, top_n=30, use_gdp=True):
    """
    Filter countries by year-specific carbon intensity threshold, exclude small territories,
    then (optionally) keep Top-N by GDP using an external CSV. Falls back to a curated
    allow-list of major economies if the GDP file is unavailable.
    Returns (list_of_country_codes, filtered_dataframe)
    """
    scenario_data = green_data[green_data['Scenario'] == scenario].copy()

    # Apply year-specific carbon intensity threshold
    ci_threshold = get_year_specific_threshold(scenario)
    green = scenario_data[scenario_data['Carbon_Intensity'] <= ci_threshold].copy()

    # Exclude small territories or countries with limited data
    excluded = get_excluded_countries()
    green = green[~green['Country'].isin(excluded)].copy()

    kept_by = "GDP"
    if use_gdp:
        gdp_df = load_gdp_table()
    else:
        gdp_df = None

    if gdp_df is not None and len(gdp_df) > 0:
        # Merge on ISO3 code
        merged = green.merge(gdp_df, left_on='Country', right_on='ISO3', how='inner')
        merged = merged.sort_values('GDP_USD', ascending=False)
        green = merged.head(top_n).copy()
    else:
        # Fall back to curated major economies list, then cap to top_n
        kept_by = "major-economies list"
        major = set(get_major_economies())
        green = green[green['Country'].isin(major)].copy()
        if len(green) > top_n:
            green = green.sort_values('Carbon_Intensity').head(top_n).copy()

    print(f"\n{scenario} - Green countries after {kept_by} selection: {len(green)} (cap {top_n})")
    return green['Country'].tolist(), green


def create_green_countries_cost_plot(component_data, green_data, tech, scenario, 
                                   output_dir):
    """
    Create horizontal bar chart showing cost components for green countries only
    """
    colors, component_names = get_component_styling()
    country_name_map = get_country_names()
    
    sheet_name = find_component_sheet(component_data, tech, scenario)
    if not sheet_name:
        print(f"No component data sheet found for {tech} in {scenario}")
        return None
    
    data = component_data[sheet_name]
    
    # Filter to green countries only (excluding small territories)
    green_countries, green_info = filter_green_countries(green_data, scenario, top_n=30, use_gdp=True)
    
    if not green_countries:
        print(f"No green countries found for {scenario}")
        return None
    
    # Filter component data to green countries (also ensure no excluded countries slip through)
    excluded_countries = get_excluded_countries()
    green_countries_in_data = [c for c in green_countries if c in data.index and c not in excluded_countries]
    
    if not green_countries_in_data:
        print(f"No green countries found in component data for {tech}")
        return None
    
    # Calculate total costs and sort
    country_totals = {}
    for country in green_countries_in_data:
        if pd.notna(country):
            row_data = data.loc[country]
            
            if isinstance(row_data, pd.Series):
                numeric_values = [v for v in row_data.values if isinstance(v, (int, float)) and pd.notna(v)]
                total = sum(numeric_values)
            else:
                numeric_cols = row_data.select_dtypes(include=[np.number])
                total = numeric_cols.sum().sum()
                
            if total > 0:
                country_totals[country] = total
    
    if not country_totals:
        print(f"No valid cost data for green countries in {tech} - {scenario}")
        return None
    
    # Sort countries by total cost (lowest to highest for green countries)
    sorted_countries = sorted(country_totals.items(), key=lambda x: x[1])
    
    countries = [item[0] for item in sorted_countries]
    
    # Get clean country names only
    country_labels = [country_name_map.get(country, country) for country in countries]
    
    # Create figure with professional dimensions (scaled for more countries)
    height = max(10, len(countries) * 0.3 + 3)
    fig, ax = plt.subplots(figsize=(10, height))
    
    # Component order - replace c_elec with detailed electricity breakdown
    component_order = ['c_capex', 'c_om', 'repex', 'c_upgrade', 
                      'c_elec_solar', 'c_elec_wind', 'c_elec_conventional', 'c_heat', 
                      'c_bio', 'c_ng', 'c_h2', 'c_h2_storage', 'c_co2', 'c_co2_storage',
                      'c_pw', 'c_iw', 'c_shipping']
    
    # Create enhanced data with electricity breakdown
    enhanced_data = data.copy()
    
    # For each country, if c_elec exists, break it down into renewable components
    if 'c_elec' in enhanced_data.columns:
        for country in enhanced_data.index:
            if pd.notna(enhanced_data.loc[country, 'c_elec']):
                total_elec_cost = enhanced_data.loc[country, 'c_elec']
                
                # Get electricity breakdown using renewable share data
                breakdown = get_electricity_breakdown(country, scenario, green_data, total_elec_cost)
                
                # Add breakdown components to enhanced data
                enhanced_data.loc[country, 'c_elec_solar'] = breakdown['c_elec_solar']
                enhanced_data.loc[country, 'c_elec_wind'] = breakdown['c_elec_wind']  
                enhanced_data.loc[country, 'c_elec_conventional'] = breakdown['c_elec_conventional']
    
    available_components = [comp for comp in component_order if comp in enhanced_data.columns or comp.startswith('c_elec_')]
    
    # Initialize left array for stacking
    lefts = np.zeros(len(countries))
    bar_handles = []
    bar_labels = []
    
    # Plot each component with consistent styling
    for component in available_components:
        values = []
        for country in countries:
            if country in enhanced_data.index:
                if component in enhanced_data.columns:
                    value = enhanced_data.loc[country, component]
                else:
                    value = 0
                if pd.isna(value):
                    value = 0
                values.append(value)
            else:
                values.append(0)
        
        if any(v > 0.001 for v in values):
            bar = ax.barh(range(len(countries)), values, left=lefts, height=0.8,
                  label=component_names.get(component, component),
                  color=colors.get(component, '#888888'),
                  edgecolor='white', linewidth=0.5, alpha=0.9)
            
            lefts += np.array(values)
            bar_handles.append(bar)
            bar_labels.append(component_names.get(component, component))
    
    # Add total cost values with better formatting
    for i, (country, total) in enumerate(sorted_countries):
        ax.text(total * 1.01, i, f'{total:.3f}', 
               va='center', ha='left', fontsize=11, fontweight='normal')
    
    # Customize plot with professional styling
    ci_threshold = get_year_specific_threshold(scenario)
    tech_name = "PEM Electrolysis" if tech == "PEM" else "RWGS-FT SAF"
    
    ax.set_title(f'{tech_name} - Cost Components (Green Grid Countries)\n'
                f'{scenario} | CI ≤ {ci_threshold:.3f} kg CO$_2$-eq/kWh', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Levelized Cost (EUR/kWh)', fontweight='bold')
    ax.set_ylabel('Countries (ranked by total cost)', fontweight='bold')
    
    # Set y-axis with clean country names
    ax.set_yticks(range(len(country_labels)))
    ax.set_yticklabels(country_labels, fontsize=11)
    ax.set_ylim(-0.5, len(countries)-0.5)
    
    # Set custom x-axis limits based on technology and scenario
    if tech == "PEM" and scenario in ["1.5 degree_2050", "1.5 degree_2030"]:
        ax.set_xlim(0, 0.16)
    elif tech == "RWGS_FT_kerosene" and scenario in ["1.5 degree_2050", "1.5 degree_2030"]:
        ax.set_xlim(0, 0.40)
    
    # Improve grid styling
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Professional legend
    ax.legend(bar_handles, bar_labels, loc='lower right',
             frameon=True, fancybox=False, edgecolor='black', 
             fontsize=11, ncol=1)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust margins
    plt.tight_layout()
    
    # Save plot with high quality
    filename = f'green_countries_{tech}_{scenario.replace(" ", "_")}_CI{ci_threshold:.3f}_clean.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved clean green countries plot: {filename}")
    return filename

def create_comparison_plot(component_data, green_data, scenario, output_dir):
    """
    Create side-by-side comparison of PEM and RWGS-FT-Kerosene for green countries
    """
    # Get green countries first to determine figure height
    green_countries, green_info = filter_green_countries(green_data, scenario, top_n=30, use_gdp=True)
    num_countries = len([c for c in green_countries if c not in get_excluded_countries()])
    
    # Make figure height scaled for more countries
    height = max(10, num_countries * 0.3 + 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, height))
    
    technologies = ['PEM', 'RWGS_FT_kerosene']
    tech_labels = ['PEM Electrolysis', 'RWGS-FT SAF']
    axes = [ax1, ax2]
    
    colors, component_names = get_component_styling()
    country_name_map = get_country_names()
    
    # Remove the duplicate call since we already got green_countries above
    if not green_countries:
        print(f"No green countries found for {scenario}")
        return None
    
    # Get excluded countries list for additional filtering
    excluded_countries = get_excluded_countries()
    
    for tech_idx, (tech, tech_label, ax) in enumerate(zip(technologies, tech_labels, axes)):
        sheet_name = find_component_sheet(component_data, tech, scenario)
        if not sheet_name:
            ax.text(0.5, 0.5, f'No data for {tech}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            continue
        
        data = component_data[sheet_name]
        
        # Filter to green countries in data (excluding small territories)
        green_countries_in_data = [c for c in green_countries if c in data.index and c not in excluded_countries]
        
        if not green_countries_in_data:
            ax.text(0.5, 0.5, f'No green countries\nin data for {tech}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            continue
        
        # Calculate totals and sort
        country_totals = {}
        for country in green_countries_in_data:
            row_data = data.loc[country]
            if isinstance(row_data, pd.Series):
                numeric_values = [v for v in row_data.values if isinstance(v, (int, float)) and pd.notna(v)]
                total = sum(numeric_values)
            else:
                numeric_cols = row_data.select_dtypes(include=[np.number])
                total = numeric_cols.sum().sum()
            
            if total > 0:
                country_totals[country] = total
        
        # Sort and prepare data
        sorted_countries = sorted(country_totals.items(), key=lambda x: x[1])
        
        countries = [item[0] for item in sorted_countries]
        
        # Get clean country labels
        country_labels = [country_name_map.get(country, country) for country in countries]
        
        # Plot components with electricity breakdown
        component_order = ['c_capex', 'c_om', 'repex', 'c_upgrade', 
                          'c_elec_solar', 'c_elec_wind', 'c_elec_conventional', 'c_heat', 
                          'c_bio', 'c_ng', 'c_h2', 'c_h2_storage', 'c_co2', 'c_co2_storage',
                          'c_pw', 'c_iw', 'c_shipping']
        
        # Create enhanced data with electricity breakdown
        enhanced_data = data.copy()
        
        # For each country, if c_elec exists, break it down into renewable components
        if 'c_elec' in enhanced_data.columns:
            for country in enhanced_data.index:
                if pd.notna(enhanced_data.loc[country, 'c_elec']):
                    total_elec_cost = enhanced_data.loc[country, 'c_elec']
                    
                    # Get electricity breakdown using renewable share data
                    breakdown = get_electricity_breakdown(country, scenario, green_data, total_elec_cost)
                    
                    # Add breakdown components to enhanced data
                    enhanced_data.loc[country, 'c_elec_solar'] = breakdown['c_elec_solar']
                    enhanced_data.loc[country, 'c_elec_wind'] = breakdown['c_elec_wind']  
                    enhanced_data.loc[country, 'c_elec_conventional'] = breakdown['c_elec_conventional']
        
        available_components = [comp for comp in component_order if comp in enhanced_data.columns or comp.startswith('c_elec_')]
        
        lefts = np.zeros(len(countries))
        
        for component in available_components:
            values = []
            for country in countries:
                if country in enhanced_data.index:
                    if component in enhanced_data.columns:
                        value = enhanced_data.loc[country, component]
                    else:
                        value = 0
                else:
                    value = 0
                if pd.isna(value):
                    value = 0
                values.append(value)
            
            if any(v > 0.001 for v in values):
                ax.barh(range(len(countries)), values, left=lefts, height=0.8,
                       label=component_names.get(component, component) if tech_idx == 0 else "",
                       color=colors.get(component, '#888888'),
                       edgecolor='white', linewidth=0.5, alpha=0.9)
                
                lefts += np.array(values)
        
        # Add total values
        for i, (country, total) in enumerate(sorted_countries):
            ax.text(total * 1.01, i, f'{total:.3f}', 
                   va='center', ha='left', fontsize=10, fontweight='normal')
        
        # Customize subplot
        ax.set_title(tech_label, fontweight='bold', fontsize=16, pad=15)
        ax.set_xlabel('Levelized Cost (EUR/kWh)', fontweight='bold')
        
        if tech_idx == 0:
            ax.set_ylabel('Countries (ranked by total cost)', fontweight='bold')
        
        ax.set_yticks(range(len(country_labels)))
        ax.set_yticklabels(country_labels, fontsize=11)
        ax.set_ylim(-0.5, len(countries)-0.5)
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add overall title
    ci_threshold = get_year_specific_threshold(scenario)
    fig.suptitle(f'Green Grid Countries Cost Comparison\n'
                f'{scenario} | CI ≤ {ci_threshold:.3f} kg CO$_2$-eq/kWh', 
                fontweight='bold', fontsize=18, y=0.95)
    
    # Create professional legend
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(6, len(handles)),
                  bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=False,
                  edgecolor='black', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.88)
    
    # Save plot with high quality
    filename = f'green_countries_comparison_{scenario.replace(" ", "_")}_CI{ci_threshold:.3f}_clean.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved clean comparison plot: {filename}")
    return filename

def main():
    """Main function to create green countries cost analysis"""
    set_plot_style()
    
    print("Green Countries Cost Analysis for PEM and RWGS-FT-Kerosene (Year-Specific CI Thresholds)")
    print("=" * 80)
    print("This analysis uses enhanced LCOE data with:")
    print("- Country-specific f_wacc_c factors for conventional electricity")  
    print("- Grid carbon intensity calculations")
    print("- Year-specific carbon intensity thresholds:")
    print("  * 2024: ≤ 0.077 kg CO₂-eq/kWh")
    print("  * 2030: ≤ 0.083 kg CO₂-eq/kWh")
    print("  * 2050: ≤ 0.099 kg CO₂-eq/kWh")
    print("- Updated renewable energy cost projections")
    
    # Load data
    print("\nLoading data...")
    green_data = load_green_countries_data()
    component_data = load_component_data()
    
    if green_data is None:
        print("ERROR: Could not load green countries data!")
        print("Please ensure the enhanced LCOE analysis has been run successfully.")
        return
    
    if component_data is None:
        print("ERROR: Could not load component cost data!")
        print("Please ensure lcox_results.xlsx exists in the output folder.")
        return
    
    # Validate green data format
    print(f"\nValidating green hydrogen data...")
    print(f"Loaded {len(green_data)} country-scenario combinations")
    print(f"Available scenarios: {sorted(green_data['Scenario'].unique())}")
    
    # Check green hydrogen statistics
    for threshold in ['Green_Standard', 'Green_Strict']:
        if threshold in green_data.columns:
            green_count = green_data[threshold].sum()
            total_count = len(green_data)
            print(f"{threshold}: {green_count}/{total_count} countries ({green_count/total_count*100:.1f}%)")
    
    # Validate component data
    print(f"\nValidating component cost data...")
    print(f"Available component sheets: {len(component_data)}")
    sheet_scenarios = set()
    for sheet in component_data.keys():
        parts = sheet.split('_')
        if len(parts) >= 2:
            sheet_scenarios.add('_'.join(parts[1:-1]))
    print(f"Component scenarios found: {sorted(sheet_scenarios)}")
    
    # Check for data compatibility
    green_scenarios = set(green_data['Scenario'].unique())
    compatible_scenarios = []
    
    for scenario in green_scenarios:
        # Check if we have corresponding component data
        has_pem = any('PEM' in sheet and find_component_sheet(component_data, 'PEM', scenario) for sheet in component_data.keys())
        has_rwgs = any('RWGS' in sheet and find_component_sheet(component_data, 'RWGS_FT_kerosene', scenario) for sheet in component_data.keys())
        
        if has_pem or has_rwgs:
            compatible_scenarios.append(scenario)
    
    print(f"Compatible scenarios (with both green data and component data): {compatible_scenarios}")
    
    if not compatible_scenarios:
        print("WARNING: No scenarios found with both green hydrogen data and component cost data!")
        print("Proceeding with available scenarios anyway...")
        compatible_scenarios = list(green_scenarios)[:3]  # Use first 3 scenarios
    
    # Create output directory
    output_dir = Path('figures') / 'green_countries_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use compatible scenarios, prioritizing key scenarios for green analysis
    preferred_scenarios = ['1.5 degree_2030', '1.5 degree_2050', '2 degree_2030', '2 degree_2050']
    scenarios = [s for s in preferred_scenarios if s in compatible_scenarios]
    
    # If no preferred scenarios, use first few compatible ones
    if not scenarios:
        scenarios = compatible_scenarios[:3]
    
    technologies = ['PEM', 'RWGS_FT_kerosene']
    
    print(f"\nProcessing scenarios: {scenarios}")
    print(f"Technologies: {technologies}")
    print("Year-specific CI thresholds: 2024: ≤0.077, 2030: ≤0.083, 2050: ≤0.099 kg CO₂-eq/kWh")
    
    # Create individual technology plots (focusing on key scenarios)
    for scenario in scenarios:
        for tech in technologies:
            ci_threshold = get_year_specific_threshold(scenario)
            print(f"\nProcessing {tech} - {scenario} - CI ≤ {ci_threshold:.3f} kg CO₂-eq/kWh")
            create_green_countries_cost_plot(
                component_data, green_data, tech, scenario, output_dir
            )
    
    # Create comparison plots
    for scenario in scenarios:
        ci_threshold = get_year_specific_threshold(scenario)
        print(f"\nCreating comparison plot for {scenario} - CI ≤ {ci_threshold:.3f} kg CO₂-eq/kWh")
        create_comparison_plot(component_data, green_data, scenario, output_dir)
    
    print(f"\nAll clean plots saved to: {output_dir}")
    print("\nGenerated professional plots include:")
    print("1. Clean country labels (names only)")
    print("2. Consistent professional color scheme")
    print("3. Improved legend placement")
    print("4. Better spacing and typography")
    print("5. EES journal-ready formatting")
    
    # Final data summary
    print(f"\nDATA SUMMARY:")
    print("=" * 50)
    print(f"Enhanced LCOE data source: lcoe_and_grid_prices_electrolyzer_optimized.xlsx")
    print(f"Total country-scenario combinations: {len(green_data)}")
    print(f"Scenarios processed: {len(scenarios)}")
    print(f"Technologies analyzed: {len(technologies)}")
    print("Year-specific CI thresholds applied:")
    print("  2024: ≤ 0.077 kg CO₂-eq/kWh")
    print("  2030: ≤ 0.083 kg CO₂-eq/kWh")
    print("  2050: ≤ 0.099 kg CO₂-eq/kWh")
    
    # Show green hydrogen eligibility by scenario
    for scenario in scenarios:
        scenario_data = green_data[green_data['Scenario'] == scenario]
        if len(scenario_data) > 0:
            ci_threshold = get_year_specific_threshold(scenario)
            green_countries_count = len(scenario_data[scenario_data['Carbon_Intensity'] <= ci_threshold])
            excluded_count = len(scenario_data[scenario_data['Country'].isin(get_excluded_countries())])
            total = len(scenario_data)
            print(f"  {scenario}: {green_countries_count}/{total-excluded_count} countries meet CI ≤ {ci_threshold:.3f} kg CO₂-eq/kWh")
    
    print(f"\nPlots generated for {len(scenarios) * len(technologies)} individual technology analyses")
    print(f"Plus {len(scenarios)} comparison plots")
    print("\nAnalysis complete! Use these figures for publication or presentation.")

if __name__ == "__main__":
    main()
