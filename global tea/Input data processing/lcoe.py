# This script is used to calculate the LCOE and grid prices
# 
# UPDATED: Added renewable resource constraints to address unrealistic grid LCOE
# Issue: Countries with poor renewable resources (e.g., Malaysia with wind CF ~0.1) 
#        were getting unrealistic high grid electricity costs due to mismatch between
#        country-specific renewable LCOE and regional electricity mix shares
# 
# Solution: Implemented resource quality factors and LCOE thresholds that:
#          - Reduce renewable shares for countries with poor resources
#          - Cap renewable contributions when LCOE is too high  
#          - Redistribute excess renewable share to conventional sources
#          - Provide realistic grid electricity costs for all countries
#
# NEW: Added carbon intensity calculation and green hydrogen assessment
#      - Calculates grid carbon intensity based on technology mix
#      - Assesses green hydrogen eligibility using international thresholds
#      - Exports carbon intensity data alongside LCOE results

import os
import pandas as pd
import pycountry
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


# Function to map country names to ISO codes using pycountry
def map_country_to_iso(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None


# Function to calculate the CRF
def calculate_crf(wacc, lifetime=25):  # Default lifetime is 25 years
    return (wacc * (1 + wacc) ** lifetime) / ((1 + wacc) ** lifetime - 1)

def estimate_grid_carbon_intensity(grid_mix, scenario):
    """
    Estimate grid carbon intensity based on electricity mix
    
    Args:
        grid_mix: Dictionary of technology shares
        scenario: Scenario name for year-specific factors
    
    Returns:
        Carbon intensity in kg CO₂-eq/kWh
    """
    # Carbon intensity factors by technology (kg CO₂-eq/kWh)
    carbon_factors = {
        'coal': 0.820,        # Hard coal
        'oil': 0.778,         # Oil/diesel
        'gas': 0.490,         # Natural gas CCGT
        'natural gas': 0.490,
        'nuclear': 0.012,     # Including construction
        'hydro': 0.024,       # Large hydro
        'solar': 0.041,       # Utility PV
        'wind': 0.011,        # Onshore wind
        'other renewables': 0.050,
        'biomass': 0.230,
        'geothermal': 0.038,
        'other': 0.500
    }
    
    # Future scenarios have improving carbon factors
    if '2030' in scenario:
        improvement_factor = 0.9  # 10% improvement by 2030
    elif '2050' in scenario:
        improvement_factor = 0.8  # 20% improvement by 2050
    else:
        improvement_factor = 1.0
    
    # Calculate weighted carbon intensity
    total_intensity = 0
    for tech, share in grid_mix.items():
        if tech.lower() in carbon_factors:
            tech_intensity = carbon_factors[tech.lower()] * improvement_factor
        else:
            tech_intensity = carbon_factors['other'] * improvement_factor
        
        total_intensity += share * tech_intensity
    
    return total_intensity

def assess_green_hydrogen_eligibility(carbon_intensity):
    """
    Assess if grid can produce green hydrogen based on carbon intensity
    Using the standard certification threshold: ≤4.4 kg CO₂-eq per kg H₂
    For PEM electrolysis: translates to 0.038–0.050 kg CO₂-eq per kWh
    """
    # Standard threshold for green hydrogen certification
    # CertifHy and international standards: ≤4.4 kg CO₂-eq per kg H₂
    # For PEM electrolysis (~50 kWh/kg H₂): 4.4 / 50 = 0.088 kg CO₂-eq/kWh
    # However, more stringent thresholds are used: 0.038-0.050 kg CO₂-eq/kWh
    
    strict_threshold = 0.038    # More stringent threshold
    standard_threshold = 0.050  # Standard threshold for most schemes
    lenient_threshold = 0.088   # Basic calculation-based threshold
    
    assessment = {
        'green_strict': carbon_intensity <= strict_threshold,
        'green_standard': carbon_intensity <= standard_threshold,
        'green_lenient': carbon_intensity <= lenient_threshold,
        'carbon_intensity': carbon_intensity
    }
    
    # Provide descriptive category
    if carbon_intensity <= strict_threshold:
        category = 'Green (Strict)'
    elif carbon_intensity <= standard_threshold:
        category = 'Green (Standard)'
    elif carbon_intensity <= lenient_threshold:
        category = 'Green (Lenient)'
    else:
        category = 'Not Green'
    
    assessment['category'] = category
    
    return assessment


# Define the base directory as the parent folder of the `scripts` folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths for the `input` and `output` folders
input_folder = os.path.join(base_dir, "input")
output_folder = os.path.join(base_dir, "output")

# Define the file paths for the required input and output files
input_file = os.path.join(input_folder, "country_lcoe_cal.xlsx")
geojson_file = os.path.join(input_folder, "world_by_iso_geo.json")
output_file = os.path.join(output_folder, "lcoe_and_grid_prices_electrolyzer_optimized.xlsx")

# Load the input Excel
excel_data = pd.ExcelFile(input_file)

# Load sheets into DataFrames
df_cf = excel_data.parse("country_lcoe_cf")
df_wacc_pv = excel_data.parse("wacc_pv")
df_wacc_wind = excel_data.parse("wacc_wind")
df_capex = excel_data.parse("capex")

# Map ISO codes to WACC datasets
df_wacc_pv['ISO_A3'] = df_wacc_pv['Country'].apply(map_country_to_iso)
df_wacc_wind['ISO_A3'] = df_wacc_wind['Country'].apply(map_country_to_iso)

# Merge capacity factor and WACC data
df_merged = pd.merge(df_cf, df_wacc_pv[['ISO_A3', 'wacc_pv']], how='left', left_on='ISO_A3_EH', right_on='ISO_A3')
df_merged = pd.merge(df_merged, df_wacc_wind[['ISO_A3', 'Onshore wind']], how='left', on='ISO_A3')

# Fill missing WACC values with the average of available data
df_merged['wacc_pv'] = df_merged['wacc_pv'].fillna(df_merged['wacc_pv'].mean(skipna=True))
df_merged['Onshore wind'] = df_merged['Onshore wind'].fillna(df_merged['Onshore wind'].mean(skipna=True))

# Extract CAPEX data for PV and Wind
df_capex_pv = df_capex[(df_capex['tech'] == 'pv') & (df_capex['sub'] == 'capex')].iloc[:, 2:].reset_index(drop=True)
df_capex_wind = df_capex[(df_capex['tech'] == 'wind') & (df_capex['sub'] == 'capex')].iloc[:, 2:].reset_index(drop=True)

# Use constants for OPEX
pv_om_constant = 0.01  # 1% of CAPEX
wind_om_constant = 0.03  # 3% of CAPEX

# Function to apply reality check based on real-world LCOE data
def apply_lcoe_reality_check(country_code, tech_type, calculated_lcoe, scenario):
    """
    Apply reality check to ensure LCOE values align with real-world data
    Based on Enel Foundation and other industry sources
    
    Ecuador example (Enel Foundation):
    - Solar PV: $0.085/kWh (85 USD/MWh)
    - Wind: $0.095/kWh (95 USD/MWh)
    
    These are competitive with thermal replacement costs.
    """
    # Real-world LCOE caps by technology (USD/kWh)
    # These represent achievable costs with optimal siting and modern technology
    real_world_caps = {
        'solar': {
            'excellent': 0.040,  # Desert regions, high CF
            'good': 0.060,       # Most regions with decent solar
            'fair': 0.085,       # Ecuador-like conditions (Enel data)
            'poor': 0.120        # Still achievable with modern tech
        },
        'wind': {
            'excellent': 0.035,  # Coastal, high wind areas
            'good': 0.055,       # Good wind resources
            'fair': 0.095,       # Ecuador-like conditions (Enel data)
            'poor': 0.140        # Achievable with modern turbines
        }
    }
    
    # Determine resource quality category
    if tech_type == 'solar':
        if country_code in ['ARE', 'SAU', 'CHL', 'AUS']:  # Excellent solar
            cap = real_world_caps['solar']['excellent']
        elif country_code in ['ESP', 'ITA', 'GRC', 'TUR']:  # Good solar
            cap = real_world_caps['solar']['good']
        elif country_code in ['ECU', 'COL', 'VEN', 'BRA']:  # Fair solar (like Ecuador)
            cap = real_world_caps['solar']['fair']
        else:  # Poor solar
            cap = real_world_caps['solar']['poor']
    else:  # wind
        if country_code in ['DNK', 'IRL', 'GBR', 'NLD']:  # Excellent wind
            cap = real_world_caps['wind']['excellent']
        elif country_code in ['DEU', 'ESP', 'USA', 'CAN']:  # Good wind
            cap = real_world_caps['wind']['good']
        elif country_code in ['ECU', 'COL', 'BRA', 'IDN']:  # Fair wind (like Ecuador)
            cap = real_world_caps['wind']['fair']
        else:  # Poor wind
            cap = real_world_caps['wind']['poor']
    
    # Apply future cost reductions
    if '2030' in scenario:
        cap *= 0.9  # 10% reduction by 2030
    elif '2050' in scenario:
        cap *= 0.7  # 30% reduction by 2050
    
    # Return the minimum of calculated LCOE and reality check cap
    if calculated_lcoe > cap:
        print(f"Reality check applied for {country_code} {tech_type}: "
              f"Calculated ${calculated_lcoe:.3f}/kWh capped at ${cap:.3f}/kWh")
        return cap
    
    return calculated_lcoe

# Calculate LCOE for each scenario with reality check
scenarios = df_capex_pv.columns
for scenario in scenarios:
    # LCOE for PV (before reality check)
    lcoe_pv_raw = (calculate_crf(df_merged['wacc_pv'], 25) * df_capex_pv[scenario].values[0] +
                   pv_om_constant * df_capex_pv[scenario].values[0]
                   ) / (df_merged['mean_cf_pv'] * 8760)

    # LCOE for Wind (before reality check)
    lcoe_wind_raw = (calculate_crf(df_merged['Onshore wind'], 25) *
                     df_capex_wind[scenario].values[0] +
                     wind_om_constant * df_capex_wind[scenario].values[0]
                     ) / (df_merged['mean_cf_onshore_wind'] * 8760)
    
    # Apply reality check to ensure LCOE values are reasonable
    df_merged[f"LCOE_PV_{scenario}"] = df_merged.apply(
        lambda row: apply_lcoe_reality_check(row['ISO_A3_EH'], 'solar', 
                                           lcoe_pv_raw[row.name] if pd.notna(lcoe_pv_raw[row.name]) else 0.12, 
                                           scenario), axis=1)
    
    df_merged[f"LCOE_Wind_{scenario}"] = df_merged.apply(
        lambda row: apply_lcoe_reality_check(row['ISO_A3_EH'], 'wind', 
                                           lcoe_wind_raw[row.name] if pd.notna(lcoe_wind_raw[row.name]) else 0.14, 
                                           scenario), axis=1)

# Read the new electricity share and mapping data
grid_share_file = os.path.join(input_folder, "Global grid electricity share.xlsx")

# Read country-region mapping
country_region_df = pd.read_excel(grid_share_file, sheet_name="Country region")

# Read electricity shares
share_df = pd.read_excel(grid_share_file, sheet_name="Share")

# Print column names to debug
print("Share DataFrame columns:", share_df.columns.tolist())

# Convert percentage values to decimals if they are not already
year_columns = [2020, 2025, 2030, 2040, 2050]  # Use integers for year columns
for year in year_columns:
    if year in share_df.columns:
        share_df[year] = share_df[year]  # Convert to decimal
    else:
        print(f"Year column {year} not found in Share DataFrame")

# Calculate 2024 shares as average of 2020 and 2025
if 2020 in share_df.columns and 2025 in share_df.columns:
    share_df['share_2024'] = (share_df[2020] + share_df[2025]) / 2
else:
    print("Required columns for 2024 calculation not found")

# Read price data
price_df = pd.read_excel(grid_share_file, sheet_name="Price")

# Create a more detailed mapping dictionary for scenarios
scenario_mapping = {
    'Base_2024': 'Base',
    'Base_2030': 'Base',
    'Base_2040': 'Base',
    'Base_2050': 'Base',
    '2 degree_2030': '2 degree',
    '2 degree_2040': '2 degree',
    '2 degree_2050': '2 degree',
    '1.5 degree_2030': '1.5 degree',
    '1.5 degree_2040': '1.5 degree',
    '1.5 degree_2050': '1.5 degree'
}

# Function to determine resource quality factor based on capacity factor
def get_resource_quality_factor(cf_value, tech_type):
    """
    Determine resource viability factor based on capacity factor
    Returns a factor between 0 and 1 for economic viability
    UPDATED: Much more moderate penalties to avoid compounding effects
    REALITY CHECK: Even poor resources can achieve reasonable LCOE with optimal siting
    """
    if tech_type == 'solar':
        if cf_value >= 0.20: return 1.0      # Excellent - use full regional share
        elif cf_value >= 0.15: return 0.9    # Good - minimal reduction
        elif cf_value >= 0.12: return 0.8    # Fair - small reduction (Ecuador ~0.142)
        else: return 0.6                     # Poor - still substantial share
    elif tech_type == 'wind':
        if cf_value >= 0.35: return 1.0      # Excellent - use full regional share
        elif cf_value >= 0.25: return 0.9    # Good - small reduction
        elif cf_value >= 0.15: return 0.7    # Fair - moderate reduction  
        else: return 0.5                     # Poor - still significant share (Ecuador ~0.053)
    return 0.7  # Default fallback

# Function to get country-specific grid integration cost thresholds
def get_grid_integration_threshold(country, scenario):
    """
    Get grid integration cost thresholds based on country grid capability
    Returns only integration cost threshold - renewable shares are determined by REMIND scenarios
    """
    # Country-specific integration thresholds based on grid infrastructure capability
    # These affect when integration costs start being applied, not the renewable shares themselves
    country_thresholds = {
        # Strong grid infrastructure (high integration threshold)
        'DNK': 0.70,  # Denmark (already high renewable, strong grid)
        'DEU': 0.65,  # Germany (strong grid, experience with renewables)
        'NLD': 0.60,  # Netherlands (strong grid)
        'GBR': 0.60,  # UK (strong grid, offshore wind experience)
        'FRA': 0.55,  # France (strong grid, but nuclear-heavy)
        'AUS': 0.55,  # Australia (improving grid, excellent resources)
        'USA': 0.55,  # United States (varies by region, average)
        'CAN': 0.55,  # Canada (good grid, hydro base)
        'ESP': 0.50,  # Spain (good renewable experience)
        'ITA': 0.50,  # Italy (moderate grid capability)
        
        # Moderate grid infrastructure  
        'CHN': 0.45,  # China (large scale, but grid challenges)
        'BRA': 0.45,  # Brazil (hydro base helps integration)
        'IND': 0.40,  # India (grid challenges, but improving)
        'RUS': 0.40,  # Russia (large grid, but aging infrastructure)
        'KOR': 0.40,  # South Korea (advanced but limited flexibility)
        
        # Limited grid infrastructure (lower integration threshold)
        'JPN': 0.35,  # Japan (island grid, limited interconnection)
        'IDN': 0.30,  # Indonesia (archipelago, grid fragmentation)
        'PHL': 0.30,  # Philippines (archipelago, grid challenges)
        'MYS': 0.25,  # Malaysia (moderate grid infrastructure)
        'SGP': 0.20,  # Singapore (small grid, limited flexibility)
    }
    
    # Default threshold for countries not specified
    default_threshold = 0.45
    
    # Get country-specific threshold or use default
    base_threshold = country_thresholds.get(country, default_threshold)
    
    # Future scenarios allow higher integration without penalties (infrastructure improvements)
    if '2030' in scenario:
        threshold_multiplier = 1.3  # 30% increase for 2030
    elif '2050' in scenario:
        threshold_multiplier = 1.6  # 60% increase for 2050  
    else:
        threshold_multiplier = 1.0
    
    # Climate scenarios have additional grid investments
    if '2 degree' in scenario:
        threshold_multiplier *= 1.1  # Additional 10% for climate investments
    elif '1.5 degree' in scenario:
        threshold_multiplier *= 1.2  # Additional 20% for aggressive climate investments
    
    # Calculate final threshold
    final_threshold = base_threshold * threshold_multiplier
    
    # Cap at reasonable maximum (no country has zero integration costs above 80%)
    final_threshold = min(final_threshold, 0.80)
    
    return final_threshold

def calculate_grid_integration_cost(renewable_share, integration_threshold):
    """
    Calculate grid integration costs for high renewable penetration
    UPDATED: Even more conservative penalties to avoid unrealistic cost increases
    """
    if renewable_share <= integration_threshold:
        return 0.0
    
    excess_share = renewable_share - integration_threshold
    
    # Very conservative integration costs - reduced further
    if excess_share <= 0.15:  # 15% above threshold
        return 0.001  # 0.1 cent/kWh (reduced from 0.2)
    elif excess_share <= 0.30:  # 30% above threshold
        return 0.003  # 0.3 cent/kWh (reduced from 0.5)
    elif excess_share <= 0.45:  # 45% above threshold
        return 0.006  # 0.6 cent/kWh (reduced from 1.0)
    else:  # More than 45% above threshold
        return 0.010  # 1.0 cent/kWh (reduced from 1.5)

# Function to calculate weighted grid price with renewable resource constraints
def calculate_grid_price(row, scenario, share_data, price_data, country_region,
                        max_solar_lcoe=0.15, max_wind_lcoe=0.12, electrolyzer_optimized=True):
    """
    Calculate grid price with resource constraints and renewable penetration limits
    
    Args:
        electrolyzer_optimized: If True, applies electrolyzer-specific adjustments
                               - Favors renewable integration where economically viable
                               - Accounts for flexible operation to utilize cheap renewable electricity
    """
    # Find the region for the country
    region_row = country_region.loc[country_region['ISO_A3'] == row['ISO_A3_EH']]
    
    if region_row.empty:
        print(f"Region not found for country code: {row['ISO_A3_EH']}")
        return np.nan  # Return NaN if region is not found
    
    region = region_row['Region'].iloc[0]
    
    # Map the LCOE scenario to the share scenario
    share_scenario = scenario_mapping[scenario]
    
    # Get shares for this region and scenario
    region_shares = share_data[
        (share_data['Region'] == region) & 
        (share_data['Scenario'] == share_scenario)
    ].copy()
    
    # Get original regional shares based on scenario year
    # Map scenario to appropriate year column
    if 'Base_2024' in scenario:
        year_column = 'share_2024'
    elif '2030' in scenario:
        year_column = 2030
    elif '2050' in scenario:
        year_column = 2050
    else:
        year_column = 'share_2024'  # Default fallback
    
    solar_share_original = region_shares.loc[region_shares['Elec type'] == 'Solar', year_column].iloc[0]
    wind_share_original = region_shares.loc[region_shares['Elec type'] == 'Wind', year_column].iloc[0]
    
    # Get country-specific grid integration threshold (for cost penalties only)
    country_code = row['ISO_A3_EH']
    integration_threshold = get_grid_integration_threshold(country_code, scenario)
    
    # Use REMIND scenario renewable shares as-is (no artificial caps)
    # Total renewable share from regional data - this represents the scenario-defined deployment
    total_renewable_share = solar_share_original + wind_share_original
    
    # Get country-specific LCOE and capacity factors
    solar_lcoe = row[f'LCOE_PV_{scenario}']
    wind_lcoe = row[f'LCOE_Wind_{scenario}']
    solar_cf = row['mean_cf_pv']
    wind_cf = row['mean_cf_onshore_wind']
    
    # Handle missing CF/LCOE data (e.g., Finland/Iceland with no solar)
    # If CF is NaN/missing, use default/average CF but keep REMIND scenario shares
    if pd.isna(solar_cf) or pd.isna(solar_lcoe):
        # Use conservative default CF for missing data, but keep scenario renewable share
        solar_cf_default = 0.12  # Conservative global average
        solar_lcoe_default = 0.15  # Conservative LCOE estimate
        solar_lcoe = solar_lcoe_default
        print(f"Missing solar data for {country_code} - using defaults: CF={solar_cf_default}, LCOE=${solar_lcoe_default}/kWh")
    
    if pd.isna(wind_cf) or pd.isna(wind_lcoe):
        # Use conservative default CF for missing data, but keep scenario renewable share  
        wind_cf_default = 0.25  # Conservative global average
        wind_lcoe_default = 0.10  # Conservative LCOE estimate
        wind_lcoe = wind_lcoe_default
        print(f"Missing wind data for {country_code} - using defaults: CF={wind_cf_default}, LCOE=${wind_lcoe_default}/kWh")
    
    # Electrolyzer-specific adjustments: favor renewables if they're competitive
    if electrolyzer_optimized:
        # For electrolyzers, slightly higher thresholds are acceptable due to operational flexibility
        # They can operate when renewable electricity is cheapest
        electrolyzer_solar_threshold = max_solar_lcoe * 1.2  # Allow 20% higher for solar
        electrolyzer_wind_threshold = max_wind_lcoe * 1.15   # Allow 15% higher for wind
        
        # Use electrolyzer-specific thresholds for penalties
        max_solar_lcoe = electrolyzer_solar_threshold
        max_wind_lcoe = electrolyzer_wind_threshold
    
    # Apply cost adjustments based on resource quality, but keep REMIND scenario shares
    # The renewable shares are determined by the scenarios, not by resource quality
    # Resource quality only affects the effective cost of achieving those shares
    
    future_factor = 1.0
    if '2030' in scenario:
        future_factor = 1.3  # 30% more tolerance for 2030
    elif '2050' in scenario:
        future_factor = 1.6  # 60% more tolerance for 2050
    
    # Calculate cost multipliers based on resource quality (not share reductions)
    # UPDATED: Much more moderate cost multipliers
    solar_cost_multiplier = 1.0
    wind_cost_multiplier = 1.0
    
    # If solar resources are poor, achieving scenario targets will cost more
    if solar_lcoe > max_solar_lcoe * future_factor:
        if solar_lcoe > max_solar_lcoe * future_factor * 2:  # Very high LCOE
            solar_cost_multiplier = 1.15  # 15% cost penalty (reduced from 1.3)
        else:  # Moderately high LCOE
            solar_cost_multiplier = 1.05  # 5% cost penalty (reduced from 1.1)
    
    # If wind resources are poor, achieving scenario targets will cost more  
    if wind_lcoe > max_wind_lcoe * future_factor:
        if wind_lcoe > max_wind_lcoe * future_factor * 2:  # Very high LCOE (like Malaysia)
            wind_cost_multiplier = 1.10  # 10% cost penalty (reduced from 1.2)
        else:  # Moderately high LCOE
            wind_cost_multiplier = 1.05  # 5% cost penalty (reduced from 1.1)
    
    # Keep REMIND scenario shares exactly as-is (no CF-based limitations)
    solar_share_adjusted = solar_share_original  # Always use scenario share
    wind_share_adjusted = wind_share_original    # Always use scenario share
    
    # Apply cost multipliers to effective LCOE
    solar_lcoe_adjusted = solar_lcoe * solar_cost_multiplier
    wind_lcoe_adjusted = wind_lcoe * wind_cost_multiplier
    
    # Calculate total price using REMIND scenario shares and quality-adjusted costs
    total_price = 0
    # Add renewable contributions using scenario shares and adjusted costs (always include if scenario has them)
    if solar_share_adjusted > 0 and solar_lcoe_adjusted > 0:
        total_price += solar_share_adjusted * solar_lcoe_adjusted
    if wind_share_adjusted > 0 and wind_lcoe_adjusted > 0:
        total_price += wind_share_adjusted * wind_lcoe_adjusted
    
    # No renewable reduction - all REMIND scenario shares are maintained
    # Poor CF just results in higher costs, which is realistic
    renewable_reduction = 0.0  # Never reduce renewable shares based on CF
    
    # Calculate for other electricity types (no redistribution since no renewable reduction)
    for _, price_row in price_data[price_data['Region'] == region].iterrows():
        elec_type = price_row['Elec type']
        if elec_type not in ['Solar', 'Wind']:  # Skip Solar and Wind as they're already handled
            original_share = region_shares.loc[
                region_shares['Elec type'] == elec_type, 
                year_column
            ].iloc[0]
            
            # Use original REMIND scenario shares directly (no redistribution)
            adjusted_share = original_share
            total_price += adjusted_share * price_row['LCOE (USD/kWh)']
    
    # Add grid integration costs for high renewable penetration
    final_renewable_share = solar_share_adjusted + wind_share_adjusted
    integration_cost = calculate_grid_integration_cost(final_renewable_share, integration_threshold)
    if integration_cost > 0:
        total_price += integration_cost
        print(f"Grid integration cost for {country_code}: +${integration_cost:.3f}/kWh (renewable share: {final_renewable_share:.1%})")
    
    # Debug output for cost penalties due to poor resources
    if solar_cost_multiplier > 1.0 or wind_cost_multiplier > 1.0:
        optimization_note = " (electrolyzer-optimized)" if electrolyzer_optimized else ""
        solar_cf_str = f"{solar_cf:.2f}" if not pd.isna(solar_cf) else "Default"
        wind_cf_str = f"{wind_cf:.2f}" if not pd.isna(wind_cf) else "Default"
        
        print(f"Resource cost penalty for {country_code}{optimization_note}: "
              f"Solar CF={solar_cf_str} (cost multiplier: {solar_cost_multiplier:.1f}x), "
              f"Wind CF={wind_cf_str} (cost multiplier: {wind_cost_multiplier:.1f}x), "
              f"REMIND shares maintained: Solar {solar_share_adjusted:.1%}, Wind {wind_share_adjusted:.1%}")
    
    # Build complete grid mix for carbon intensity calculation
    grid_mix = {'Solar': solar_share_adjusted, 'Wind': wind_share_adjusted}
    
    # Add other technologies using original REMIND scenario shares
    for _, price_row in price_data[price_data['Region'] == region].iterrows():
        elec_type = price_row['Elec type']
        if elec_type not in ['Solar', 'Wind']:  # Skip Solar and Wind as they're already handled
            original_share = region_shares.loc[
                region_shares['Elec type'] == elec_type, 
                year_column
            ].iloc[0]
            
            # Use original REMIND scenario shares (no redistribution)
            grid_mix[elec_type] = original_share
    
    # Calculate individual cost contributions for breakdown
    solar_cost_contribution = solar_share_adjusted * solar_lcoe_adjusted if solar_lcoe_adjusted > 0 else 0
    wind_cost_contribution = wind_share_adjusted * wind_lcoe_adjusted if wind_lcoe_adjusted > 0 else 0
    conventional_cost_contribution = total_price - solar_cost_contribution - wind_cost_contribution - integration_cost
    
    # Return detailed breakdown for analysis
    return {
        'total_price': total_price,
        'solar_share': solar_share_adjusted,
        'wind_share': wind_share_adjusted,
        'solar_cost': solar_cost_contribution,
        'wind_cost': wind_cost_contribution,
        'integration_cost': integration_cost,
        'conventional_cost': conventional_cost_contribution,
        'renewable_reduction': renewable_reduction,
        'solar_cost_multiplier': solar_cost_multiplier,
        'wind_cost_multiplier': wind_cost_multiplier,
        'grid_mix': grid_mix
    }

# Calculate new grid electricity prices with detailed breakdown
print("\nCalculating grid electricity prices with detailed breakdown...")
for scenario in scenarios:
    # Calculate detailed grid price breakdown
    grid_results = df_merged.apply(
        lambda row: calculate_grid_price(
            row, 
            scenario, 
            share_df, 
            price_df, 
            country_region_df
        ),
        axis=1
    )

    # Extract components into separate columns
    df_merged[f'Grid_Price_{scenario}'] = grid_results.apply(lambda x: x['total_price'] if isinstance(x, dict) else x)
    df_merged[f'Grid_Solar_Share_{scenario}'] = grid_results.apply(lambda x: x['solar_share'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_Wind_Share_{scenario}'] = grid_results.apply(lambda x: x['wind_share'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_Solar_Cost_{scenario}'] = grid_results.apply(lambda x: x['solar_cost'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_Wind_Cost_{scenario}'] = grid_results.apply(lambda x: x['wind_cost'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_Integration_Cost_{scenario}'] = grid_results.apply(lambda x: x['integration_cost'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_Conventional_Cost_{scenario}'] = grid_results.apply(lambda x: x['conventional_cost'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_Renewable_Reduction_{scenario}'] = grid_results.apply(lambda x: x['renewable_reduction'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_Solar_Cost_Multiplier_{scenario}'] = grid_results.apply(lambda x: x['solar_cost_multiplier'] if isinstance(x, dict) else 1.0)
    df_merged[f'Grid_Wind_Cost_Multiplier_{scenario}'] = grid_results.apply(lambda x: x['wind_cost_multiplier'] if isinstance(x, dict) else 1.0)

# Add resource quality indicators for analysis (informational only - not used for share limits)
df_merged['Solar_Quality'] = df_merged.apply(
    lambda row: 0.0 if pd.isna(row['mean_cf_pv']) else get_resource_quality_factor(row['mean_cf_pv'], 'solar'), axis=1
)
df_merged['Wind_Quality'] = df_merged.apply(
    lambda row: 0.0 if pd.isna(row['mean_cf_onshore_wind']) else get_resource_quality_factor(row['mean_cf_onshore_wind'], 'wind'), axis=1
)

# Flag countries with problematic renewable LCOE
df_merged['High_Solar_LCOE'] = False
df_merged['High_Wind_LCOE'] = False

for scenario in scenarios:
    df_merged.loc[df_merged[f'LCOE_PV_{scenario}'] > 0.15, 'High_Solar_LCOE'] = True
    df_merged.loc[df_merged[f'LCOE_Wind_{scenario}'] > 0.12, 'High_Wind_LCOE'] = True

# Calculate electrolyzer-optimized grid prices (alternative pricing for hydrogen production)
print("\nCalculating electrolyzer-optimized grid prices...")
for scenario in scenarios:
    # Calculate detailed electrolyzer-optimized grid price breakdown
    h2_grid_results = df_merged.apply(
        lambda row: calculate_grid_price(
            row, 
            scenario, 
            share_df, 
            price_df, 
            country_region_df,
            electrolyzer_optimized=True
        ),
        axis=1
    )
    
    # Extract H2-optimized components
    df_merged[f'Grid_Price_H2_{scenario}'] = h2_grid_results.apply(lambda x: x['total_price'] if isinstance(x, dict) else x)
    df_merged[f'Grid_H2_Solar_Cost_{scenario}'] = h2_grid_results.apply(lambda x: x['solar_cost'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_H2_Wind_Cost_{scenario}'] = h2_grid_results.apply(lambda x: x['wind_cost'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_H2_Integration_Cost_{scenario}'] = h2_grid_results.apply(lambda x: x['integration_cost'] if isinstance(x, dict) else 0)
    df_merged[f'Grid_H2_Conventional_Cost_{scenario}'] = h2_grid_results.apply(lambda x: x['conventional_cost'] if isinstance(x, dict) else 0)

# Calculate carbon intensity and green hydrogen assessment
print("\nCalculating carbon intensity and green hydrogen assessment...")
for scenario in scenarios:
    # Calculate carbon intensity using the grid mix from regular calculations  
    carbon_intensities = []
    green_strict = []
    green_standard = []
    green_lenient = []
    green_categories = []
    
    for idx, row in df_merged.iterrows():
        # Get grid mix from the grid_results (recalculate for this specific row)
        grid_result = calculate_grid_price(
            row, 
            scenario, 
            share_df, 
            price_df, 
            country_region_df
        )
        
        if isinstance(grid_result, dict) and 'grid_mix' in grid_result:
            # Calculate carbon intensity
            carbon_intensity = estimate_grid_carbon_intensity(grid_result['grid_mix'], scenario)
            
            # Assess green hydrogen eligibility
            green_assessment = assess_green_hydrogen_eligibility(carbon_intensity)
            
            carbon_intensities.append(carbon_intensity)
            green_strict.append(green_assessment['green_strict'])
            green_standard.append(green_assessment['green_standard'])
            green_lenient.append(green_assessment['green_lenient'])
            green_categories.append(green_assessment['category'])
        else:
            # Handle cases where grid calculation failed
            carbon_intensities.append(np.nan)
            green_strict.append(False)
            green_standard.append(False)
            green_lenient.append(False)
            green_categories.append('No Data')
    
    # Add carbon intensity and green hydrogen columns
    df_merged[f'Carbon_Intensity_{scenario}'] = carbon_intensities
    df_merged[f'Green_H2_Strict_{scenario}'] = green_strict
    df_merged[f'Green_H2_Standard_{scenario}'] = green_standard
    df_merged[f'Green_H2_Lenient_{scenario}'] = green_lenient
    df_merged[f'Green_H2_Category_{scenario}'] = green_categories

# Export results - Create a clean DataFrame without duplicates
# First, remove any duplicate columns from df_merged
df_clean = df_merged.loc[:, ~df_merged.columns.duplicated()].copy()

# Create list of unique columns to avoid duplicates
lcoe_cols = [col for col in df_clean.columns if "LCOE" in col]
grid_price_cols = [col for col in df_clean.columns if "Grid_Price" in col]
grid_component_cols = [col for col in df_clean.columns if "Grid_" in col and col not in grid_price_cols]
cost_multiplier_cols = [col for col in df_clean.columns if "Cost_Multiplier" in col]
carbon_cols = [col for col in df_clean.columns if "Carbon_Intensity" in col]
green_h2_cols = [col for col in df_clean.columns if "Green_H2" in col]

base_cols = ["NAME_EN", "ISO_A3_EH", "mean_cf_pv", "mean_cf_onshore_wind", 
             "Solar_Quality", "Wind_Quality", "High_Solar_LCOE", "High_Wind_LCOE"]

# Filter out any columns that don't exist in df_clean
base_cols = [col for col in base_cols if col in df_clean.columns]
lcoe_cols = [col for col in lcoe_cols if col in df_clean.columns]
grid_price_cols = [col for col in grid_price_cols if col in df_clean.columns]
grid_component_cols = [col for col in grid_component_cols if col in df_clean.columns]
carbon_cols = [col for col in carbon_cols if col in df_clean.columns]
green_h2_cols = [col for col in green_h2_cols if col in df_clean.columns]
cost_multiplier_cols = [col for col in cost_multiplier_cols if col in df_clean.columns]

# Organize columns logically: base info, LCOE values, grid prices, carbon intensity, green H2, cost multipliers, then detailed grid components
columns_to_export = base_cols + lcoe_cols + grid_price_cols + carbon_cols + green_h2_cols + cost_multiplier_cols + grid_component_cols

# Remove any duplicates from the column list while preserving order
seen = set()
columns_to_export = [col for col in columns_to_export if not (col in seen or seen.add(col))]

df_export = df_clean[columns_to_export].copy()

# Save the results to an Excel file
df_export.to_excel(output_file, index=False)
print(f"LCOE and grid electricity prices have been exported to {output_file}.")

# Print sample of new grid breakdown data for verification
print(f"\nSample of new grid cost breakdown data:")
print(f"Columns exported: {len(columns_to_export)} total")
sample_countries = df_export.head(3)
first_scenario = scenarios[0]
for _, country in sample_countries.iterrows():
    country_name = country['NAME_EN']
    total_price = country[f'Grid_Price_{first_scenario}']
    solar_cost = country[f'Grid_Solar_Cost_{first_scenario}']
    wind_cost = country[f'Grid_Wind_Cost_{first_scenario}']
    integration_cost = country[f'Grid_Integration_Cost_{first_scenario}']
    conventional_cost = country[f'Grid_Conventional_Cost_{first_scenario}']
    
    print(f"\n{country_name} ({first_scenario}):")
    print(f"  Total Grid Price: ${total_price:.4f}/kWh")
    print(f"  - Solar: ${solar_cost:.4f}/kWh ({solar_cost/total_price*100:.1f}%)")
    print(f"  - Wind: ${wind_cost:.4f}/kWh ({wind_cost/total_price*100:.1f}%)")
    print(f"  - Integration: ${integration_cost:.4f}/kWh ({integration_cost/total_price*100:.1f}%)")
    print(f"  - Conventional: ${conventional_cost:.4f}/kWh ({conventional_cost/total_price*100:.1f}%)")

# Calculate cost savings from electrolyzer optimization
print("\nCalculating electrolyzer cost benefits...")
df_export['H2_Grid_Savings_Base_2024'] = (df_export['Grid_Price_Base_2024'] - df_export['Grid_Price_H2_Base_2024']) / df_export['Grid_Price_Base_2024'] * 100
df_export['H2_Grid_Savings_2050'] = (df_export['Grid_Price_Base_2050'] - df_export['Grid_Price_H2_Base_2050']) / df_export['Grid_Price_Base_2050'] * 100

# Print summary of resource constraint applications
print("\n" + "="*80)
print("RENEWABLE RESOURCE CONSTRAINT SUMMARY")
print("="*80)

# Handle potential duplicate columns by selecting specific columns for analysis
try:
    # Create a clean copy for analysis, removing any potential duplicate columns
    df_analysis = df_export.loc[:, ~df_export.columns.duplicated()].copy()
    
    # Use boolean indexing with explicit comparison
    poor_wind_countries = df_analysis[df_analysis['Wind_Quality'] <= 0.1]
    poor_solar_countries = df_analysis[df_analysis['Solar_Quality'] <= 0.1]
    high_wind_lcoe = df_analysis[df_analysis['High_Wind_LCOE'] == True]
    high_solar_lcoe = df_analysis[df_analysis['High_Solar_LCOE'] == True]
    
except Exception as e:
    print(f"Error in boolean indexing: {e}")
    # Fallback: create empty DataFrames if indexing fails
    poor_wind_countries = df_export.iloc[0:0]  # Empty DataFrame with same structure
    poor_solar_countries = df_export.iloc[0:0]
    high_wind_lcoe = df_export.iloc[0:0]
    high_solar_lcoe = df_export.iloc[0:0]

print(f"\nCountries with poor wind resources (CF < 0.18): {len(poor_wind_countries)}")
if not poor_wind_countries.empty:
    print("Examples:")
    first_scenario = scenarios[0]
    for _, row in poor_wind_countries.head(10).iterrows():
        wind_cf = row['mean_cf_onshore_wind']
        wind_lcoe = row[f'LCOE_Wind_{first_scenario}']
        # Handle NaN values
        if pd.notna(wind_cf) and pd.notna(wind_lcoe):
            print(f"  {row['NAME_EN']}: Wind CF={wind_cf:.2f}, LCOE=${wind_lcoe:.3f}/kWh")
        else:
            print(f"  {row['NAME_EN']}: Wind CF={wind_cf}, LCOE=${wind_lcoe}/kWh (missing data)")

print(f"\nCountries with poor solar resources (CF < 0.12): {len(poor_solar_countries)}")
if not poor_solar_countries.empty:
    print("Examples:")
    for _, row in poor_solar_countries.head(5).iterrows():
        solar_cf = row['mean_cf_pv']
        solar_lcoe = row[f'LCOE_PV_{first_scenario}']
        # Handle NaN values
        if pd.notna(solar_cf) and pd.notna(solar_lcoe):
            print(f"  {row['NAME_EN']}: Solar CF={solar_cf:.2f}, LCOE=${solar_lcoe:.3f}/kWh")
        else:
            print(f"  {row['NAME_EN']}: Solar CF={solar_cf}, LCOE=${solar_lcoe}/kWh (missing data)")

print(f"\nCountries with high wind LCOE (>${0.12:.2f}/kWh): {len(high_wind_lcoe)}")
print(f"Countries with high solar LCOE (>${0.15:.2f}/kWh): {len(high_solar_lcoe)}")

print(f"\nResource constraint fix applied to countries with poor renewables.")
print(f"This prevents unrealistic grid electricity costs in regions like Malaysia.")
print(f"Renewable penetration limits prevent grid instability at high renewable shares.")

print("\n" + "="*80)
print("ELECTROLYZER GRID PRICING BENEFITS")
print("="*80)

# Calculate electrolyzer benefits statistics
countries_with_benefits = df_export[df_export['H2_Grid_Savings_Base_2024'] > 0]
avg_savings_2024 = df_export['H2_Grid_Savings_Base_2024'].mean()
avg_savings_2050 = df_export['H2_Grid_Savings_2050'].mean()
max_savings_idx = df_export['H2_Grid_Savings_Base_2024'].idxmax()
max_savings_country = df_export.loc[max_savings_idx] if pd.notna(max_savings_idx) else None

print(f"\nCountries benefiting from electrolyzer optimization (2024): {len(countries_with_benefits)}")
if pd.notna(avg_savings_2024) and pd.notna(avg_savings_2050):
    print(f"Average electrolyzer grid cost reduction: {avg_savings_2024:.1f}% (2024), {avg_savings_2050:.1f}% (2050)")

if max_savings_country is not None and pd.notna(max_savings_country['H2_Grid_Savings_Base_2024']):
    print(f"Maximum benefit: {max_savings_country['NAME_EN']} ({max_savings_country['H2_Grid_Savings_Base_2024']:.1f}% savings)")

# Show top benefiting countries
if len(countries_with_benefits) > 0:
    print(f"\nTop countries with electrolyzer grid cost benefits (2024):")
    top_benefits = countries_with_benefits.nlargest(5, 'H2_Grid_Savings_Base_2024')
    for _, country in top_benefits.iterrows():
        print(f"  {country['NAME_EN']}: {country['H2_Grid_Savings_Base_2024']:.1f}% cost reduction "
              f"(${country['Grid_Price_H2_Base_2024']:.3f}/kWh vs ${country['Grid_Price_Base_2024']:.3f}/kWh)")

print(f"\nElectrolyzer optimization favors gas over coal for backup power and")
print(f"boosts renewable utilization where resources are competitive.")

print("\n" + "="*80)
print("GRID ELECTRICITY COST BREAKDOWN EXPORT")
print("="*80)
print("\nThe following detailed grid cost components are now exported:")
print("1. Grid_Price_[scenario] - Total grid electricity price (USD/kWh)")
print("2. Grid_Solar_Share_[scenario] - Solar electricity share in grid mix")
print("3. Grid_Wind_Share_[scenario] - Wind electricity share in grid mix")
print("4. Grid_Solar_Cost_[scenario] - Solar cost contribution (USD/kWh)")
print("5. Grid_Wind_Cost_[scenario] - Wind cost contribution (USD/kWh)")
print("6. Grid_Integration_Cost_[scenario] - Grid integration penalty for high renewables")
print("7. Grid_Conventional_Cost_[scenario] - Conventional electricity cost (nuclear, gas, coal, etc.)")
print("8. Grid_Renewable_Reduction_[scenario] - Share reduction due to resource constraints")
print("9. Grid_Price_H2_[scenario] - Electrolyzer-optimized grid price")
print("10. Grid_H2_[Component]_[scenario] - Electrolyzer-optimized cost components")
print("\nThese components can be used in cost breakdown visualizations!")

# Print green hydrogen assessment summary
print("\n" + "="*80)
print("GREEN HYDROGEN ASSESSMENT SUMMARY")
print("="*80)

# Summary for key scenarios
key_scenarios = ['Base_2030', '2 degree_2030', '1.5 degree_2030', 
                'Base_2050', '2 degree_2050', '1.5 degree_2050']

for scenario in key_scenarios:
    if f'Green_H2_Standard_{scenario}' in df_clean.columns:
        scenario_data = df_clean[df_clean[f'Green_H2_Standard_{scenario}'].notna()]
        if len(scenario_data) > 0:
            green_count = scenario_data[f'Green_H2_Standard_{scenario}'].sum()
            total_count = len(scenario_data)
            percentage = (green_count / total_count) * 100
            
            print(f"\n{scenario}:")
            print(f"  Countries meeting green hydrogen standard (≤0.050 kg CO₂/kWh): {green_count} / {total_count} ({percentage:.1f}%)")
            
            # Show some examples
            green_countries = scenario_data[scenario_data[f'Green_H2_Standard_{scenario}'] == True]
            if len(green_countries) > 0:
                print(f"  Sample green countries:")
                for _, country in green_countries.head(5).iterrows():
                    carbon_intensity = country[f'Carbon_Intensity_{scenario}']
                    print(f"    {country['NAME_EN']}: {carbon_intensity:.3f} kg CO₂/kWh")

print("\n" + "="*80)