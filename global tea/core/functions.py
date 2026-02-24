"""
Global TEA Model - Core Functions Module

This module contains the core calculation functions for the Global Techno-Economic Analysis model.
It includes functions for loading data, calculating costs, and processing results.

"""

import os
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import geopandas as gpd


def load_input_data(filepath):
    """
    Load and preprocess all input sheets from Excel file with improved handling for
    potential data mismatches or missing fields.
    """
    try:
        # Load TEA data with proper structure and validate columns
        tea_data = pd.read_excel(filepath, sheet_name='TEA data')
        if 'tech' not in tea_data.columns or 'sub' not in tea_data.columns:
            raise ValueError("TEA data sheet must contain 'tech' and 'sub' columns.")
        
        for col in tea_data.columns:
            if col not in ['sub', 'tech']:
                # Convert percentage strings to floats for relevant columns
                if tea_data[col].dtype == object and any(tea_data[col].astype(str).str.contains('%', na=False)):
                    tea_data[col] = tea_data[col].str.rstrip('%').astype('float') / 100.0
        
        tea_data_pivoted = tea_data.pivot(
            index='tech',
            columns='sub',
            values=['Base_2024', 'Base_2030', '2 degree_2030', '1.5 degree_2030', 
                    'Base_2050', '2 degree_2050', '1.5 degree_2050']
        )
        
        # Load energy and material balance - this is the same for all countries and scenarios
        energy_balance = pd.read_excel(filepath, sheet_name='energy and material balance')
        if 'tech' not in energy_balance.columns:
            raise ValueError("Energy balance sheet must contain 'tech' column.")
            
        # Convert to numeric, replacing any non-numeric values with 0
        numeric_columns = ['m_pw', 'm_bio', 'm_iw', 'm_co2', 'm_ng', 'm_h2', 'e_elec', 'e_heat']
        for col in numeric_columns:
            if col in energy_balance.columns:
                energy_balance[col] = pd.to_numeric(energy_balance[col], errors='coerce').fillna(0)
        
        # Set tech as index
        energy_balance_pivoted = energy_balance.set_index('tech')
        
        # Load other data sheets
        f_wacc_t = pd.read_excel(filepath, sheet_name='f_wacc_t', index_col=0)
        if f_wacc_t.empty:
            raise ValueError("f_wacc_t sheet is empty or improperly formatted.")
        
        f_wacc_c = pd.read_excel(filepath, sheet_name='f_wacc_c', usecols=['ISO_A3_EH', 'f_wacc_c'])
        if f_wacc_c.empty or 'ISO_A3_EH' not in f_wacc_c.columns:
            raise ValueError("f_wacc_c sheet must contain 'ISO_A3_EH' and 'f_wacc_c' columns.")
        
        lcoe = pd.read_excel(filepath, sheet_name='lcoe', index_col='ISO_A3_EH')
        if lcoe.empty:
            raise ValueError("lcoe sheet is empty or improperly formatted.")
        
        # Ensure the country code column is properly formatted
        f_wacc_c['ISO_A3_EH'] = f_wacc_c['ISO_A3_EH'].str.strip()
        
        # Remove any rows with missing or zero WACC factors
        f_wacc_c = f_wacc_c[f_wacc_c['f_wacc_c'] > 0]
        
        # Get valid countries (those that have both WACC and LCOE data)
        valid_countries = list(set(f_wacc_c['ISO_A3_EH']).intersection(set(lcoe.index)))
        
        # Filter f_wacc_c and lcoe to only include valid countries
        f_wacc_c = f_wacc_c[f_wacc_c['ISO_A3_EH'].isin(valid_countries)]
        lcoe = lcoe.loc[valid_countries]
        
        # Load nuclear and CSP data
        nuclear = pd.read_excel(filepath, sheet_name='lcoe_nuc', usecols=['ISO_A3_EH', 'Plant type', 'lcoe_nuc'])
        if nuclear.empty:
            print("Warning: lcoe_nuc sheet is empty or improperly formatted.")
        
        csp = pd.read_excel(filepath, sheet_name='lcoe_csp')
        if csp.empty:
            print("Warning: lcoe_csp sheet is empty or improperly formatted.")
        
        # Filter nuclear and CSP data to valid countries
        nuclear = nuclear[nuclear['ISO_A3_EH'].isin(valid_countries)]
        csp = csp[csp['ISO_A3_EH'].isin(valid_countries)]
        
        # Define constants and scenarios
        price_constants = {
            'p_pw': 0.00025,  # euro/kg
            'p_bio': 0.072,   # euro/kg
            'p_iw': 0.000001, # euro/kg
            'p_ng': {
                'USA': 0.158,
                'RUS': 0.05,
                'CHN': 0.337,
                'JPN': 0.337,
                # EU countries
                'AUT': 0.337, 'BEL': 0.337, 'BGR': 0.337, 'HRV': 0.337, 'CYP': 0.337,
                'CZE': 0.337, 'DNK': 0.337, 'EST': 0.337, 'FIN': 0.337, 'FRA': 0.337,
                'DEU': 0.337, 'GRC': 0.337, 'HUN': 0.337, 'IRL': 0.337, 'ITA': 0.337,
                'LVA': 0.337, 'LTU': 0.337, 'LUX': 0.337, 'MLT': 0.337, 'NLD': 0.337,
                'POL': 0.337, 'PRT': 0.337, 'ROU': 0.337, 'SVK': 0.337, 'SVN': 0.337,
                'ESP': 0.337, 'SWE': 0.337,
                'default': 0.2
            }
        }
        
        scenarios = ['Base_2024', 'Base_2030', '2 degree_2030', '1.5 degree_2030', 
                     'Base_2050', '2 degree_2050', '1.5 degree_2050']
        
        # Define product allocation factors for multi-product technologies
        # Values are based on LHV and mass flow rates
        # Format: {tech: {product1: allocation_factor, product2: allocation_factor}}
        product_allocation = {
            'SR_FT': {'diesel': 0.7, 'kerosene': 0.4},
            'ST_FT': {'diesel': 0.65, 'kerosene': 0.35},
            'RWGS_FT': {'diesel': 0.6, 'kerosene': 0.4},
            'TG_FT': {'diesel': 0.35, 'kerosene': 0.55},
            'HVO': {'diesel': 0.8, 'kerosene': 0.8},
            'B_PYR': {'kerosene': 1.0},  # Pyrolysis focused entirely on kerosene
            'RWGS_MeOH': {'methanol': 0.50, 'DME': 0.15, 'kerosene': 0.35}
        }
        
        # Define upgrading CAPEX requirements for secondary products
        # These are the additional capital costs for upgrading to specific products
        # Values in EUR per kW of output capacity
        upgrading_capex = {
            'kerosene': {
                'SR_FT': 250,     # Additional CAPEX for upgrading FT diesel to SAF
                'ST_FT': 250,     # Additional CAPEX for upgrading FT diesel to SAF
                'RWGS_FT': 250,   # Additional CAPEX for upgrading FT diesel to SAF
                'TG_FT': 250,     # Additional CAPEX for upgrading FT diesel to SAF
                'HVO': 200,       # CAPEX for HVO to kerosene (HEFA) 
                'B_PYR': 300,     # CAPEX for pyrolysis to kerosene
                'RWGS_MeOH': 400  # Higher CAPEX for methanol to jet fuel (MTJ) process
            },
            'DME': {
                'RWGS_MeOH': 150  # CAPEX for methanol to DME process
            }
        }
        
        # Lower Heating Values (LHV) for different products in MJ/kg
        lhv_values = {
            'diesel': 43.0,
            'kerosene': 43.2,
            'methanol': 19.9,
            'DME': 28.8,
            'hydrogen': 120.0    # LHV for hydrogen
        }
        
        # Load new country-specific OPEX data
        try:
            country_opex = pd.read_excel(filepath, sheet_name='opex')
            if 'ISO_A3' not in country_opex.columns or 'value' not in country_opex.columns:
                print("Warning: OPEX sheet should contain 'ISO_A3' and 'value' columns.")
                country_opex = None
            else:
                # Set ISO_A3 as index for easier lookup
                country_opex = country_opex.set_index('ISO_A3')['value']
        except Exception as e:
            print(f"Warning: Could not load country-specific OPEX data: {str(e)}")
            country_opex = None
        
        # Load country-specific biomass price data
        try:
            biomass_prices = pd.read_excel(filepath, sheet_name='bio_lig')
            if 'ISO_A3' not in biomass_prices.columns or 'price(perkg)' not in biomass_prices.columns:
                print("Warning: bio_lig sheet should contain 'ISO_A3' and 'price(perkg)' columns.")
                biomass_prices = None
            else:
                # Set ISO_A3 as index for easier lookup
                biomass_prices = biomass_prices.set_index('ISO_A3')['price(perkg)']
                
                # Create a mapping of countries to continents for fallback prices
                country_continent = {}
                # Define continents and their countries
                continents = {
                    'Europe': ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 
                              'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 
                              'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE', 'GBR'],
                    'North_America': ['USA', 'CAN', 'MEX'],
                    'South_America': ['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'GUY', 'PRY', 'PER', 'SUR', 'URY', 'VEN'],
                    'Asia': ['CHN', 'JPN', 'KOR', 'IND', 'IDN', 'MYS', 'PHL', 'SGP', 'THA', 'VNM'],
                    'Africa': ['DZA', 'AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CPV', 'CMR', 'CAF', 'TCD', 
                              'COM', 'COG', 'CIV', 'DJI', 'EGY', 'GNQ', 'ERI', 'SWZ', 'ETH', 'GAB', 
                              'GMB', 'GHA', 'GIN', 'GNB', 'KEN', 'LSO', 'LBR', 'LBY', 'MDG', 'MWI', 
                              'MLI', 'MRT', 'MUS', 'MAR', 'MOZ', 'NAM', 'NER', 'NGA', 'RWA', 'STP', 
                              'SEN', 'SYC', 'SLE', 'SOM', 'ZAF', 'SSD', 'SDN', 'TZA', 'TGO', 'TUN', 
                              'UGA', 'ZMB', 'ZWE'],
                    'Oceania': ['AUS', 'FJI', 'KIR', 'MHL', 'FSM', 'NRU', 'NZL', 'PLW', 'PNG', 'WSM', 'SLB', 'TON', 'TUV', 'VUT']
                }
                
                # Populate the country_continent mapping
                for continent, countries in continents.items():
                    for country in countries:
                        country_continent[country] = continent
        except Exception as e:
            print(f"Warning: Could not load biomass price data: {str(e)}")
            biomass_prices = None
            country_continent = {}
        
        return {
            'tea_data': tea_data_pivoted,
            'f_wacc_t': f_wacc_t,
            'f_wacc_c': f_wacc_c,
            'energy_balance': energy_balance_pivoted,
            'lcoe': lcoe,
            'nuclear': nuclear,
            'csp': csp,
            'price_constants': price_constants,
            'scenarios': scenarios,
            'valid_countries': valid_countries,
            'product_allocation': product_allocation,
            'upgrading_capex': upgrading_capex,
            'lhv_values': lhv_values,
            'country_opex': country_opex,  # Add new country-specific OPEX data
            'biomass_prices': biomass_prices,  # Add new biomass price data
            'country_continent': country_continent  # Add country-continent mapping
        }
    except Exception as e:
        print(f"Error loading input data: {str(e)}")
        raise

   
def calculate_wacc(wacc_base, f_wacc_t, f_wacc_c, tech, country, scenario):
    """
    Calculate Weighted Average Cost of Capital (WACC) for specific technology, country and scenario.
    
    The WACC is calculated as: WACC = base_rate × technology_factor × country_factor
    
    Args:
        wacc_base (float): Base WACC rate (typically 0.08 for 8%)
        f_wacc_t (pd.DataFrame): Technology-specific WACC factors
        f_wacc_c (pd.DataFrame): Country-specific WACC factors
        tech (str): Technology identifier
        country (str): Country code (ISO 3-letter)
        scenario (str): Scenario name
        
    Returns:
        float: Calculated WACC rate for the specific technology-country-scenario combination
        
    Notes:
        - Technology factors account for technology-specific risks
        - Country factors account for country-specific economic and political risks
        - Used in calculating Capital Recovery Factor (CRF) for annualized costs
    """
    return wacc_base * f_wacc_t.loc[tech, scenario] * f_wacc_c.loc[f_wacc_c['ISO_A3_EH'] == country, 'f_wacc_c'].iloc[0]

def calculate_replacement_cost_annual(unit_inv_cost, system_lifetime, component_lifetime, interest_rate, rep_factor=0.75):
    """
    Calculates annualized replacement costs for a technology component using proper discounted cash flow.
    
    Args:
        unit_inv_cost (float): Initial investment cost [EUR].
        system_lifetime (int): Lifetime of the system [years].
        component_lifetime (int): Lifetime of the component [years].
        interest_rate (float): Interest rate [decimal, e.g., 0.08 for 8%].
        rep_factor (float): Share of replacement costs compared to full investment [-].
    
    Returns:
        float: Annualized replacement cost [EUR/year].
    """
    if component_lifetime <= 0 or system_lifetime <= 0:
        return 0
    
    # Calculate CRF for annualization
    crf = interest_rate / (1 - 1 / (1 + interest_rate) ** system_lifetime)
    
    # Replacement cost is reduced due to technological improvement and component reuse
    replacement_cost = rep_factor * unit_inv_cost
    
    c_rep_total = 0
    
    # 1. If component lifetime > system lifetime: calculate residual value
    if component_lifetime > system_lifetime:
        # Fraction of component life remaining at end of system life
        remaining_fraction = 1 - (system_lifetime / component_lifetime)
        # Present value of residual value
        residual_pv = (replacement_cost * remaining_fraction) / ((1 + interest_rate) ** system_lifetime)
        # Annualized residual value (credit)
        c_rep_total -= crf * residual_pv
        
    # 2. If component lifetime < system lifetime: calculate replacement costs
    elif component_lifetime < system_lifetime:
        # Number of full replacements needed
        full_replacements = int(system_lifetime / component_lifetime)
        
        # Calculate discounted replacement costs
        replacement_year = component_lifetime
        for i in range(full_replacements):
            # Present value of replacement at year 'replacement_year'
            replacement_pv = replacement_cost / ((1 + interest_rate) ** replacement_year)
            # Annualized replacement cost
            c_rep_total += crf * replacement_pv
            # Move to next replacement
            replacement_year += component_lifetime
        
        # Calculate residual value for the final partial replacement
        remaining_years = system_lifetime % component_lifetime
        if remaining_years > 0:
            # Fraction of final component used
            used_fraction = remaining_years / component_lifetime
            # Remaining fraction has residual value
            remaining_fraction = 1 - used_fraction
            # Present value of residual value
            residual_pv = (replacement_cost * remaining_fraction) / ((1 + interest_rate) ** system_lifetime)
            # Annualized residual value (credit)
            c_rep_total -= crf * residual_pv
    
    # 3. If component lifetime == system lifetime: no replacement needed
    else:
        c_rep_total = 0
    
    return max(0, c_rep_total)  # Ensure non-negative result

def get_electricity_price(tech, country, scenario, lcoe_data, nuclear_data, csp_data):
   """Get appropriate electricity price based on technology type with future cost reductions"""
   
   # Define cost reduction factors for nuclear and CSP in future scenarios
   # These represent learning curves and technology improvements
   nuclear_cost_factors = {
       'Base_2024': 1.0,
       'Base_2030': 0.90,         # 10% reduction by 2030 (SMR deployment, improved construction)
       'Base_2050': 0.75,         # 25% reduction by 2050 (mature SMR technology, economies of scale)
       '2 degree_2030': 0.88,     # 12% reduction (accelerated deployment in climate scenarios)
       '2 degree_2050': 0.70,     # 30% reduction (major nuclear renaissance)
       '1.5 degree_2030': 0.85,   # 15% reduction (aggressive deployment for deep decarbonization)
       '1.5 degree_2050': 0.65    # 35% reduction (breakthrough SMR and advanced reactor technologies)
   }
   
   csp_cost_factors = {
       'Base_2024': 1.0,
       'Base_2030': 0.80,         # 20% reduction by 2030 (continued learning curve)
       'Base_2050': 0.55,         # 45% reduction by 2050 (mature technology, improved materials)
       '2 degree_2030': 0.78,     # 22% reduction (accelerated deployment)
       '2 degree_2050': 0.45,     # 55% reduction (major CSP deployment)
       '1.5 degree_2030': 0.75,   # 25% reduction (aggressive deployment)
       '1.5 degree_2050': 0.36    # 64% reduction (breakthrough thermal storage and efficiency)
   }
   
   if tech in ['HTSE', 'CuCl']:
       if country in nuclear_data['ISO_A3_EH'].values:
           base_lcoe = nuclear_data.loc[nuclear_data['ISO_A3_EH'] == country, 'lcoe_nuc'].iloc[0]
           cost_factor = nuclear_cost_factors.get(scenario, 1.0)
           return base_lcoe * cost_factor
   elif tech in ['SR_FT', 'ST_FT']:
       if country in csp_data['ISO_A3_EH'].values:
           base_lcoe = csp_data.loc[csp_data['ISO_A3_EH'] == country, 'lcoe_csp'].iloc[0]
           cost_factor = csp_cost_factors.get(scenario, 1.0)
           return base_lcoe * cost_factor
   return lcoe_data.loc[country, scenario]

def calculate_cost_components(tech, country, scenario, input_data, h2_prices=None, dac_prices=None, ad_prices=None, product=None):
    """
    Calculate cost components for a specific technology, country, scenario and product.
    Only includes components that are relevant for the specific technology (non-zero inputs).
    
    Args:
        tech: Technology identifier
        country: Country code
        scenario: Scenario name
        input_data: Dictionary with all input data
        h2_prices: Dictionary with hydrogen prices
        dac_prices: Dictionary with DAC prices
        product: Specific product to calculate costs for (diesel, kerosene, methanol, DME)
                 If None, returns costs for the default product
    
    Returns:
        Dictionary with cost components, total cost, and additional metadata
    """
    t = 25  # plant lifetime in years
    cf = 8000/8760  # capacity factor
    
    h2_energy_content = 33.3  # kWh/kg (LHV of hydrogen) - correct energy content value
    
    try:
        # Get technology-specific data
        tea_data = input_data['tea_data']
        energy_balance = input_data['energy_balance']
        
        # Skip calculation for electricity-based fuels if LCOE is 0
        lcoe = get_electricity_price(tech, country, scenario, 
                                  input_data['lcoe'], 
                                  input_data['nuclear'], 
                                  input_data['csp'])
        
        # For electricity-based technologies, return None if LCOE is 0
        if lcoe == 0 and tech in ['PEM', 'AE', 'SOEC', 'HTSE', 'CuCl', 'PTM', 'RWGS_MeOH', 'RWGS_FT', 'ST_FT', 'HB', 'DAC']:
            return None
        
        # Get CAPEX, O&M, lifetime and efficiency
        capex = tea_data.loc[tech, (scenario, 'capex')]
        om = tea_data.loc[tech, (scenario, 'om')]
        lt = tea_data.loc[tech, (scenario, 'lt')]
        eff = tea_data.loc[tech, (scenario, 'eff')]
        
        # Convert lt to years if likely in hours
        if lt > 1000:
            lt = lt / 8760
        
        # Calculate WACC and CRF
        wacc = calculate_wacc(0.08, input_data['f_wacc_t'], input_data['f_wacc_c'], 
                            tech, country, scenario)
        crf = wacc/(1-1/(1+wacc)**t)
        
        # Initialize empty components dictionary
        components = {}
        
        # Determine if this is a multi-product technology and get allocation factor
        allocation_factor = 1.0  # Default for single-product technologies
        
        # Only apply allocation if a specific product is requested and technology has multiple products
        if product is not None and tech in input_data['product_allocation']:
            if product in input_data['product_allocation'][tech]:
                allocation_factor = input_data['product_allocation'][tech][product]
            else:
                print(f"Warning: Product '{product}' not defined for technology '{tech}'. Using default allocation.")
        
        # Only add electricity cost if electricity is used
        e_elec = energy_balance.loc[tech, 'e_elec']
        if e_elec > 0:
            # Apply efficiency scaling for all technologies
            components['c_elec'] = (e_elec / eff) * lcoe
        
        # Only add heat cost if heat is used
        e_heat = energy_balance.loc[tech, 'e_heat']
        if e_heat > 0:
            # Apply efficiency scaling for all technologies
            # Heat cost factor depends on technology type
            if tech in ['SR_FT', 'ST_FT']:
                # For CSP technologies, direct thermal heat is much more efficient
                # 0.3 factor represents direct solar thermal heat cost
                heat_cost_factor = 0.3
            else:
                # For other technologies, 1.2 factor represents transfer efficiency from electricity to heat
                heat_cost_factor = 1.2
            
            components['c_heat'] = (e_heat / eff) * lcoe * heat_cost_factor
        
        # Get material prices
        prices = input_data['price_constants']
        
        # Only add water costs if water is used
        m_pw = energy_balance.loc[tech, 'm_pw']
        if m_pw > 0:
            components['c_pw'] = (m_pw / eff) * prices['p_pw']
            
        m_iw = energy_balance.loc[tech, 'm_iw']
        if m_iw > 0:
            components['c_iw'] = (m_iw / eff) * prices['p_iw']
        
        # Only add biomass cost if biomass is used - WITH COUNTRY-SPECIFIC PRICES
        m_bio = energy_balance.loc[tech, 'm_bio']
        if m_bio > 0:
            # Check if we have biomass price data
            if 'biomass_prices' in input_data and input_data['biomass_prices'] is not None:
                # Determine biomass price based on country and technology
                biomass_price = None
                
                # First, try to get the country-specific price
                if country in input_data['biomass_prices'].index:
                    biomass_price = input_data['biomass_prices'].loc[country]
                else:
                    # If country not found, try to find a country in the same continent
                    if 'country_continent' in input_data and country in input_data['country_continent']:
                        continent = input_data['country_continent'][country]
                        # Find countries in the same continent that have biomass prices
                        continent_countries = [c for c in input_data['country_continent'] 
                                             if input_data['country_continent'][c] == continent 
                                             and c in input_data['biomass_prices'].index]
                        
                        if continent_countries:
                            # Use the average price of countries in the same continent
                            continent_prices = [input_data['biomass_prices'].loc[c] for c in continent_countries]
                            biomass_price = sum(continent_prices) / len(continent_prices)
                
                # If still no price, use the default
                if biomass_price is None:
                    biomass_price = prices['p_bio']
                
                # For AD and HTL, the price is 1.5 times cheaper (divide by 1.5)
                if tech in ['AD', 'HTL']:
                    biomass_price = biomass_price / 1.5
                
                components['c_bio'] = (m_bio / eff) * biomass_price
            else:
                # Use default price from price_constants
                components['c_bio'] = (m_bio / eff) * prices['p_bio']
        
        # Only add natural gas cost if natural gas is used
        m_ng = energy_balance.loc[tech, 'm_ng']
        if m_ng > 0:
            # For SR-FT technology, use biomethane from AD instead of conventional natural gas
            if tech == 'SR_FT' and ad_prices is not None:
                if scenario in ad_prices and country in ad_prices[scenario]:
                    # AD price is in EUR/kWh of biomethane fuel, convert to EUR/kg
                    # Biomethane has energy content of ~50 MJ/kg = ~13.9 kWh/kg (LHV)
                    biomethane_energy_content = 13.9  # kWh/kg (LHV of biomethane)
                    # Convert EUR/kWh to EUR/kg by multiplying by energy content
                    biomethane_cost_per_kg = ad_prices[scenario][country] * biomethane_energy_content
                    components['c_ng'] = (m_ng / eff) * biomethane_cost_per_kg
                    # Using AD-derived biomethane price for SR-FT technology
                else:
                    # Fallback to conventional natural gas price if no AD price available
                    p_ng = prices['p_ng'].get(country, prices['p_ng']['default'])
                    components['c_ng'] = (m_ng / eff) * p_ng
                    print(f"WARNING: SR-FT using conventional NG price for {country}: {p_ng:.4f} EUR/kg")
            else:
                # For all other technologies, use conventional natural gas price
                p_ng = prices['p_ng'].get(country, prices['p_ng']['default'])
                components['c_ng'] = (m_ng / eff) * p_ng
        
        # Only add CO2 cost if CO2 is used - use the calculated DAC cost
        m_co2 = energy_balance.loc[tech, 'm_co2']
        if m_co2 > 0 and dac_prices is not None:
            if scenario in dac_prices and country in dac_prices[scenario]:
                # Apply efficiency scaling for all technologies
                components['c_co2'] = (m_co2 / eff) * dac_prices[scenario][country]
        
        # Only add hydrogen cost if hydrogen is used and PEM prices are available
        m_h2 = energy_balance.loc[tech, 'm_h2']
        if m_h2 > 0 and h2_prices is not None:
            if (scenario in h2_prices and country in h2_prices[scenario] and 
                'PEM' in h2_prices[scenario][country]):
                # Convert hydrogen price: LCOH is in EUR/kWh, but m_h2 is in kg
                # We multiply LCOH by energy content (kWh/kg) to get EUR/kg
                # For hydrogen, 1 kg contains approximately 33.3 kWh (LHV)
                h2_cost_per_kg = h2_prices[scenario][country]['PEM'] * h2_energy_content
                
                # For upgrading technologies (HVO, HTL, B_PYR), hydrogen is used for upgrading/hydrogenation
                # and should not be subject to plant efficiency scaling
                if tech in ['HVO', 'HTL', 'B_PYR']:
                    components['c_h2'] = m_h2 * h2_cost_per_kg * allocation_factor
                else:
                    # For synthesis technologies, hydrogen is a stoichiometric requirement for the entire process
                    # Apply efficiency scaling but NOT allocation factor (hydrogen is needed for total process)
                    components['c_h2'] = (m_h2 / eff) * h2_cost_per_kg
                
                # Hydrogen cost calculation completed
        
        # Add CO2 storage costs for any technology that uses CO2 (not just PtX fuels)
        # This should be considered independently from hydrogen
        if m_co2 > 0 and tech != 'HB':  # HB (ammonia production) doesn't use CO2
            components['c_co2_storage'] = 0.02 * energy_balance.loc[tech, 'm_co2']
        
        # Calculate REPEX using improved discounted cash flow method
        # Define replacement factors based on technology type
        if tech in ['SOEC', 'HTSE', 'CuCl']:
            rep_factor = 0.3  # 30% replacement cost (70% is SOEC stack that gets replaced)
        elif tech in ['PEM', 'AE']:
            rep_factor = 0.5  # 50% replacement cost (50% is stack/electrolyzer core)
        else:
            rep_factor = 0.03  # 3% replacement cost for non-electrolyzer technologies
        
        # For electrolyzer technologies, component lifetime is operating hours based
        if tech in ['PEM', 'AE', 'SOEC', 'HTSE', 'CuCl']:
            # Convert operating hours lifetime to years based on capacity factor
            component_lifetime_years = lt / (cf * 8760)
        else:
            # For other technologies, use the lifetime directly as years
            component_lifetime_years = lt
        
        # Calculate annualized replacement cost using improved method
        repex_annual = calculate_replacement_cost_annual(
            unit_inv_cost=capex,
            system_lifetime=t,  # 25 years
            component_lifetime=component_lifetime_years,
            interest_rate=wacc,
            rep_factor=rep_factor
        )
        
        # Convert to EUR/kWh by dividing by annual energy production
        if repex_annual > 0:
            # Apply efficiency scaling and allocation factor
            repex = repex_annual / (8760 * cf * eff * allocation_factor)
            
            # Apply reasonable cap (70% of annualized CAPEX)
            max_repex = 0.7 * crf * capex / (8760 * cf * eff * allocation_factor)
            repex = min(repex, max_repex)
            
            components['repex'] = repex

        # Add upgrading CAPEX if a specific product is requested
        upgrade_capex = 0
        if product is not None and product in input_data['upgrading_capex'] and tech in input_data['upgrading_capex'][product]:
            upgrade_capex = input_data['upgrading_capex'][product][tech]
            components['c_upgrade'] = crf * upgrade_capex / (8760 * cf * allocation_factor)

        # INCLUDE COUNTRY-SPECIFIC OPEX FACTOR
        opex_factor = 1.0  # Default factor
        if 'country_opex' in input_data and input_data['country_opex'] is not None and country in input_data['country_opex'].index:
            opex_factor = input_data['country_opex'].loc[country]
        
        # Always include CAPEX and O&M costs - using formula: CRF*(CAPEX+FOC)/(8760*CF*η)
        # Apply efficiency in denominator for ALL technologies
        components['c_capex'] = crf * capex / (8760 * cf * eff * allocation_factor)
        components['c_om'] = om * capex * opex_factor / (8760 * cf * eff * allocation_factor)
        
        # Add hydrogen storage cost for PtX fuels (only if hydrogen is used)
        if tech in ['PTM', 'RWGS_MeOH', 'RWGS_FT', 'ST_FT', 'HB', 'HTL', 'HVO', 'B_PYR'] and m_h2 > 0:
            # Only add hydrogen storage if we're also accounting for hydrogen costs
            if 'c_h2' in components:
                # Hydrogen storage costs in EUR/kWh by scenario year
                # These are costs per kWh of hydrogen stored
                storage_cost_per_kwh = {
                    '2022': 0.82,   # EUR/kWh in 2022
                    '2030': 0.60,   # EUR/kWh in 2030  
                    '2050': 0.27    # EUR/kWh in 2050
                }
                
                # Determine scenario year
                scenario_year = "2022"  # Default
                if "2030" in scenario:
                    scenario_year = "2030"
                elif "2050" in scenario:
                    scenario_year = "2050"
                
                # Calculate storage cost proportional to hydrogen consumption
                # Convert hydrogen mass (kg) to energy content (kWh) and apply storage cost
                h2_energy_content_kwh = m_h2 * h2_energy_content * 0.0000313
                components['c_h2_storage'] = h2_energy_content_kwh * storage_cost_per_kwh[scenario_year]
        
        # Calculate total LCOX
        total_lcox = sum(components.values())
        
        return {
            'components': components,  # Only includes non-zero components
            'total': total_lcox,
            'wacc': wacc,
            'allocation_factor': allocation_factor,
            'product': product if product is not None else 'default'
        }
        
    except Exception as e:
        print(f"Error calculating costs for {tech} in {country} ({scenario}): {str(e)}")
        return None

def calculate_cost_components_monte_carlo(tech, country, scenario, input_data, h2_prices=None, dac_prices=None, ad_prices=None, product=None, parameter_samples=None):
    """
    Monte Carlo version of cost component calculation
    
    Args:
        tech: Technology identifier
        country: Country code
        scenario: Scenario name
        input_data: Dictionary with all input data
        h2_prices: Dictionary with hydrogen prices
        dac_prices: Dictionary with DAC prices
        product: Specific product to calculate costs for
        parameter_samples: Dictionary with parameter samples from Monte Carlo simulation
        
    Returns:
        Array of LCOX values from Monte Carlo simulation
    """
    if parameter_samples is None:
        # No samples provided, return deterministic result
        result = calculate_cost_components(tech, country, scenario, input_data, h2_prices, dac_prices, ad_prices, product)
        return result['total'] if result else None
    
    num_samples = len(parameter_samples['capex'])
    energy_balance = input_data['energy_balance']
    t = 25  # plant lifetime in years
    h2_energy_content = 33.3  # kWh/kg (LHV of hydrogen)
    
    # Extract year from scenario for uncertainty scaling
    scenario_year = "2022"
    if "2030" in scenario:
        scenario_year = "2030"
    elif "2050" in scenario:
        scenario_year = "2050"
    
    try:
        # Initialize arrays for results
        total_lcox = np.zeros(num_samples)
        
        # Generate hydrogen price uncertainty samples if needed
        h2_price_samples = None
        if h2_prices is not None and scenario in h2_prices and country in h2_prices[scenario] and 'PEM' in h2_prices[scenario][country]:
            base_h2_price = h2_prices[scenario][country]['PEM']
            # Generate hydrogen price samples with increasing uncertainty for future years
            from monte_carlo import generate_hydrogen_price_distribution
            h2_price_samples = generate_hydrogen_price_distribution(base_h2_price, scenario_year, num_samples)
        
        # Generate DAC price uncertainty samples if needed
        dac_price_samples = None
        if dac_prices is not None and scenario in dac_prices and country in dac_prices[scenario]:
            base_dac_price = dac_prices[scenario][country]
            # Generate DAC price samples with increasing uncertainty for future years
            from monte_carlo import generate_dac_price_distribution
            dac_price_samples = generate_dac_price_distribution(base_dac_price, scenario_year, num_samples)
        
        # Get country-specific OPEX factor
        opex_factor = 1.0  # Default factor
        if 'country_opex' in input_data and input_data['country_opex'] is not None and country in input_data['country_opex'].index:
            opex_factor = input_data['country_opex'].loc[country]
            # Add small uncertainty to the OPEX factor for Monte Carlo
            opex_factor_samples = np.random.normal(opex_factor, opex_factor * 0.05, num_samples)
            opex_factor_samples = np.clip(opex_factor_samples, opex_factor * 0.9, opex_factor * 1.1)
        else:
            opex_factor_samples = np.ones(num_samples)
        
        # Get country-specific biomass price if applicable
        biomass_price = None
        if 'biomass_prices' in input_data and input_data['biomass_prices'] is not None:
            if country in input_data['biomass_prices'].index:
                biomass_price = input_data['biomass_prices'].loc[country]
            elif 'country_continent' in input_data and country in input_data['country_continent']:
                continent = input_data['country_continent'][country]
                continent_countries = [c for c in input_data['country_continent'] 
                                      if input_data['country_continent'][c] == continent 
                                      and c in input_data['biomass_prices'].index]
                
                if continent_countries:
                    continent_prices = [input_data['biomass_prices'].loc[c] for c in continent_countries]
                    biomass_price = sum(continent_prices) / len(continent_prices)
        
        # If no biomass price found, use default
        if biomass_price is None:
            biomass_price = input_data['price_constants']['p_bio']
        
        # For AD and HTL, the price is 1.5 times cheaper
        if tech in ['AD', 'HTL']:
            biomass_price = biomass_price / 1.5
        
        # Add uncertainty to biomass price for Monte Carlo
        biomass_factor = 1.0
        if scenario_year == "2030":
            biomass_factor_samples = np.random.triangular(0.9, 1.0, 1.2, num_samples)
        elif scenario_year == "2050":
            biomass_factor_samples = np.random.triangular(0.8, 1.0, 1.3, num_samples)
        else:
            biomass_factor_samples = np.random.triangular(0.95, 1.0, 1.1, num_samples)
        
        biomass_price_samples = biomass_price * biomass_factor_samples
        
        # Use a vectorized approach where possible
        for i in range(num_samples):
            # Get parameter values for this sample
            capex = parameter_samples['capex'][i]
            om = parameter_samples['om'][i]
            lt = parameter_samples['lt'][i]
            eff = parameter_samples['eff'][i]
            cf = parameter_samples['cf'][i]
            
            # Convert lt to years if likely in hours
            if lt > 1000:
                lt = lt / 8760
            
            # Calculate WACC and CRF for this sample
            wacc = 0.08 * parameter_samples['f_wacc_t'][i] * parameter_samples['f_wacc_c'][i]
            crf = wacc/(1-1/(1+wacc)**t)
            
            # Determine allocation factor
            allocation_factor = 1.0
            if product is not None and tech in input_data['product_allocation']:
                if product in input_data['product_allocation'][tech]:
                    allocation_factor = input_data['product_allocation'][tech][product]
            
            # Initialize cost components for this sample
            components_sum = 0
            
            # Electricity cost
            e_elec = energy_balance.loc[tech, 'e_elec'] if 'e_elec' in energy_balance.columns else 0
            if e_elec > 0 and 'lcoe' in parameter_samples:
                # Apply efficiency scaling for all technologies
                components_sum += (e_elec / eff) * parameter_samples['lcoe'][i]
            
            # Heat cost
            e_heat = energy_balance.loc[tech, 'e_heat'] if 'e_heat' in energy_balance.columns else 0
            if e_heat > 0 and 'lcoe' in parameter_samples:
                # Apply efficiency scaling for all technologies
                # Heat cost factor depends on technology type
                if tech in ['SR_FT', 'ST_FT']:
                    # For CSP technologies, direct thermal heat is much more efficient
                    # 0.3 factor represents direct solar thermal heat cost
                    heat_cost_factor = 0.3
                else:
                    # For other technologies, 1.2 factor represents transfer efficiency from electricity to heat
                    heat_cost_factor = 1.2
                
                components_sum += (e_heat / eff) * parameter_samples['lcoe'][i] * heat_cost_factor
            
            # Process water cost
            m_pw = energy_balance.loc[tech, 'm_pw'] if 'm_pw' in energy_balance.columns else 0
            if m_pw > 0 and 'p_pw' in parameter_samples:
                components_sum += (m_pw / eff) * parameter_samples['p_pw'][i]
            
            # Input water cost
            m_iw = energy_balance.loc[tech, 'm_iw'] if 'm_iw' in energy_balance.columns else 0
            if m_iw > 0 and 'p_iw' in parameter_samples:
                components_sum += (m_iw / eff) * parameter_samples['p_iw'][i]
            
            # Biomass cost - now with country-specific prices
            m_bio = energy_balance.loc[tech, 'm_bio'] if 'm_bio' in energy_balance.columns else 0
            if m_bio > 0:
                components_sum += (m_bio / eff) * biomass_price_samples[i]
            
            # Natural gas cost
            m_ng = energy_balance.loc[tech, 'm_ng'] if 'm_ng' in energy_balance.columns else 0
            if m_ng > 0:
                # For SR-FT technology, use biomethane from AD instead of conventional natural gas
                if tech == 'SR_FT' and ad_prices is not None:
                    if scenario in ad_prices and country in ad_prices[scenario]:
                        # Use AD-derived biomethane price with some uncertainty for Monte Carlo
                        base_biomethane_price = ad_prices[scenario][country]
                        # Add uncertainty to the biomethane price based on scenario year
                        if scenario_year == "2030":
                            biomethane_factor = np.random.triangular(0.9, 1.0, 1.2)
                        elif scenario_year == "2050":
                            biomethane_factor = np.random.triangular(0.8, 1.0, 1.3)
                        else:
                            biomethane_factor = np.random.triangular(0.95, 1.0, 1.1)
                        
                        biomethane_price_sample = base_biomethane_price * biomethane_factor
                        biomethane_energy_content = 13.9  # kWh/kg (LHV of biomethane)
                        # Convert EUR/kWh to EUR/kg by multiplying by energy content
                        biomethane_cost_per_kg = biomethane_price_sample * biomethane_energy_content
                        components_sum += (m_ng / eff) * biomethane_cost_per_kg
                    else:
                        # Fallback to conventional natural gas price if no AD price available
                        if 'p_ng' in parameter_samples:
                            components_sum += (m_ng / eff) * parameter_samples['p_ng'][i]
                else:
                    # For all other technologies, use conventional natural gas price
                    if 'p_ng' in parameter_samples:
                        components_sum += (m_ng / eff) * parameter_samples['p_ng'][i]
            
            # CO2 cost with uncertainty
            m_co2 = energy_balance.loc[tech, 'm_co2'] if 'm_co2' in energy_balance.columns else 0
            if m_co2 > 0 and dac_prices is not None:
                if scenario in dac_prices and country in dac_prices[scenario]:
                    # Use DAC price sample if available, otherwise use base price
                    dac_price = dac_price_samples[i] if dac_price_samples is not None else dac_prices[scenario][country]
                    # Apply efficiency scaling for all technologies
                    components_sum += (m_co2 / eff) * dac_price
            
            # Hydrogen cost with uncertainty
            m_h2 = energy_balance.loc[tech, 'm_h2'] if 'm_h2' in energy_balance.columns else 0
            if m_h2 > 0 and h2_prices is not None:
                if (scenario in h2_prices and country in h2_prices[scenario] and 
                    'PEM' in h2_prices[scenario][country]):
                    # Use H2 price sample if available, otherwise use base price
                    h2_price = h2_price_samples[i] if h2_price_samples is not None else h2_prices[scenario][country]['PEM']
                    h2_cost_per_kg = h2_price * h2_energy_content
                    
                    # For upgrading technologies (HVO, HTL, B_PYR), hydrogen is used for upgrading/hydrogenation
                    # and should not be subject to plant efficiency scaling
                    if tech in ['HVO', 'HTL', 'B_PYR']:
                        components_sum += m_h2 * h2_cost_per_kg * allocation_factor
                    else:
                        # For synthesis technologies, hydrogen is a stoichiometric requirement for the entire process
                        # Apply efficiency scaling but NOT allocation factor (hydrogen is needed for total process)
                        components_sum += (m_h2 / eff) * h2_cost_per_kg
            
            # CO2 storage cost (with some uncertainty proportional to CO2 price)
            if m_co2 > 0 and tech != 'HB':
                # Add some uncertainty to CO2 storage cost based on scenario year
                storage_cost_base = 0.02  # EUR/kg
                storage_cost_factor = 1.0
                if scenario_year == "2030":
                    storage_cost_factor = np.random.triangular(0.9, 1.0, 1.2)
                elif scenario_year == "2050":
                    storage_cost_factor = np.random.triangular(0.8, 1.0, 1.4)
                components_sum += storage_cost_base * storage_cost_factor * m_co2
            
            # Calculate REPEX using improved discounted cash flow method
            # Define replacement factors based on technology type
            if tech in ['SOEC', 'HTSE', 'CuCl']:
                rep_factor = 0.3  # 30% replacement cost (70% is SOEC stack that gets replaced)
            elif tech in ['PEM', 'AE']:
                rep_factor = 0.5  # 50% replacement cost (50% is stack/electrolyzer core)
            else:
                rep_factor = 0.03  # 3% replacement cost for non-electrolyzer technologies
            
            # For electrolyzer technologies, component lifetime is operating hours based
            if tech in ['PEM', 'AE', 'SOEC', 'HTSE', 'CuCl']:
                # Convert operating hours lifetime to years based on capacity factor
                component_lifetime_years = lt / (cf * 8760)
            else:
                # For other technologies, use the lifetime directly as years
                component_lifetime_years = lt
            
            # Calculate annualized replacement cost using improved method
            repex_annual = calculate_replacement_cost_annual(
                unit_inv_cost=capex,
                system_lifetime=t,  # 25 years
                component_lifetime=component_lifetime_years,
                interest_rate=wacc,
                rep_factor=rep_factor
            )
            
            # Convert to EUR/kWh by dividing by annual energy production
            if repex_annual > 0:
                # Apply efficiency scaling and allocation factor
                repex = repex_annual / (8760 * cf * eff * allocation_factor)
                
                # Apply reasonable cap (70% of annualized CAPEX)
                max_repex = 0.7 * crf * capex / (8760 * cf * eff * allocation_factor)
                repex = min(repex, max_repex)
                
                components_sum += repex
            
            # Add upgrading CAPEX if needed
            upgrade_capex = 0
            if product is not None and product in input_data['upgrading_capex'] and tech in input_data['upgrading_capex'][product]:
                upgrade_capex = input_data['upgrading_capex'][product][tech]
                # Add uncertainty to upgrading CAPEX based on scenario year
                upgrade_factor = 1.0
                if scenario_year == "2030":
                    upgrade_factor = np.random.triangular(0.9, 1.0, 1.2)
                elif scenario_year == "2050":
                    upgrade_factor = np.random.triangular(0.8, 1.0, 1.3)
                components_sum += crf * upgrade_capex * upgrade_factor / (8760 * cf * allocation_factor)
            
            # CAPEX and O&M costs - using formula: CRF*(CAPEX+FOC)/(8760*CF*η)
            components_sum += crf * capex / (8760 * cf * eff * allocation_factor)
            components_sum += om * capex * opex_factor_samples[i] / (8760 * cf * eff * allocation_factor)
            
            # Hydrogen storage with uncertainty - proportional to hydrogen consumption
            if tech in ['PTM', 'RWGS_MeOH', 'RWGS_FT', 'ST_FT', 'HB', 'HTL', 'HVO', 'B_PYR'] and m_h2 > 0:
                if h2_prices is not None and scenario in h2_prices and country in h2_prices[scenario] and 'PEM' in h2_prices[scenario][country]:
                    # Hydrogen storage costs in EUR/kWh by scenario year with uncertainty
                    if scenario_year == "2022":
                        storage_cost_per_kwh = np.random.triangular(0.075, 0.082, 0.090, 1)[0]  # ±10% uncertainty around 0.082
                    elif scenario_year == "2030":
                        storage_cost_per_kwh = np.random.triangular(0.050, 0.060, 0.075, 1)[0]  # ±17% uncertainty around 0.060
                    elif scenario_year == "2050":
                        storage_cost_per_kwh = np.random.triangular(0.020, 0.027, 0.035, 1)[0]  # ±26% uncertainty around 0.027
                    else:
                        storage_cost_per_kwh = np.random.triangular(0.075, 0.082, 0.090, 1)[0]  # Default to 2022 values
                    
                    # Calculate storage cost proportional to hydrogen consumption  
                    # Convert hydrogen mass (kg) to energy content (kWh) and apply storage cost
                    h2_energy_content_kwh = m_h2 * h2_energy_content * 0.0000313
                    components_sum += h2_energy_content_kwh * storage_cost_per_kwh
            
            # Store the total LCOX for this sample
            total_lcox[i] = components_sum
        
        return total_lcox
        
    except Exception as e:
        print(f"Error in Monte Carlo calculation for {tech} in {country} ({scenario}): {str(e)}")
        return None

def save_results_to_excel(results, output_file):
    """
    Save results to Excel file with standardized sheet names
    """
    try:
        # Create a mapping for scenario name standardization
        scenario_mapping = {
            'Base_2024': 'Base24',
            'Base_2030': 'Base30',
            'Base_2050': 'Base50',
            '2 degree_2030': '2deg30',
            '2 degree_2050': '2deg50',
            '1.5 degree_2030': '15deg30',
            '1.5 degree_2050': '15deg50'
        }

        with pd.ExcelWriter(output_file) as writer:
            # Save LCOX values
            for scenario in results['lcox_values'].keys():
                sheet_name = f'LCOX_{scenario_mapping.get(scenario, scenario[:10])}'
                df = pd.DataFrame(results['lcox_values'][scenario])
                df.to_excel(writer, sheet_name=sheet_name)
            
            # Save detailed cost components
            print("\nDebug: Cost Components Analysis")
            print("-" * 50)
            
            for scenario in results['cost_components'].keys():
                for tech in results['cost_components'][scenario].keys():
                    print(f"\nTechnology: {tech}, Scenario: {scenario}")
                    
                    tech_data = results['cost_components'][scenario][tech]
                    if tech_data:
                        # Get first country's data to see available components
                        first_country = next(iter(tech_data.keys()))
                        components = tech_data[first_country].keys()
                        print(f"Available components: {list(components)}")
                        
                        df = pd.DataFrame.from_dict(tech_data, orient='index')
                        print(f"Number of countries: {len(df)}")
                        print(f"Sample data for first country:")
                        print(df.head(1))
                        
                        # Create standardized sheet name
                        scenario_code = scenario_mapping.get(scenario, scenario[:10])
                        tech_code = tech.replace(' ', '').replace('-', '')[:10]
                        sheet_name = f'Comp_{scenario_code}_{tech_code}'
                        
                        df.to_excel(writer, sheet_name=sheet_name)
                    else:
                        print(f"No data available for {tech}")
            
            # Save component averages
            print("\nSaving component averages...")
            avg_components = {}
            for scenario in results['cost_components'].keys():
                avg_components[scenario] = {}
                for tech in results['cost_components'][scenario].keys():
                    if results['cost_components'][scenario][tech]:
                        tech_data = results['cost_components'][scenario][tech]
                        components = next(iter(tech_data.values())).keys()
                        
                        avg_components[scenario][tech] = {
                            comp: np.mean([
                                country_data[comp] 
                                for country_data in tech_data.values() 
                                if comp in country_data
                            ]) 
                            for comp in components
                        }
            
            # Save averages with standardized scenario names
            avg_df = pd.DataFrame.from_dict(
                {(scenario_mapping.get(scen, scen[:10]), tech): comps 
                 for scen, techs in avg_components.items() 
                 for tech, comps in techs.items()},
                orient='index'
            )
            avg_df.index = pd.MultiIndex.from_tuples(avg_df.index, names=['Scenario', 'Technology'])
            avg_df.to_excel(writer, sheet_name='Component_Averages')
            
            # Save hydrogen prices
            if 'h2_prices' in results and results['h2_prices']:
                h2_df = pd.DataFrame()
                for scenario in results['h2_prices']:
                    scenario_df = pd.DataFrame()
                    for country in results['h2_prices'][scenario]:
                        if 'PEM' in results['h2_prices'][scenario][country]:
                            scenario_df.loc[country, 'PEM_EUR_per_kWh'] = results['h2_prices'][scenario][country]['PEM']
                            # Convert to EUR/kg for easier interpretation
                            scenario_df.loc[country, 'PEM_EUR_per_kg'] = results['h2_prices'][scenario][country]['PEM'] * 33.3
        
                    if not scenario_df.empty:
                        h2_df = pd.concat([h2_df, scenario_df.add_suffix(f'_{scenario}')], axis=1)
    
                if not h2_df.empty:
                    h2_df.to_excel(writer, sheet_name='Hydrogen_Prices')

            # Save CO2 capture prices
            if 'dac_prices' in results and results['dac_prices']:
                co2_df = pd.DataFrame()
                for scenario in results['dac_prices']:
                    scenario_data = []
                    for country, price in results['dac_prices'][scenario].items():
                        scenario_data.append({'Country': country, 'Price_EUR_per_kWh': price})
        
                    if scenario_data:
                        scenario_df = pd.DataFrame(scenario_data).set_index('Country')
                        co2_df = pd.concat([co2_df, scenario_df.add_suffix(f'_{scenario}')], axis=1)
    
                if not co2_df.empty:
                    co2_df.to_excel(writer, sheet_name='CO2_Capture_Prices')

            # Save AD (biomethane) prices
            if 'ad_prices' in results and results['ad_prices']:
                ad_df = pd.DataFrame()
                for scenario in results['ad_prices']:
                    scenario_data = []
                    for country, price in results['ad_prices'][scenario].items():
                        scenario_data.append({'Country': country, 'Price_EUR_per_kWh': price})
        
                    if scenario_data:
                        scenario_df = pd.DataFrame(scenario_data).set_index('Country')
                        ad_df = pd.concat([ad_df, scenario_df.add_suffix(f'_{scenario}')], axis=1)
    
                if not ad_df.empty:
                    ad_df.to_excel(writer, sheet_name='Biomethane_Prices')
            
    except Exception as e:
        print(f"Error saving results to Excel: {str(e)}")
        raise

def process_lcox_calculations(input_data, output_file, scenarios=None, h2_prices=None, dac_prices=None):
    """
    Process LCOX calculations for all technologies and countries, and save results to Excel.
    
    Args:
        input_data: Dictionary with all input data
        output_file: Path to output Excel file
        scenarios: List of scenarios to process (default: all)
        h2_prices: Dictionary with hydrogen prices (optional)
        dac_prices: Dictionary with DAC prices (optional)
    """
    try:
        # Use all scenarios if none specified
        if scenarios is None:
            scenarios = input_data['scenarios']
        
        results = {
            'lcox_values': {},
            'cost_components': {},
            'h2_prices': h2_prices,
            'dac_prices': dac_prices
        }
        
        # Process each scenario
        for scenario in scenarios:
            scenario = scenario.strip()  # Remove any extra spaces
            
            if scenario not in input_data['scenarios']:
                print(f"Warning: Scenario '{scenario}' not found in input data. Skipping.")
                continue
            
            print(f"\nProcessing scenario: {scenario}")
            
            # Initialize scenario results
            scenario_results = {
                'lcox_values': {},
                'cost_components': {}
            }
            
            # Process each technology
            for tech in input_data['tea_data'].index:
                tech = tech.strip()  # Remove any extra spaces
                
                print(f" Calculating LCOX for technology: {tech}...", end='')
                
                # Calculate costs
                country_results = {}
                
                for country in input_data['valid_countries']:
                    country = country.strip()  # Remove any extra spaces
                    
                    # Calculate costs
                    costs = calculate_cost_components(tech, country, scenario, input_data, h2_prices, dac_prices)
                    
                    if costs is not None:
                        country_results[country] = costs
                    else:
                        print(f"\n  Warning: Could not calculate costs for {tech} in {country} ({scenario}).")
                
                # Store results for this technology and scenario
                if country_results:
                    scenario_results['lcox_values'][tech] = {country: res['total'] for country, res in country_results.items()}
                    scenario_results['cost_components'][tech] = {country: res['components'] for country, res in country_results.items()}
                else:
                    print(f" Warning: No valid results for technology {tech} in scenario {scenario}.")
            
            # Store scenario results
            results['lcox_values'][scenario] = scenario_results['lcox_values']
            results['cost_components'][scenario] = scenario_results['cost_components']
        
        # Save all results to Excel
        save_results_to_excel(results, output_file)
        
    except Exception as e:
        print(f"Error processing LCOX calculations: {str(e)}")
        raise