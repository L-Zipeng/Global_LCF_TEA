"""
Global TEA Model - Main Execution Module

This module contains the main functions for running the Global Techno-Economic Analysis model.
It supports both deterministic and Monte Carlo uncertainty analysis.

"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from functions import *
from monte_carlo import sample_parameters, analyze_monte_carlo_results, run_monte_carlo_analysis

# Optimized helper functions for performance
def precalculate_common_values(input_data):
    """Pre-calculate values that don't change across iterations to avoid repeated pandas lookups"""
    print("Pre-calculating common values for optimization...")
    
    precomputed = {
        'tea_values': {},
        'energy_values': {},
        'wacc_cache': {},
        'price_cache': {}
    }
    
    # Pre-extract all TEA data values to avoid repeated pandas loc operations
    for tech in input_data['tea_data'].index:
        precomputed['tea_values'][tech] = {}
        for scenario in input_data['scenarios']:
            precomputed['tea_values'][tech][scenario] = {
                'capex': input_data['tea_data'].loc[tech, (scenario, 'capex')],
                'om': input_data['tea_data'].loc[tech, (scenario, 'om')],
                'lt': input_data['tea_data'].loc[tech, (scenario, 'lt')],
                'eff': input_data['tea_data'].loc[tech, (scenario, 'eff')]
            }
    
    # Pre-extract energy balance values
    for tech in input_data['energy_balance'].index:
        precomputed['energy_values'][tech] = {
            'e_elec': input_data['energy_balance'].loc[tech, 'e_elec'],
            'e_heat': input_data['energy_balance'].loc[tech, 'e_heat'],
            'm_pw': input_data['energy_balance'].loc[tech, 'm_pw'],
            'm_iw': input_data['energy_balance'].loc[tech, 'm_iw'],
            'm_bio': input_data['energy_balance'].loc[tech, 'm_bio'],
            'm_ng': input_data['energy_balance'].loc[tech, 'm_ng'],
            'm_h2': input_data['energy_balance'].loc[tech, 'm_h2'],
            'm_co2': input_data['energy_balance'].loc[tech, 'm_co2']
        }
    
    return precomputed

def cache_parameter_samples(input_data, use_monte_carlo, num_samples):
    """Cache parameter samples by technology-scenario to avoid redundant sampling"""
    if not use_monte_carlo:
        return {}
    
    print("Caching parameter samples for optimization...")
    parameter_cache = {}
    
    for tech in input_data['tea_data'].index:
        parameter_cache[tech] = {}
        for scenario in input_data['scenarios']:
            print(f"  Caching parameters for {tech}, {scenario}...")
            # Sample technology-specific parameters once per tech-scenario
            # Country-specific parameters (WACC_c, LCOE, natural gas prices) will be handled separately
            parameter_cache[tech][scenario] = sample_tech_specific_parameters(
                input_data, tech, scenario, num_samples
            )
    
    return parameter_cache

def sample_tech_specific_parameters(input_data, tech, scenario, num_samples):
    """
    Sample only technology-scenario specific parameters (not country-specific ones)
    This is an optimized version that avoids redundant sampling
    """
    tea_data = input_data['tea_data']
    
    # Extract the year from the scenario
    scenario_year = "2022"
    if "2030" in scenario:
        scenario_year = "2030" 
    elif "2050" in scenario:
        scenario_year = "2050"
    
    # Get baseline values using precomputed data if available
    if 'precomputed' in input_data and tech in input_data['precomputed']['tea_values']:
        tea_values = input_data['precomputed']['tea_values'][tech][scenario]
        baseline_capex = tea_values['capex']
        baseline_om = tea_values['om']
        baseline_lt = tea_values['lt']
        baseline_eff = tea_values['eff']
    else:
        # Fallback to pandas lookups if precomputed data not available
        baseline_capex = tea_data.loc[tech, (scenario, 'capex')]
        baseline_om = tea_data.loc[tech, (scenario, 'om')]
        baseline_lt = tea_data.loc[tech, (scenario, 'lt')]
        baseline_eff = tea_data.loc[tech, (scenario, 'eff')]
    
    # Generate distributions for technology-specific parameters
    from monte_carlo import (generate_capex_distribution, generate_om_distribution, 
                            generate_lifetime_distribution, generate_efficiency_distribution,
                            generate_capacity_factor_distribution, generate_wacc_distribution,
                            generate_material_price_distribution)
    
    samples = {
        'capex': generate_capex_distribution(baseline_capex, scenario_year, tech, num_samples),
        'om': generate_om_distribution(baseline_om, scenario_year, tech, num_samples),
        'lt': generate_lifetime_distribution(baseline_lt, scenario_year, tech, num_samples),
        'eff': generate_efficiency_distribution(baseline_eff, scenario_year, tech, num_samples),
        'cf': generate_capacity_factor_distribution(8000/8760, scenario_year, tech, num_samples)
    }
    
    # Technology-specific WACC factor (not country-specific)
    baseline_f_wacc_t = input_data['f_wacc_t'].loc[tech, scenario]
    samples['f_wacc_t'] = generate_wacc_distribution(baseline_f_wacc_t, False, scenario_year, tech, num_samples)
    
    # Material prices that are NOT country-specific
    if 'precomputed' in input_data and tech in input_data['precomputed']['energy_values']:
        energy_values = input_data['precomputed']['energy_values'][tech]
    else:
        energy_balance = input_data['energy_balance']
        energy_values = {
            'm_pw': energy_balance.loc[tech, 'm_pw'] if tech in energy_balance.index else 0,
            'm_iw': energy_balance.loc[tech, 'm_iw'] if tech in energy_balance.index else 0,
            'm_bio': energy_balance.loc[tech, 'm_bio'] if tech in energy_balance.index else 0,
            'm_ng': energy_balance.loc[tech, 'm_ng'] if tech in energy_balance.index else 0
        }
    
    price_constants = input_data['price_constants']
    
    # Sample material prices for materials used by this technology
    if energy_values['m_pw'] > 0:
        samples['p_pw'] = generate_material_price_distribution(price_constants['p_pw'], 'water_process', scenario_year, num_samples)
    
    if energy_values['m_iw'] > 0:
        samples['p_iw'] = generate_material_price_distribution(price_constants['p_iw'], 'water_input', scenario_year, num_samples)
    
    if energy_values['m_bio'] > 0:
        samples['p_bio'] = generate_material_price_distribution(price_constants['p_bio'], 'biomass', scenario_year, num_samples)
    
    # Note: p_ng (natural gas) and lcoe, f_wacc_c are country-specific and will be handled separately
    
    return samples

def get_country_specific_parameters(input_data, tech, country, scenario, num_samples, cached_params):
    """
    Generate country-specific parameters and combine with cached tech-specific parameters
    This optimizes by only generating what varies by country
    """
    # Start with cached technology-specific parameters
    combined_samples = cached_params.copy()
    
    # Extract scenario year
    scenario_year = "2022"
    if "2030" in scenario:
        scenario_year = "2030"
    elif "2050" in scenario:
        scenario_year = "2050"
    
    from monte_carlo import (generate_wacc_distribution, generate_electricity_price_distribution,
                            generate_material_price_distribution)
    from functions import get_electricity_price
    
    # Country-specific WACC factor
    baseline_f_wacc_c = input_data['f_wacc_c'].loc[input_data['f_wacc_c']['ISO_A3_EH'] == country, 'f_wacc_c'].iloc[0]
    combined_samples['f_wacc_c'] = generate_wacc_distribution(baseline_f_wacc_c, True, scenario_year, tech, num_samples)
    
    # Country-specific electricity price (LCOE)
    baseline_lcoe = get_electricity_price(tech, country, scenario, 
                                         input_data['lcoe'], 
                                         input_data['nuclear'], 
                                         input_data['csp'])
    if baseline_lcoe > 0:
        combined_samples['lcoe'] = generate_electricity_price_distribution(baseline_lcoe, scenario_year, tech, num_samples)
    else:
        combined_samples['lcoe'] = np.zeros(num_samples)
    
    # Country-specific natural gas price (if this technology uses natural gas)
    if 'precomputed' in input_data and tech in input_data['precomputed']['energy_values']:
        m_ng = input_data['precomputed']['energy_values'][tech]['m_ng']
    else:
        m_ng = input_data['energy_balance'].loc[tech, 'm_ng'] if tech in input_data['energy_balance'].index else 0
    
    if m_ng > 0:
        p_ng = input_data['price_constants']['p_ng'].get(country, input_data['price_constants']['p_ng']['default'])
        combined_samples['p_ng'] = generate_material_price_distribution(p_ng, 'natural_gas', scenario_year, num_samples)
    
    return combined_samples

# Modified process_lcox_calculations function
def process_lcox_calculations(input_data, use_monte_carlo=False, num_samples=1000):
    """
    Process LCOX calculations for all technologies, countries, and scenarios
    
    Args:
        input_data (dict): Dictionary containing all input data
        use_monte_carlo (bool): Whether to use Monte Carlo simulation
        num_samples (int): Number of samples to use in Monte Carlo simulation
    
    Returns:
        dict: Dictionary containing results
    """
    # Initialize timers for performance analysis
    timers = {
        'pem_h2': 0,
        'dac': 0,
        'other_techs': 0,
        'sampling': 0,
        'mc_calc': 0,
        'deterministic': 0,
        'optimization': 0
    }
    
    # OPTIMIZATION: Pre-calculate common values and cache parameter samples
    opt_start = time.time()
    precomputed = precalculate_common_values(input_data)
    parameter_cache = cache_parameter_samples(input_data, use_monte_carlo, num_samples)
    timers['optimization'] = time.time() - opt_start
    
    # Add precomputed values to input_data for easy access
    input_data['precomputed'] = precomputed
    
    # Data validation summary
    nuclear_countries = set(input_data['nuclear']['ISO_A3_EH']) if 'nuclear' in input_data else set()
    csp_countries = set(input_data['csp']['ISO_A3_EH']) if 'csp' in input_data else set()
    
    print(f"Loaded data for {len(input_data['valid_countries'])} countries")
    print(f"Nuclear-capable: {len(nuclear_countries)}, CSP-capable: {len(csp_countries)} countries")
    
    # First run the deterministic calculation
    print("\nRunning deterministic calculation...")
    deterministic_start = time.time()
    
    # Initialize results structure
    results = {
        'lcox_values': {},
        'cost_components': {},
        'monte_carlo_results': {} if use_monte_carlo else None,
        'monte_carlo_statistics': {} if use_monte_carlo else None
    }
    
    # Initialize results data structures
    for scenario in input_data['scenarios']:
        results['lcox_values'][scenario] = {}
        results['cost_components'][scenario] = {}
        if use_monte_carlo:
            results['monte_carlo_results'][scenario] = {}
            results['monte_carlo_statistics'][scenario] = {}
    
    # Step 1: Calculate PEM hydrogen prices first (used as input for some technologies)
    print("\nCalculating PEM hydrogen prices...")
    h2_prices = {}
    pem_start_time = time.time()
    
    for scenario in input_data['scenarios']:
        h2_prices[scenario] = {}
        for country in input_data['valid_countries']:
            h2_result = calculate_cost_components('PEM', country, scenario, input_data)
            if h2_result:
                # Initialize country in h2_prices if not already there
                if country not in h2_prices[scenario]:
                    h2_prices[scenario][country] = {}
                
                # Store as PEM technology specifically
                h2_prices[scenario][country]['PEM'] = h2_result['total']
                
                # Also store results for PEM in the main results
                if 'PEM' not in results['lcox_values'][scenario]:
                    results['lcox_values'][scenario]['PEM'] = {}
                    results['cost_components'][scenario]['PEM'] = {}
                
                results['lcox_values'][scenario]['PEM'][country] = h2_result['total']
                results['cost_components'][scenario]['PEM'][country] = h2_result['components']
                
                # If using Monte Carlo, calculate PEM hydrogen prices with uncertainty
                if use_monte_carlo:
                    if 'PEM' not in results['monte_carlo_results'][scenario]:
                        results['monte_carlo_results'][scenario]['PEM'] = {}
                        results['monte_carlo_statistics'][scenario]['PEM'] = {}
                    
                    # Use optimized parameters for PEM (OPTIMIZATION: Cached tech params + country-specific)
                    print(f"  Using optimized parameters for PEM in {country}, {scenario}...")
                    sampling_start = time.time()
                    parameter_samples = get_country_specific_parameters(
                        input_data, 'PEM', country, scenario, num_samples, parameter_cache['PEM'][scenario]
                    )
                    timers['sampling'] += time.time() - sampling_start
                    
                    # Calculate Monte Carlo results
                    mc_calc_start = time.time()
                    mc_results = calculate_cost_components_monte_carlo(
                        'PEM', country, scenario, input_data, 
                        parameter_samples=parameter_samples
                    )
                    timers['mc_calc'] += time.time() - mc_calc_start
                    
                    if mc_results is not None:
                        results['monte_carlo_results'][scenario]['PEM'][country] = mc_results
                        
                        # Store statistics
                        mc_stats = {
                            'mean': np.mean(mc_results),
                            'median': np.median(mc_results),
                            'std': np.std(mc_results),
                            'p10': np.percentile(mc_results, 10),
                            'p90': np.percentile(mc_results, 90),
                            'min': np.min(mc_results),
                            'max': np.max(mc_results)
                        }
                        results['monte_carlo_statistics'][scenario]['PEM'][country] = mc_stats
    
    timers['pem_h2'] = time.time() - pem_start_time
    
    # Step 2: Calculate DAC costs (used for CO2 pricing)
    print("\nCalculating DAC costs...")
    dac_prices = {}
    dac_start_time = time.time()
    
    for scenario in input_data['scenarios']:
        dac_prices[scenario] = {}
        for country in input_data['valid_countries']:
            dac_result = calculate_cost_components('DAC', country, scenario, input_data, h2_prices=h2_prices)
            if dac_result:
                dac_prices[scenario][country] = dac_result['total']
                
                # Also store results for DAC in the main results
                if 'DAC' not in results['lcox_values'][scenario]:
                    results['lcox_values'][scenario]['DAC'] = {}
                    results['cost_components'][scenario]['DAC'] = {}
                
                results['lcox_values'][scenario]['DAC'][country] = dac_result['total']
                results['cost_components'][scenario]['DAC'][country] = dac_result['components']
                
                # If using Monte Carlo, calculate DAC costs with uncertainty
                if use_monte_carlo:
                    if 'DAC' not in results['monte_carlo_results'][scenario]:
                        results['monte_carlo_results'][scenario]['DAC'] = {}
                        results['monte_carlo_statistics'][scenario]['DAC'] = {}
                    
                    # Use optimized parameters for DAC (OPTIMIZATION: Cached tech params + country-specific)
                    print(f"  Using optimized parameters for DAC in {country}, {scenario}...")
                    sampling_start = time.time()
                    parameter_samples = get_country_specific_parameters(
                        input_data, 'DAC', country, scenario, num_samples, parameter_cache['DAC'][scenario]
                    )
                    timers['sampling'] += time.time() - sampling_start
                    
                    # Calculate Monte Carlo results
                    mc_calc_start = time.time()
                    mc_results = calculate_cost_components_monte_carlo(
                        'DAC', country, scenario, input_data, 
                        h2_prices=h2_prices,
                        parameter_samples=parameter_samples
                    )
                    timers['mc_calc'] += time.time() - mc_calc_start
                    
                    if mc_results is not None:
                        results['monte_carlo_results'][scenario]['DAC'][country] = mc_results
                        
                        # Store statistics
                        mc_stats = {
                            'mean': np.mean(mc_results),
                            'median': np.median(mc_results),
                            'std': np.std(mc_results),
                            'p10': np.percentile(mc_results, 10),
                            'p90': np.percentile(mc_results, 90),
                            'min': np.min(mc_results),
                            'max': np.max(mc_results)
                        }
                        results['monte_carlo_statistics'][scenario]['DAC'][country] = mc_stats
    
    timers['dac'] = time.time() - dac_start_time
    
    # Step 3: Calculate AD costs first (needed for SR-FT biomethane)
    print("\nCalculating AD costs for biomethane...")
    ad_prices = {}
    
    for scenario in input_data['scenarios']:
        ad_prices[scenario] = {}
        for country in input_data['valid_countries']:
            ad_result = calculate_cost_components('AD', country, scenario, input_data, h2_prices=h2_prices, dac_prices=dac_prices)
            if ad_result:
                ad_prices[scenario][country] = ad_result['total']
                
                # Also store results for AD in the main results
                if 'AD' not in results['lcox_values'][scenario]:
                    results['lcox_values'][scenario]['AD'] = {}
                    results['cost_components'][scenario]['AD'] = {}
                
                results['lcox_values'][scenario]['AD'][country] = ad_result['total']
                results['cost_components'][scenario]['AD'][country] = ad_result['components']
                
                # If using Monte Carlo, calculate AD costs with uncertainty
                if use_monte_carlo:
                    if 'AD' not in results['monte_carlo_results'][scenario]:
                        results['monte_carlo_results'][scenario]['AD'] = {}
                        results['monte_carlo_statistics'][scenario]['AD'] = {}
                    
                    # Use optimized parameters for AD (OPTIMIZATION: Cached tech params + country-specific)
                    print(f"  Using optimized parameters for AD in {country}, {scenario}...")
                    sampling_start = time.time()
                    parameter_samples = get_country_specific_parameters(
                        input_data, 'AD', country, scenario, num_samples, parameter_cache['AD'][scenario]
                    )
                    timers['sampling'] += time.time() - sampling_start
                    
                    # Calculate Monte Carlo results
                    mc_calc_start = time.time()
                    mc_results = calculate_cost_components_monte_carlo(
                        'AD', country, scenario, input_data, 
                        h2_prices=h2_prices,
                        dac_prices=dac_prices,
                        parameter_samples=parameter_samples
                    )
                    timers['mc_calc'] += time.time() - mc_calc_start
                    
                    if mc_results is not None:
                        results['monte_carlo_results'][scenario]['AD'][country] = mc_results
                        
                        # Store statistics
                        mc_stats = {
                            'mean': np.mean(mc_results),
                            'median': np.median(mc_results),
                            'std': np.std(mc_results),
                            'p10': np.percentile(mc_results, 10),
                            'p90': np.percentile(mc_results, 90),
                            'min': np.min(mc_results),
                            'max': np.max(mc_results)
                        }
                        results['monte_carlo_statistics'][scenario]['AD'][country] = mc_stats
    
    # Step 4: Calculate LCOX for all other technologies
    print("\nProcessing LCOX calculations for remaining technologies...")
    other_techs_start_time = time.time()
    
    total_techs = len(input_data['tea_data'].index)
    
    # Define multi-product technologies and their products
    multi_product_techs = {
        'SR_FT': ['diesel', 'kerosene'],
        'ST_FT': ['diesel', 'kerosene'],
        'RWGS_FT': ['diesel', 'kerosene'],
        'TG_FT': ['diesel', 'kerosene'],
        'HVO': ['diesel', 'kerosene'],
        'B_PYR': ['kerosene'],  # Updated to produce kerosene only
        'RWGS_MeOH': ['methanol', 'DME', 'kerosene']
    }
    
    # Define technologies that produce kerosene (for fuel result list)
    kerosene_techs = ['HTL']  # HTL is a dedicated kerosene technology
    for tech, products in multi_product_techs.items():
        if 'kerosene' in products:
            # For multi-product technologies with kerosene output, 
            # create a specific identifier for the kerosene product
            kerosene_techs.append(f"{tech}_kerosene")
    
    for i, tech in enumerate(input_data['tea_data'].index, 1):
        # Skip PEM, DAC, and AD as they were already calculated
        if tech in ['PEM', 'DAC', 'AD']:
            continue
            
        print(f"Processing {tech} ({i}/{total_techs})...")
        
        # Check if this is a multi-product technology
        is_multi_product = tech in multi_product_techs
        
        # Initialize data structures for this technology
        for scenario in input_data['scenarios']:
            # Only create entries for the base technology if it's not a multi-product technology
            if not is_multi_product:
                if tech not in results['lcox_values'][scenario]:
                    results['lcox_values'][scenario][tech] = {}
                    results['cost_components'][scenario][tech] = {}
                    if use_monte_carlo:
                        results['monte_carlo_results'][scenario][tech] = {}
                        results['monte_carlo_statistics'][scenario][tech] = {}
            
            # For multi-product technologies, initialize product-specific results only
            if is_multi_product:
                for product in multi_product_techs[tech]:
                    tech_product = f"{tech}_{product}"
                    results['lcox_values'][scenario][tech_product] = {}
                    results['cost_components'][scenario][tech_product] = {}
                    if use_monte_carlo:
                        results['monte_carlo_results'][scenario][tech_product] = {}
                        results['monte_carlo_statistics'][scenario][tech_product] = {}
            
            # Determine eligible countries for this technology
            if tech in ['HTSE', 'CuCl']:
                countries = set(input_data['nuclear']['ISO_A3_EH']).intersection(input_data['valid_countries'])
            elif tech in ['SR_FT', 'ST_FT']:
                countries = set(input_data['csp']['ISO_A3_EH']).intersection(input_data['valid_countries'])
            else:
                countries = input_data['valid_countries']
            
            # Calculate costs for each country
            for country in countries:
                # For standard single-product technologies
                if not is_multi_product:
                    # Deterministic calculation
                    result = calculate_cost_components(
                        tech, country, scenario, input_data, 
                        h2_prices=h2_prices,
                        dac_prices=dac_prices,
                        ad_prices=ad_prices
                    )
                    if result:
                        results['lcox_values'][scenario][tech][country] = result['total']
                        results['cost_components'][scenario][tech][country] = result['components']
                    
                    # Monte Carlo calculation if enabled
                    if use_monte_carlo:
                        print(f"  Using optimized parameters for {tech} in {country}, {scenario}...")
                        sampling_start = time.time()
                        parameter_samples = get_country_specific_parameters(
                            input_data, tech, country, scenario, num_samples, parameter_cache[tech][scenario]
                        )
                        timers['sampling'] += time.time() - sampling_start
                        
                        mc_calc_start = time.time()
                        mc_results = calculate_cost_components_monte_carlo(
                            tech, country, scenario, input_data, 
                            h2_prices=h2_prices,
                            dac_prices=dac_prices,
                            ad_prices=ad_prices,
                            parameter_samples=parameter_samples
                        )
                        timers['mc_calc'] += time.time() - mc_calc_start
                        
                        if mc_results is not None:
                            results['monte_carlo_results'][scenario][tech][country] = mc_results
                            
                            # Store statistics
                            mc_stats = {
                                'mean': np.mean(mc_results),
                                'median': np.median(mc_results),
                                'std': np.std(mc_results),
                                'p10': np.percentile(mc_results, 10),
                                'p90': np.percentile(mc_results, 90),
                                'min': np.min(mc_results),
                                'max': np.max(mc_results)
                            }
                            results['monte_carlo_statistics'][scenario][tech][country] = mc_stats
                
                # For multi-product technologies, calculate costs for each product
                else:
                    for product in multi_product_techs[tech]:
                        # Deterministic calculation
                        result = calculate_cost_components(
                            tech, country, scenario, input_data, 
                            h2_prices=h2_prices,
                            dac_prices=dac_prices,
                            ad_prices=ad_prices,
                            product=product
                        )
                        if result:
                            tech_product = f"{tech}_{product}"
                            results['lcox_values'][scenario][tech_product][country] = result['total']
                            results['cost_components'][scenario][tech_product][country] = result['components']
                        
                        # Monte Carlo calculation if enabled
                        if use_monte_carlo:
                            tech_product = f"{tech}_{product}"
                            print(f"  Using optimized parameters for {tech_product} in {country}, {scenario}...")
                            sampling_start = time.time()
                            parameter_samples = get_country_specific_parameters(
                                input_data, tech, country, scenario, num_samples, parameter_cache[tech][scenario]
                            )
                            timers['sampling'] += time.time() - sampling_start
                            
                            mc_calc_start = time.time()
                            mc_results = calculate_cost_components_monte_carlo(
                                tech, country, scenario, input_data, 
                                h2_prices=h2_prices,
                                dac_prices=dac_prices,
                                ad_prices=ad_prices,
                                product=product,
                                parameter_samples=parameter_samples
                            )
                            timers['mc_calc'] += time.time() - mc_calc_start
                            
                            if mc_results is not None:
                                results['monte_carlo_results'][scenario][tech_product][country] = mc_results
                                
                                # Store statistics
                                mc_stats = {
                                    'mean': np.mean(mc_results),
                                    'median': np.median(mc_results),
                                    'std': np.std(mc_results),
                                    'p10': np.percentile(mc_results, 10),
                                    'p90': np.percentile(mc_results, 90),
                                    'min': np.min(mc_results),
                                    'max': np.max(mc_results)
                                }
                                results['monte_carlo_statistics'][scenario][tech_product][country] = mc_stats
    
    # Add a special list of kerosene-producing technologies for easier analysis
    results['kerosene_techs'] = kerosene_techs
    
    timers['other_techs'] = time.time() - other_techs_start_time
    
    # Print timing summary
    # Performance summary
    total_time = sum(timers.values())
    print(f"\nAnalysis completed in {total_time:.1f} seconds")
    if use_monte_carlo:
        print(f"Monte Carlo analysis with optimized parameter caching enabled")
    
    # Add h2_prices, dac_prices, and ad_prices to results
    results['h2_prices'] = h2_prices
    results['dac_prices'] = dac_prices
    results['ad_prices'] = ad_prices

    return results

def generate_monte_carlo_plots(results, output_dir):
    """
    Generate plots from Monte Carlo simulation results
    
    Args:
        results: Dictionary with Monte Carlo results
        output_dir: Directory to save plots
    """
    # Create output directory for plots
    plots_dir = output_dir / 'monte_carlo_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    if not results['monte_carlo_results']:
        print("No Monte Carlo results to plot")
        return
    
    # Select a sample of key countries for plots (to avoid too many plots)
    key_countries = ['USA', 'DEU', 'CHN', 'JPN', 'BRA', 'ZAF', 'AUS']
    
    # For each scenario
    for scenario in results['monte_carlo_results'].keys():
        scenario_dir = plots_dir / scenario
        os.makedirs(scenario_dir, exist_ok=True)
        
        # For each technology
        for tech in results['monte_carlo_results'][scenario].keys():
            tech_dir = scenario_dir / tech
            os.makedirs(tech_dir, exist_ok=True)
            
            # Plot distributions for available countries (focus on key countries if available)
            countries = list(results['monte_carlo_results'][scenario][tech].keys())
            plot_countries = [c for c in key_countries if c in countries]
            if not plot_countries and countries:
                # If none of the key countries are available, use first 5 countries
                plot_countries = countries[:min(5, len(countries))]
            
            # Skip if no countries to plot
            if not plot_countries:
                continue
                
            # Distribution plots for key countries
            plt.figure(figsize=(10, 6))
            
            for country in plot_countries:
                if country in results['monte_carlo_results'][scenario][tech]:
                    data = results['monte_carlo_results'][scenario][tech][country]
                    
                    # Plot kernel density estimate
                    sns.kdeplot(data, label=country)
            
            plt.title(f'LCOX Distribution - {tech} ({scenario})')
            plt.xlabel('LCOX (EUR/kWh)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.savefig(tech_dir / f'distribution_key_countries.png', dpi=300)
            plt.close()
            
            # Box plots comparing countries
            plt.figure(figsize=(12, 6))
            
            # Prepare data for box plot
            box_data = []
            box_labels = []
            
            for country in plot_countries:
                if country in results['monte_carlo_results'][scenario][tech]:
                    box_data.append(results['monte_carlo_results'][scenario][tech][country])
                    box_labels.append(country)
            
            # Create box plot
            if box_data:
                plt.boxplot(box_data, labels=box_labels, showfliers=False)
                plt.title(f'LCOX Comparison - {tech} ({scenario})')
                plt.ylabel('LCOX (EUR/kWh)')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                
                # Save the plot
                plt.savefig(tech_dir / f'boxplot_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    # Generate tornado plots for selected technologies, countries, and scenarios
    # These show sensitivity of results to different input factors
    # Implementation would be complex and likely require a separate function

def save_monte_carlo_results(results, output_dir):
    """
    Save Monte Carlo results and statistics to Excel files
    
    Args:
        results: Dictionary with results
        output_dir: Output directory
    """
    if not results['monte_carlo_statistics']:
        print("No Monte Carlo results to save")
        return
    
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
    
    # Create a mapping for technology names to avoid truncation
    tech_name_mapping = {
        'RWGS_MeOH_kerosene': 'RWGS_MeOH_kerosene',
        'RWGS_MeOH_methanol': 'RWGS_MeOH_methanol', 
        'RWGS_FT_kerosene': 'RWGS_FT_kerosene',
        'RWGS_FT_kerosen': 'RWGS_FT_kerosene',  # Fix common misspelling
        'RWGS_MeOH_keros': 'RWGS_MeOH_kerosene',  # Fix truncated name
        'RWGS_MeOH_metha': 'RWGS_MeOH_methanol',  # Fix truncated name
    }
    
    # Save Monte Carlo statistics by scenario
    for scenario in results['monte_carlo_statistics'].keys():
        scenario_code = scenario_mapping.get(scenario, scenario[:10])
        
        # Create a dataframe to store statistics
        stats_dfs = {}
        
        for tech in results['monte_carlo_statistics'][scenario].keys():
            # Use proper technology name instead of truncating
            clean_tech_name = tech_name_mapping.get(tech, tech)
            
            # Create a safe sheet name (Excel limit is 31 characters)
            if len(f'Stats_{clean_tech_name}') <= 31:
                sheet_name = f'Stats_{clean_tech_name}'
            else:
                # Only truncate if absolutely necessary for Excel limit
                max_tech_len = 31 - 6  # 31 - len('Stats_')
                sheet_name = f'Stats_{clean_tech_name[:max_tech_len]}'
            
            # Skip if no data for this technology
            if not results['monte_carlo_statistics'][scenario][tech]:
                continue
            
            # Convert to DataFrame
            stats_df = pd.DataFrame.from_dict(
                {country: stats for country, stats in results['monte_carlo_statistics'][scenario][tech].items()},
                orient='index'
            )
            
            # Store the DataFrame with the clean tech name as key for consistency
            stats_dfs[clean_tech_name] = stats_df
        
        if stats_dfs:
            # Save to Excel
            with pd.ExcelWriter(output_dir / f'monte_carlo_stats_{scenario_code}.xlsx') as writer:
                for clean_tech_name, df in stats_dfs.items():
                    # Create safe sheet name
                    if len(f'Stats_{clean_tech_name}') <= 31:
                        sheet_name = f'Stats_{clean_tech_name}'
                    else:
                        max_tech_len = 31 - 6
                        sheet_name = f'Stats_{clean_tech_name[:max_tech_len]}'
                    df.to_excel(writer, sheet_name=sheet_name)
    
    # Also save aggregated statistics across all technologies and scenarios
    tech_comparison = {}
    
    for scenario in results['monte_carlo_statistics'].keys():
        scenario_code = scenario_mapping.get(scenario, scenario[:10])
        
        for tech in results['monte_carlo_statistics'][scenario].keys():
            # Skip if no data for this technology
            if not results['monte_carlo_statistics'][scenario][tech]:
                continue
            
            # For each country, get median and 90% confidence interval
            for country in results['monte_carlo_statistics'][scenario][tech].keys():
                stats = results['monte_carlo_statistics'][scenario][tech][country]
                
                key = (scenario_code, tech, country)
                tech_comparison[key] = {
                    'median': stats['median'],
                    'p10': stats['p10'],
                    'p90': stats['p90'],
                    'std': stats['std']
                }
    
    # Convert to DataFrame with multi-index
    tech_comp_df = pd.DataFrame.from_dict(tech_comparison, orient='index')
    tech_comp_df.index = pd.MultiIndex.from_tuples(tech_comp_df.index, names=['Scenario', 'Technology', 'Country'])
    
    # Save to Excel
    tech_comp_df.to_excel(output_dir / 'monte_carlo_technology_comparison.xlsx')

def main(use_monte_carlo=True, num_samples=1000):
    """Main function to run the Global TEA model"""
    # Record start time
    start_time = time.time()
    
    # Define output directory (relative to project root)
    output_dir = Path(__file__).parent.parent.parent / 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data (relative to project root)
    print("Loading input data...")
    data_file = Path(__file__).parent.parent.parent / 'data' / 'TEA input.xlsx'
    input_data = load_input_data(str(data_file))
    
    if not input_data:
        print("Failed to load input data.")
        return
    
    # Process LCOX calculations
    print("\nProcessing LCOX calculations...")
    results = process_lcox_calculations(input_data, use_monte_carlo, num_samples)
    
    # Calculate and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nMonte Carlo analysis completed in {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time/num_samples:.4f} seconds")
    
    print("Saving results to Excel...")
    # Save results to Excel
    save_results_to_excel(results, output_dir / 'lcox_results.xlsx')
    
    # If Monte Carlo was used, save Monte Carlo results and generate plots
    if use_monte_carlo and results['monte_carlo_results']:
        print("Saving Monte Carlo results...")
        save_monte_carlo_results(results, output_dir)
        
        # Skip plotting for speed
        # print("Generating Monte Carlo plots...")
        # generate_monte_carlo_plots(results, output_dir)
    
    print("Analysis completed successfully!")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Global TEA model')
    parser.add_argument('--no-monte-carlo', action='store_true', help='Skip Monte Carlo simulation')
    parser.add_argument('--samples', type=int, default=1000, help='Number of Monte Carlo samples')
    args = parser.parse_args()
    
    main(use_monte_carlo=not args.no_monte_carlo, num_samples=args.samples)