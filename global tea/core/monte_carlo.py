"""
Global TEA Model - Monte Carlo Analysis Module

This module provides Monte Carlo uncertainty analysis capabilities for the Global TEA model.
It includes functions for parameter sampling, uncertainty quantification, and statistical analysis.

"""

import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def triangular_sample(min_val, mode_val, max_val, size=1):
    """Generate random samples from a triangular distribution"""
    return np.random.triangular(min_val, mode_val, max_val, size)

def pert_sample(min_val, mode_val, max_val, size=1, gamma=4):
    """
    Generate random samples from a PERT distribution
    PERT is like triangular but with smoother tails
    """
    alpha = 1 + gamma * (mode_val - min_val) / (max_val - min_val)
    beta = 1 + gamma * (max_val - mode_val) / (max_val - min_val)
    # Scale the beta distribution to the range [min_val, max_val]
    return min_val + (max_val - min_val) * np.random.beta(alpha, beta, size)

def truncated_normal_sample(mean, std, lower_bound, upper_bound, size=1):
    """Generate random samples from a truncated normal distribution"""
    return stats.truncnorm(
        (lower_bound - mean) / std, 
        (upper_bound - mean) / std, 
        loc=mean, 
        scale=std
    ).rvs(size)

def generate_capex_distribution(baseline_capex, scenario_year="2022", tech=None, size=1):
    """
    Generate CAPEX samples using a PERT distribution
    For future scenarios, increase uncertainty
    """
    # Base uncertainty parameters
    min_factor = 0.8
    max_factor = 1.3
    
    # Increase uncertainty for future years
    if scenario_year == "2030":
        min_factor = 0.75
        max_factor = 1.4
    elif scenario_year == "2050":
        min_factor = 0.7
        max_factor = 1.5
    
    # Special handling for technologies that need wider uncertainty ranges
    if tech in ["PTM", "HB", "RWGS_MeOH_metha", "RWGS_MeOH_DME", "FAME"]:
        # Increase uncertainty for these technologies
        min_factor -= 0.05
        max_factor += 0.1
    
    min_val = min_factor * baseline_capex
    mode_val = baseline_capex
    max_val = max_factor * baseline_capex
    
    return pert_sample(min_val, mode_val, max_val, size)

def generate_om_distribution(baseline_om, scenario_year="2022", tech=None, size=1):
    """
    Generate O&M cost samples using a lognormal distribution
    For future scenarios, increase uncertainty
    """
    # Base uncertainty
    sigma = 0.15  # 15% standard deviation
    
    # Increase uncertainty for future years
    if scenario_year == "2030":
        sigma = 0.18  # 18% standard deviation
    elif scenario_year == "2050":
        sigma = 0.22  # 22% standard deviation
    
    # Special handling for technologies that need wider uncertainty ranges
    if tech in ["PTM", "HB", "RWGS_MeOH_metha", "RWGS_MeOH_DME", "FAME"]:
        sigma += 0.05  # Additional 5% standard deviation
    
    mu = np.log(baseline_om) - 0.5 * sigma**2
    return np.random.lognormal(mu, sigma, size)

def generate_efficiency_distribution(baseline_eff, scenario_year="2022", tech=None, size=1):
    """
    Generate efficiency samples using a truncated normal distribution
    For future scenarios, increase uncertainty
    """
    mean = baseline_eff
    
    # Base uncertainty
    std_factor = 0.06  # 6% standard deviation
    
    # Increase uncertainty for future years
    if scenario_year == "2030":
        std_factor = 0.08  # 8% standard deviation
    elif scenario_year == "2050":
        std_factor = 0.1  # 10% standard deviation
    
    # Special handling for technologies that need wider uncertainty ranges
    if tech in ["PTM", "HB", "RWGS_MeOH_metha", "RWGS_MeOH_DME", "FAME"]:
        std_factor += 0.03  # Additional 3% standard deviation
    
    std = std_factor * baseline_eff
    lower_bound = max(0.5 * baseline_eff, 0.1)  # Cannot be less than 50% of baseline or 10%
    upper_bound = min(1.1 * baseline_eff, 1.0)  # Cannot exceed 110% of baseline or 100%
    
    return truncated_normal_sample(mean, std, lower_bound, upper_bound, size)

def generate_lifetime_distribution(baseline_lt, scenario_year="2022", tech=None, size=1):
    """
    Generate lifetime samples using a triangular distribution
    For future scenarios, increase uncertainty
    """
    # Base uncertainty parameters
    min_factor = 0.8
    max_factor = 1.1
    
    # Increase uncertainty for future years
    if scenario_year == "2030":
        min_factor = 0.75
        max_factor = 1.15
    elif scenario_year == "2050":
        min_factor = 0.7
        max_factor = 1.2
    
    # Special handling for technologies that need wider uncertainty ranges
    if tech in ["PTM", "HB", "RWGS_MeOH_metha", "RWGS_MeOH_DME", "FAME"]:
        min_factor -= 0.05
        max_factor += 0.05
    
    min_val = min_factor * baseline_lt
    mode_val = baseline_lt
    max_val = max_factor * baseline_lt
    
    return triangular_sample(min_val, mode_val, max_val, size)

def generate_wacc_distribution(baseline_wacc, is_country_factor=False, scenario_year="2022", tech=None, size=1):
    """
    Generate WACC factor samples using a triangular distribution
    For future scenarios, increase uncertainty
    """
    # Base uncertainty
    if is_country_factor:
        std_dev = 0.18  # 18% standard deviation for country factors
    else:
        std_dev = 0.12  # 12% standard deviation for technology factors
    
    # Increase uncertainty for future years
    if scenario_year == "2030":
        std_dev *= 1.2  # 20% increase in standard deviation
    elif scenario_year == "2050":
        std_dev *= 1.5  # 50% increase in standard deviation
    
    # Special handling for technologies that need wider uncertainty ranges
    if tech in ["PTM", "HB", "RWGS_MeOH_metha", "RWGS_MeOH_DME", "FAME"] and not is_country_factor:
        std_dev += 0.03  # Additional 3% standard deviation
    
    min_val = max(baseline_wacc * (1 - 2*std_dev), 0.01)  # Prevent negative or zero values
    mode_val = baseline_wacc
    max_val = baseline_wacc * (1 + 2*std_dev)
    
    return triangular_sample(min_val, mode_val, max_val, size)

def generate_capacity_factor_distribution(baseline_cf, scenario_year="2022", tech=None, size=1):
    """
    Generate capacity factor samples using a beta distribution
    For future scenarios, increase uncertainty
    """
    # Scale beta parameters to get mean close to baseline
    alpha = 9
    beta = 3
    
    # Adjust shape parameters to increase uncertainty for future years
    if scenario_year == "2030":
        alpha = 8
        beta = 2.8
    elif scenario_year == "2050":
        alpha = 7
        beta = 2.5
    
    # Special handling for technologies that need wider uncertainty ranges
    if tech in ["PTM", "HB", "RWGS_MeOH_metha", "RWGS_MeOH_DME", "FAME"]:
        alpha -= 1  # Increase variance
        beta -= 0.5
    
    # Scale to reasonable range for capacity factor
    min_cf = 0.5
    max_cf = 0.95
    # Generate samples from beta distribution and scale to [min_cf, max_cf]
    samples = np.random.beta(alpha, beta, size)
    return min_cf + (max_cf - min_cf) * samples

def generate_electricity_price_distribution(baseline_lcoe, scenario_year="2022", tech=None, size=1):
    """
    Generate electricity price samples using a lognormal distribution
    For future scenarios, increase uncertainty
    """
    # Base uncertainty
    sigma = 0.22  # 22% standard deviation
    
    # Increase uncertainty for future years
    if scenario_year == "2030":
        sigma = 0.28  # 28% standard deviation
    elif scenario_year == "2050":
        sigma = 0.35  # 35% standard deviation
    
    mu = np.log(baseline_lcoe) - 0.5 * sigma**2
    return np.random.lognormal(mu, sigma, size)

def generate_material_price_distribution(baseline_price, material_type, scenario_year="2022", size=1):
    """
    Generate material price samples based on material type
    For future scenarios, increase uncertainty
    """
    # Factor to increase uncertainty for future years
    uncertainty_factor = 1.0
    if scenario_year == "2030":
        uncertainty_factor = 1.2
    elif scenario_year == "2050":
        uncertainty_factor = 1.5
    
    if material_type == 'natural_gas':
        # Lognormal for natural gas - more volatile
        sigma = 0.30 * uncertainty_factor
        mu = np.log(baseline_price) - 0.5 * sigma**2
        return np.random.lognormal(mu, sigma, size)
    elif material_type == 'biomass':
        # Normal for biomass
        std = 0.20 * uncertainty_factor * baseline_price
        return np.random.normal(baseline_price, std, size)
    elif material_type.startswith('water'):
        # Normal with low volatility for water
        std = 0.10 * uncertainty_factor * baseline_price
        return np.random.normal(baseline_price, std, size)
    else:
        # Default to triangular for other materials
        min_val = (0.85 - 0.1 * (uncertainty_factor - 1)) * baseline_price
        mode_val = baseline_price
        max_val = (1.25 + 0.15 * (uncertainty_factor - 1)) * baseline_price
        return triangular_sample(min_val, mode_val, max_val, size)

def generate_dac_price_distribution(baseline_dac, scenario_year="2022", size=1):
    """
    Generate DAC price samples using a triangular distribution
    For future scenarios, increase uncertainty
    """
    # Base uncertainty parameters
    min_factor = 0.7
    max_factor = 1.5
    
    # Increase uncertainty for future years
    if scenario_year == "2030":
        min_factor = 0.6
        max_factor = 1.7
    elif scenario_year == "2050":
        min_factor = 0.5
        max_factor = 2.0
    
    min_val = min_factor * baseline_dac
    mode_val = baseline_dac
    max_val = max_factor * baseline_dac
    return triangular_sample(min_val, mode_val, max_val, size)

def generate_hydrogen_price_distribution(baseline_h2, scenario_year="2022", size=1):
    """
    Generate hydrogen price samples using a lognormal distribution
    For future scenarios, increase uncertainty
    """
    # Base uncertainty
    sigma = 0.25  # 25% standard deviation
    
    # Increase uncertainty for future years
    if scenario_year == "2030":
        sigma = 0.3  # 30% standard deviation
    elif scenario_year == "2050":
        sigma = 0.4  # 40% standard deviation
    
    mu = np.log(baseline_h2) - 0.5 * sigma**2
    return np.random.lognormal(mu, sigma, size)

def sample_parameters(input_data, tech, country, scenario, num_samples=1000):
    """
    Generate all parameter samples for a specific technology, country, and scenario
    
    Args:
        input_data: Dictionary of input data
        tech: Technology identifier
        country: Country code
        scenario: Scenario name
        num_samples: Number of Monte Carlo samples to generate
        
    Returns:
        Dictionary of parameter samples
    """
    tea_data = input_data['tea_data']
    
    # Extract the year from the scenario (2022, 2030, 2050)
    scenario_year = "2022"
    if "2030" in scenario:
        scenario_year = "2030"
    elif "2050" in scenario:
        scenario_year = "2050"
    
    # Check if this technology is FAME (for special handling)
    is_fame = (tech == "FAME")
    
    # Get baseline values
    baseline_capex = tea_data.loc[tech, (scenario, 'capex')]
    baseline_om = tea_data.loc[tech, (scenario, 'om')]
    baseline_lt = tea_data.loc[tech, (scenario, 'lt')]
    baseline_eff = tea_data.loc[tech, (scenario, 'eff')]
    
    # Generate distributions for each parameter
    samples = {
        'capex': generate_capex_distribution(baseline_capex, scenario_year, tech, num_samples),
        'om': generate_om_distribution(baseline_om, scenario_year, tech, num_samples),
        'lt': generate_lifetime_distribution(baseline_lt, scenario_year, tech, num_samples),
        'eff': generate_efficiency_distribution(baseline_eff, scenario_year, tech, num_samples),
        'cf': generate_capacity_factor_distribution(8000/8760, scenario_year, tech, num_samples)
    }
    
    # WACC factors
    baseline_f_wacc_t = input_data['f_wacc_t'].loc[tech, scenario]
    samples['f_wacc_t'] = generate_wacc_distribution(baseline_f_wacc_t, False, scenario_year, tech, num_samples)
    
    baseline_f_wacc_c = input_data['f_wacc_c'].loc[input_data['f_wacc_c']['ISO_A3_EH'] == country, 'f_wacc_c'].iloc[0]
    samples['f_wacc_c'] = generate_wacc_distribution(baseline_f_wacc_c, True, scenario_year, tech, num_samples)
    
    # Electricity price (LCOE)
    from functions import get_electricity_price
    baseline_lcoe = get_electricity_price(tech, country, scenario, 
                                     input_data['lcoe'], 
                                     input_data['nuclear'], 
                                     input_data['csp'])
    if baseline_lcoe > 0:
        samples['lcoe'] = generate_electricity_price_distribution(baseline_lcoe, scenario_year, tech, num_samples)
    else:
        samples['lcoe'] = np.zeros(num_samples)
    
    # Material prices - only include if used by this technology
    energy_balance = input_data['energy_balance']
    price_constants = input_data['price_constants']
    
    m_pw = energy_balance.loc[tech, 'm_pw'] if tech in energy_balance.index and 'm_pw' in energy_balance.columns else 0
    if m_pw > 0:
        samples['p_pw'] = generate_material_price_distribution(price_constants['p_pw'], 'water_process', scenario_year, num_samples)
    
    m_iw = energy_balance.loc[tech, 'm_iw'] if tech in energy_balance.index and 'm_iw' in energy_balance.columns else 0
    if m_iw > 0:
        samples['p_iw'] = generate_material_price_distribution(price_constants['p_iw'], 'water_input', scenario_year, num_samples)
    
    m_bio = energy_balance.loc[tech, 'm_bio'] if tech in energy_balance.index and 'm_bio' in energy_balance.columns else 0
    if m_bio > 0:
        samples['p_bio'] = generate_material_price_distribution(price_constants['p_bio'], 'biomass', scenario_year, num_samples)
    
    m_ng = energy_balance.loc[tech, 'm_ng'] if tech in energy_balance.index and 'm_ng' in energy_balance.columns else 0
    if m_ng > 0:
        p_ng = price_constants['p_ng'].get(country, price_constants['p_ng']['default'])
        samples['p_ng'] = generate_material_price_distribution(p_ng, 'natural_gas', scenario_year, num_samples)
    
    # Add extra logging for FAME
    if is_fame:
        print(f"Sampling for FAME in {country}, {scenario} (Year: {scenario_year})")
        print(f"  CAPEX range: {np.min(samples['capex']):.2f} - {np.max(samples['capex']):.2f}")
        print(f"  Efficiency range: {np.min(samples['eff']):.2f} - {np.max(samples['eff']):.2f}")
    
    return samples

def analyze_monte_carlo_results(results):
    """
    Analyze Monte Carlo simulation results
    
    Args:
        results: Dictionary of simulation results
        
    Returns:
        DataFrame with statistical analysis
    """
    analysis = {}
    
    # For each scenario
    for scenario in results.keys():
        analysis[scenario] = {}
        
        # For each technology
        for tech in results[scenario].keys():
            # Skip if no results for this technology
            if not results[scenario][tech]:
                continue
                
            # Create a dataframe with results for all countries
            country_results = pd.DataFrame(results[scenario][tech])
            
            # Calculate statistics
            analysis[scenario][tech] = {
                'mean': country_results.mean(),
                'median': country_results.median(),
                'std': country_results.std(),
                'min': country_results.min(),
                'max': country_results.max(),
                'p10': country_results.quantile(0.1),
                'p90': country_results.quantile(0.9)
            }
    
    return analysis

def run_monte_carlo_analysis(
    input_data=None,
    input_file='data/TEA input.xlsx',
    output_dir='output',
    num_samples=1000,
    scenarios=None,
    save_results=True
):
    """
    Run Monte Carlo analysis with advanced methodology:
    1. Sequential supply chain modeling (H2 → DAC → Fuels)
    2. Proper multi-product technology handling
    3. Geographic constraint enforcement
    4. Comprehensive uncertainty modeling
    
    Args:
        input_data: Pre-loaded input data (optional)
        input_file: Path to input file if input_data not provided
        output_dir: Directory to save results
        num_samples: Number of Monte Carlo samples
        scenarios: List of scenarios (default: all scenarios)
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary with monte_carlo_results and monte_carlo_statistics
    """
    print("Running Monte Carlo analysis with advanced methodology...")
    
    # Load input data if not provided
    if input_data is None:
        from functions import load_input_data
        print(f"Loading input data from {input_file}...")
        input_data = load_input_data(input_file)
        
        if not input_data:
            print("Failed to load input data.")
            return None
    
    # Define scenarios if not provided
    if scenarios is None:
        scenarios = ['Base_2024', 'Base_2030', 'Base_2050', 
                    '2 degree_2030', '2 degree_2050', 
                    '1.5 degree_2030', '1.5 degree_2050']
    
    # Create output directory
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results structure
    results = {
        'monte_carlo_results': {},
        'monte_carlo_statistics': {}
    }
    
    # Initialize results data structures
    for scenario in scenarios:
        results['monte_carlo_results'][scenario] = {}
        results['monte_carlo_statistics'][scenario] = {}

    # Step 1: Calculate PEM hydrogen prices first (used as input for some technologies)
    print("\nCalculating PEM hydrogen prices...")
    h2_prices = {}
    
    for scenario in scenarios:
        h2_prices[scenario] = {}
        for country in input_data['valid_countries']:
            # Sample parameters for PEM
            print(f"  Sampling parameters for PEM in {country}, {scenario}...")
            parameter_samples = sample_parameters(input_data, 'PEM', country, scenario, num_samples)
            
            # Calculate Monte Carlo results
            from functions import calculate_cost_components_monte_carlo
            mc_results = calculate_cost_components_monte_carlo(
                'PEM', country, scenario, input_data, 
                parameter_samples=parameter_samples
            )
            
            if mc_results is not None:
                # Initialize PEM in results if not already there
                if 'PEM' not in results['monte_carlo_results'][scenario]:
                    results['monte_carlo_results'][scenario]['PEM'] = {}
                    results['monte_carlo_statistics'][scenario]['PEM'] = {}
                
                # Store results
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
                
                # Store the median value in h2_prices for use in other calculations
                if country not in h2_prices[scenario]:
                    h2_prices[scenario][country] = {}
                h2_prices[scenario][country]['PEM'] = np.median(mc_results)
    
    # Step 2: Calculate DAC costs (used for CO2 pricing)
    print("\nCalculating DAC costs...")
    dac_prices = {}
    
    for scenario in scenarios:
        dac_prices[scenario] = {}
        for country in input_data['valid_countries']:
            # Sample parameters for DAC
            print(f"  Sampling parameters for DAC in {country}, {scenario}...")
            parameter_samples = sample_parameters(input_data, 'DAC', country, scenario, num_samples)
            
            # Calculate Monte Carlo results
            from functions import calculate_cost_components_monte_carlo
            mc_results = calculate_cost_components_monte_carlo(
                'DAC', country, scenario, input_data, 
                h2_prices=h2_prices,
                parameter_samples=parameter_samples
            )
            
            if mc_results is not None:
                # Initialize DAC in results if not already there
                if 'DAC' not in results['monte_carlo_results'][scenario]:
                    results['monte_carlo_results'][scenario]['DAC'] = {}
                    results['monte_carlo_statistics'][scenario]['DAC'] = {}
                
                # Store results
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
                
                # Store the median value in dac_prices for use in other calculations
                dac_prices[scenario][country] = np.median(mc_results)
    
    # Step 3: Calculate LCOX for all other technologies
    print("\nProcessing LCOX calculations for remaining technologies...")
    
    # Define multi-product technologies and their products
    multi_product_techs = {
        'SR_FT': ['diesel', 'kerosene'],
        'ST_FT': ['diesel', 'kerosene'],
        'RWGS_FT': ['diesel', 'kerosene'],
        'TG_FT': ['diesel', 'kerosene'],
        'HVO': ['diesel', 'kerosene'],
        'B_PYR': ['kerosene'],
        'RWGS_MeOH': ['methanol', 'DME', 'kerosene']
    }
    
    # Check if FAME exists in the dataset
    has_fame = 'FAME' in input_data['tea_data'].index
    if has_fame:
        print("FAME technology found in dataset.")
    else:
        print("WARNING: FAME technology not found in dataset!")
    
    # Process all technologies
    total_techs = len(input_data['tea_data'].index)
    count = 0
    
    for tech in input_data['tea_data'].index:
        count += 1
        print(f"\nProcessing technology {count}/{total_techs}: {tech}")
        
        # Skip PEM and DAC (already processed)
        if tech in ['PEM', 'DAC']:
            continue
        
        # Check if this is a multi-product technology
        is_multi_product = tech in multi_product_techs
        
        # Process regular technologies
        if not is_multi_product:
            for scenario in scenarios:
                if tech not in results['monte_carlo_results'][scenario]:
                    results['monte_carlo_results'][scenario][tech] = {}
                    results['monte_carlo_statistics'][scenario][tech] = {}
                
                # Determine eligible countries for this technology
                if tech in ['HTSE', 'CuCl']:
                    eligible_countries = set(input_data['nuclear']['ISO_A3_EH']).intersection(input_data['valid_countries'])
                elif tech in ['SR_FT', 'ST_FT']:
                    eligible_countries = set(input_data['csp']['ISO_A3_EH']).intersection(input_data['valid_countries'])
                else:
                    eligible_countries = input_data['valid_countries']
                
                # Only process eligible countries
                for country in eligible_countries:
                    # Sample parameters
                    print(f"  Sampling parameters for {tech} in {country}, {scenario}...")
                    parameter_samples = sample_parameters(input_data, tech, country, scenario, num_samples)
                    
                    # Calculate Monte Carlo results
                    from functions import calculate_cost_components_monte_carlo
                    mc_results = calculate_cost_components_monte_carlo(
                        tech, country, scenario, input_data, 
                        h2_prices=h2_prices,
                        dac_prices=dac_prices,
                        parameter_samples=parameter_samples
                    )
                    
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
        
        # Process multi-product technologies
        else:
            products = multi_product_techs[tech]
            for product in products:
                tech_product = f"{tech}_{product}"
                
                for scenario in scenarios:
                    if tech_product not in results['monte_carlo_results'][scenario]:
                        results['monte_carlo_results'][scenario][tech_product] = {}
                        results['monte_carlo_statistics'][scenario][tech_product] = {}
                    
                    # Determine eligible countries for this technology
                    if tech in ['HTSE', 'CuCl']:
                        eligible_countries = set(input_data['nuclear']['ISO_A3_EH']).intersection(input_data['valid_countries'])
                    elif tech in ['SR_FT', 'ST_FT']:
                        eligible_countries = set(input_data['csp']['ISO_A3_EH']).intersection(input_data['valid_countries'])
                    else:
                        eligible_countries = input_data['valid_countries']
                    
                    # Only process eligible countries
                    for country in eligible_countries:
                        # Sample parameters
                        print(f"  Sampling parameters for {tech_product} in {country}, {scenario}...")
                        parameter_samples = sample_parameters(input_data, tech, country, scenario, num_samples)
                        
                        # Calculate Monte Carlo results
                        from functions import calculate_cost_components_monte_carlo
                        mc_results = calculate_cost_components_monte_carlo(
                            tech, country, scenario, input_data, 
                            h2_prices=h2_prices,
                            dac_prices=dac_prices,
                            product=product,
                            parameter_samples=parameter_samples
                        )
                        
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
    
    # Save results if requested
    if save_results:
        # Add this just before the pickle dump:
        results['h2_prices'] = h2_prices 
        results['dac_prices'] = dac_prices

        output_file = output_dir / 'monte_carlo_results.pkl'
        print(f"\nSaving Monte Carlo results to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Also save statistics to Excel files
        for scenario in scenarios:
            # Create mapping for Excel file names
            scenario_mapping = {
                'Base_2024': 'Base24',
                'Base_2030': 'Base30',
                'Base_2050': 'Base50',
                '2 degree_2030': '2deg30',
                '2 degree_2050': '2deg50',
                '1.5 degree_2030': '15deg30',
                '1.5 degree_2050': '15deg50'
            }
            
            # Save statistics to Excel
            excel_file = output_dir / f"monte_carlo_stats_{scenario_mapping[scenario]}.xlsx"
            print(f"Saving statistics for {scenario} to {excel_file}...")
            
            with pd.ExcelWriter(excel_file) as writer:
                for tech in results['monte_carlo_statistics'][scenario]:
                    # Convert statistics to DataFrame
                    df = pd.DataFrame(results['monte_carlo_statistics'][scenario][tech]).T
                    # Save to Excel
                    df.to_excel(writer, sheet_name=f"Stats_{tech}")
    
    print("\nMonte Carlo analysis completed successfully!")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Monte Carlo analysis with advanced methodology')
    parser.add_argument('--input', default='data/TEA input.xlsx', help='Input data file')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--samples', type=int, default=1000, help='Number of Monte Carlo samples')
    args = parser.parse_args()
    
    start_time = time.time()
    run_monte_carlo_analysis(
        input_file=args.input,
        output_dir=args.output,
        num_samples=args.samples
    )
    end_time = time.time()
    
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes") 