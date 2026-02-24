# Global Techno-Economic Analysis (TEA) Model

## Overview

This repository contains the Global TEA model for evaluating the techno-economic feasibility of various energy conversion technologies across different countries and scenarios. The model supports both deterministic and Monte Carlo uncertainty analysis for comprehensive cost assessment.

## Features

- **Multi-technology Analysis**: Supports various energy conversion technologies including electrolysis, direct air capture, synthetic fuel production, and more
- **Global Coverage**: Analysis across multiple countries with country-specific economic parameters
- **Scenario Analysis**: Multiple climate and economic scenarios (Base 2024/2030/2050, 2°C, 1.5°C pathways)
- **Monte Carlo Uncertainty**: Advanced uncertainty quantification with technology-specific distributions
- **Multi-product Technologies**: Handles technologies that produce multiple outputs (e.g., Fischer-Tropsch producing both diesel and kerosene)
- **Supply Chain Integration**: Sequential modeling of hydrogen, CO2 capture, and synthetic fuel production
- **Comprehensive Visualization**: Extensive plotting capabilities for results analysis

## Repository Structure

```
Model code package/
├── data/                          # Input data files
│   ├── TEA input.xlsx             # Main input data (technologies, costs, scenarios)
│   ├── country_lcoe_cf.csv        # Country-specific LCOE and capacity factors
│   ├── f_wacc_c.csv               # Country-specific WACC factors
│   ├── gdp_2024.csv               # GDP data for economic analysis
│   └── technology_wacc_input.csv  # Technology-specific WACC inputs
├── global tea/
│   ├── core/                      # Core model functions
│   │   ├── main.py                # Main execution script
│   │   ├── functions.py           # Core calculation functions
│   │   └── monte_carlo.py         # Monte Carlo analysis functions
│   ├── analysis/                  # Analysis scripts
│   │   ├── comprehensive_sensitivity_analysis.py
│   │   ├── country_ranked_cost_analysis.py
│   │   ├── green_countries_cost_analysis.py
│   │   └── total_delivered_cost_analysis.py
│   ├── visualization/             # Plotting and visualization
│   │   ├── aggregatedplot.py
│   │   ├── costplotbar.py
│   │   ├── plot_individual_maps.py
│   │   ├── plot_optimal_technology_maps.py
│   │   ├── plot.py
│   │   ├── radial_cost_visualization.py
│   │   └── visualize_delivered_cost_routes.py
│   └── Input data processing/     # Data preprocessing utilities
│       ├── calculate_f_wacc_t.py
│       ├── calculate_future_capex.py
│       ├── export_capex_format.py
│       └── lcoe.py
└── output/                        # Results and output files
    ├── figures/                   # Generated plots and visualizations
    └── results/                   # Excel files with numerical results
```

## Installation

1. **Prerequisites**: Python 3.8 or higher

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```python
   import pandas as pd
   import numpy as np
   import geopandas as gpd
   print("All dependencies installed successfully!")
   ```

## Quick Start

### Basic Usage

1. **Navigate to the core directory**:
   ```bash
   cd "global tea/core"
   ```

2. **Run the main model** (deterministic analysis):
   ```bash
   python main.py --no-monte-carlo
   ```

3. **Run with Monte Carlo analysis**:
   ```bash
   python main.py --samples 1000
   ```

### Scenarios

- **Base 2024**: Current technology costs
- **Base 2030/2050**: Baseline technology learning curves
- **2°C 2030/2050**: Climate policy scenarios
- **1.5°C 2030/2050**: Aggressive decarbonization scenarios

### Monte Carlo Analysis

The model includes comprehensive uncertainty analysis:

```python
from monte_carlo import run_monte_carlo_analysis

# Run Monte Carlo with custom parameters
results = run_monte_carlo_analysis(
    input_file='data/TEA input.xlsx',
    output_dir='output',
    num_samples=1000,
    scenarios=['Base_2024', 'Base_2030'],
    save_results=True
)
```

### Custom Analysis Scripts

Use the analysis modules for specific studies:

```python
# Country ranking analysis
from analysis.country_ranked_cost_analysis import analyze_country_costs

# Green countries analysis
from analysis.green_countries_cost_analysis import identify_green_countries

# Sensitivity analysis
from analysis.comprehensive_sensitivity_analysis import run_sensitivity_analysis
```

## Output Files

### Results Structure

- **lcox_results.xlsx**: Levelized cost of X (LCOX) for all technologies
- **monte_carlo_stats_[scenario].xlsx**: Statistical analysis of Monte Carlo results
- **Maritime_Routes_Segments.xlsx**: Transport cost analysis
- **Technology comparison files**: Comparative analysis across technologies

### Visualization Outputs

- **Cost maps**: Geographic visualization of technology costs
- **Technology maps**: Optimal technology selection by region
- **Cost breakdown charts**: Component-wise cost analysis
- **Uncertainty plots**: Monte Carlo result distributions

## Contact

For questions or support, please contact zipeng.liu@psi.ch.
