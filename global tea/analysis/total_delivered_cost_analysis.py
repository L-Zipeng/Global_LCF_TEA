"""
Total Delivered Cost Analysis: PEM LCOH + Transport to Basel
===========================================================

This script combines PEM hydrogen production costs (LCOH) with transport costs
to Basel/Switzerland to create comprehensive delivered cost maps similar to 
the best_routes visualizations.

Author: Global TEA Analysis
Date: 2024
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from shapely.geometry import shape
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def preprocess_geojson_safely(geojson_path):
    """
    Load and preprocess GeoJSON, skipping invalid geometries and removing Antarctica.
    """
    try:
        with open(geojson_path, 'r') as file:
            raw_data = json.load(file)
        
        processed_features = []
        for feature in raw_data["features"]:
            try:
                geom = shape(feature["geometry"])
                if feature["properties"].get("iso_a3") == "ATA":
                    continue
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if geom.is_valid and not geom.is_empty:
                    processed_features.append({
                        "geometry": geom,
                        "ISO_A3": feature["properties"].get("iso_a3", None),
                        "name": feature["properties"].get("name", None)
                    })
            except Exception as e:
                print(f"Skipping problematic feature for {feature['properties'].get('name', 'Unknown')}: {e}")
        
        if not processed_features:
            raise ValueError("No valid features found in GeoJSON file")
        
        gdf = gpd.GeoDataFrame(processed_features, crs="EPSG:4326")
        print(f"Successfully processed {len(gdf)} features")
        return gdf
    
    except Exception as e:
        print(f"Error processing GeoJSON: {e}")
        raise

def load_pem_production_costs(lcox_path='output/lcox_results.xlsx'):
    """
    Load PEM hydrogen production costs from LCOX results file.
    
    Returns:
        dict: Dictionary with year as key and dataframe with country costs as value
    """
    production_costs = {}
    
    # Define sheet mapping for different years
    year_sheets = {
        '2024': 'LCOX_Base24',
        '2030': 'LCOX_Base30', 
        '2050': 'LCOX_Base50'
    }
    
    try:
        for year, sheet_name in year_sheets.items():
            try:
                df = pd.read_excel(lcox_path, sheet_name=sheet_name, index_col=0)
                
                # Extract PEM costs - the countries are in the index
                if 'PEM' in df.columns:
                    pem_df = df[['PEM']].reset_index()
                    pem_df.columns = ['ISO_A3', 'PEM_LCOH']
                    
                    # Remove rows with NaN costs
                    pem_df = pem_df.dropna(subset=['PEM_LCOH'])
                    
                    # Convert country codes to strings and clean them
                    pem_df['ISO_A3'] = pem_df['ISO_A3'].astype(str).str.strip()
                    
                    # Debug: Print sample of actual country codes
                    print(f"Sample {year} PEM countries: {pem_df['ISO_A3'].head(10).tolist()}")
                    
                    production_costs[year] = pem_df
                    print(f"Loaded PEM costs for {year}: {len(pem_df)} countries")
                else:
                    print(f"Warning: PEM column not found in {sheet_name}")
                    
            except Exception as e:
                print(f"Warning: Could not load {sheet_name} from {lcox_path}: {e}")
                
    except Exception as e:
        print(f"Error loading LCOX file {lcox_path}: {e}")
        return {}
    
    return production_costs

def load_transport_costs(transport_path='output/Switzerland_Transport_Pathways.csv'):
    """
    Load transport costs to Switzerland/Basel.
    
    Returns:
        pandas.DataFrame: Transport costs by country, year, scenario
    """
    try:
        # Try CSV first, then Excel
        if Path(transport_path).exists():
            transport_df = pd.read_csv(transport_path)
        else:
            # Try Excel version
            excel_path = transport_path.replace('.csv', '.xlsx')
            if Path(excel_path).exists():
                transport_df = pd.read_excel(excel_path)
            else:
                raise FileNotFoundError(f"Transport data not found at {transport_path} or {excel_path}")
        
        # Clean and standardize column names and data
        transport_df['origin_country'] = transport_df['origin_country'].astype(str).str.strip()
        transport_df['year'] = transport_df['year'].astype(str)
        
        # Filter for the best route per country/year/scenario (minimum cost)
        best_routes = transport_df.loc[transport_df.groupby(['origin_country', 'year', 'scenario'])['total_cost'].idxmin()]
        
        print(f"Loaded transport costs: {len(best_routes)} best routes")
        print(f"Years available: {sorted(best_routes['year'].unique())}")
        print(f"Scenarios available: {sorted(best_routes['scenario'].unique())}")
        
        return best_routes
        
    except Exception as e:
        print(f"Error loading transport costs: {e}")
        return pd.DataFrame()

def combine_costs(production_costs, transport_costs):
    """
    Combine PEM production costs with transport costs.
    
    Returns:
        pandas.DataFrame: Combined costs with total delivered cost
    """
    combined_data = []
    
    # Debug: Print available data
    print("Available production cost years:", list(production_costs.keys()))
    print("Available transport cost years:", transport_costs['year'].unique() if not transport_costs.empty else "No transport data")
    
    if not transport_costs.empty:
        print("Sample transport data countries:", transport_costs['origin_country'].head(10).tolist())
        print("Sample production data countries:", list(production_costs['2022']['ISO_A3'].head(10)) if '2022' in production_costs else "No 2022 data")
    
    for year in production_costs.keys():
        pem_df = production_costs[year]
        
        # Debug: Print PEM data info
        print(f"\nProcessing year {year}:")
        print(f"  PEM countries: {len(pem_df)}")
        print(f"  Sample PEM countries: {pem_df['ISO_A3'].head(10).tolist()}")
        
        # Filter transport costs for this year - try both string and int conversion
        transport_year = transport_costs[
            (transport_costs['year'] == year) | 
            (transport_costs['year'] == int(year))
        ]
        
        print(f"  Transport routes for {year}: {len(transport_year)}")
        
        if len(transport_year) == 0:
            print(f"  No transport data found for year {year}")
            continue
        
        for scenario in transport_year['scenario'].unique():
            transport_scenario = transport_year[transport_year['scenario'] == scenario]
            
            print(f"  Transport routes for {year}-{scenario}: {len(transport_scenario)}")
            print(f"  Sample transport countries: {transport_scenario['origin_country'].head(10).tolist()}")
            
            # Merge PEM and transport costs
            merged = pem_df.merge(
                transport_scenario[['origin_country', 'total_cost', 'carrier', 'transport_mode']], 
                left_on='ISO_A3', 
                right_on='origin_country', 
                how='inner'
            )
            
            print(f"  Merged data points: {len(merged)}")
            
            if len(merged) > 0:
                # Unit conversion constants
                CHF_TO_EUR = 0.95  # Approximate CHF to EUR exchange rate
                KWH_PER_KG_H2 = 33.33  # Lower heating value of hydrogen (kWh/kg)
                
                # Convert transport costs from CHF/kgH2 to EUR/kWh
                merged['transport_cost_eur_kwh'] = merged['total_cost'] * CHF_TO_EUR / KWH_PER_KG_H2
                
                # Calculate total delivered cost (both in EUR/kWh)
                merged['total_delivered_cost'] = merged['PEM_LCOH'] + merged['transport_cost_eur_kwh']
                
                # Keep original transport cost for reference
                merged['transport_cost_original_chf_kg'] = merged['total_cost']
                merged['total_cost'] = merged['transport_cost_eur_kwh']  # Update to EUR/kWh for consistency
                
                merged['year'] = year
                merged['scenario'] = scenario
                
                combined_data.append(merged)
                print(f"  Added {len(merged)} rows for {year}-{scenario}")
    
    if combined_data:
        result_df = pd.concat(combined_data, ignore_index=True)
        print(f"\nCombined costs: {len(result_df)} country-year-scenario combinations")
        return result_df
    else:
        print("\nWarning: No data could be combined")
        return pd.DataFrame()

def create_delivered_cost_maps(combined_df, world_gdf, output_dir='figures/delivered_costs'):
    """
    Create world maps showing total delivered cost (PEM + Transport) to Basel.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define years and scenarios to plot
    years = sorted(combined_df['year'].unique())
    scenarios = sorted(combined_df['scenario'].unique())
    
    # Create a custom colormap (reverse viridis for cost - higher costs in darker colors)
    colors = plt.cm.viridis_r(np.linspace(0, 1, 256))
    cost_cmap = LinearSegmentedColormap.from_list('delivered_costs', colors)
    
    for year in years:
        for scenario in scenarios:
            # Filter data for current year and scenario
            current_data = combined_df[
                (combined_df['year'] == year) & 
                (combined_df['scenario'] == scenario)
            ].copy()
            
            if len(current_data) == 0:
                continue
            
            # Prepare data for mapping
            map_data = current_data[['ISO_A3', 'total_delivered_cost', 'PEM_LCOH', 'total_cost', 'carrier', 'transport_mode']].copy()
            
            # Merge with world geodataframe
            merged_world = world_gdf.merge(map_data, on='ISO_A3', how='left')
            
            # Get min and max for color scaling
            vmin = current_data['total_delivered_cost'].min()
            vmax = current_data['total_delivered_cost'].max()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Plot the map
            merged_world.plot(
                column='total_delivered_cost',
                ax=ax,
                legend=True,
                cmap=cost_cmap,
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
                legend_kwds={
                    'label': 'Total Delivered Cost (EUR/kg H₂)',
                    'orientation': 'horizontal',
                    'shrink': 0.7,
                    'pad': 0.05,
                    'aspect': 30,
                    'fraction': 0.046
                },
                missing_kwds={'color': 'lightgrey', 'alpha': 0.5}
            )
            
            # Highlight Switzerland
            switzerland = world_gdf[world_gdf['ISO_A3'] == 'CHE']
            if not switzerland.empty:
                switzerland.plot(ax=ax, color='red', edgecolor='black', linewidth=2, alpha=0.8)
            
            # Add title
            clean_scenario = scenario.replace('base_', '').replace('_', ' ').title()
            ax.set_title(f'Total Delivered H₂ Cost to Basel (PEM Production + Transport)\n{year} - {clean_scenario}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            
            # Save figure
            filename = f'total_delivered_cost_{year}_{scenario}.png'
            plt.savefig(
                output_path / filename,
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            
            print(f"Saved: {filename}")
    
    print(f"All delivered cost maps saved to {output_path}")

def create_cost_breakdown_maps(combined_df, world_gdf, output_dir='figures/delivered_costs'):
    """
    Create maps showing the breakdown between production and transport costs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    years = sorted(combined_df['year'].unique())
    scenarios = sorted(combined_df['scenario'].unique())
    
    for year in years:
        for scenario in scenarios:
            # Filter data
            current_data = combined_df[
                (combined_df['year'] == year) & 
                (combined_df['scenario'] == scenario)
            ].copy()
            
            if len(current_data) == 0:
                continue
            
            # Calculate percentage split
            current_data['production_share'] = current_data['PEM_LCOH'] / current_data['total_delivered_cost'] * 100
            current_data['transport_share'] = current_data['total_cost'] / current_data['total_delivered_cost'] * 100
            
            # Create subplot for breakdown
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Merge with world data
            map_data = current_data[['ISO_A3', 'production_share', 'transport_share']].copy()
            merged_world = world_gdf.merge(map_data, on='ISO_A3', how='left')
            
            # Plot production cost share
            merged_world.plot(
                column='production_share',
                ax=ax1,
                legend=True,
                cmap='RdYlBu_r',
                norm=plt.Normalize(vmin=0, vmax=100),
                legend_kwds={
                    'label': 'Production Cost Share (%)',
                    'orientation': 'horizontal',
                    'shrink': 0.8,
                    'pad': 0.05
                },
                missing_kwds={'color': 'lightgrey', 'alpha': 0.5}
            )
            ax1.set_title('Production Cost Share (%)', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Plot transport cost share
            merged_world.plot(
                column='transport_share',
                ax=ax2,
                legend=True,
                cmap='RdYlBu',
                norm=plt.Normalize(vmin=0, vmax=100),
                legend_kwds={
                    'label': 'Transport Cost Share (%)',
                    'orientation': 'horizontal',
                    'shrink': 0.8,
                    'pad': 0.05
                },
                missing_kwds={'color': 'lightgrey', 'alpha': 0.5}
            )
            ax2.set_title('Transport Cost Share (%)', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # Highlight Switzerland on both maps
            switzerland = world_gdf[world_gdf['ISO_A3'] == 'CHE']
            if not switzerland.empty:
                switzerland.plot(ax=ax1, color='red', edgecolor='black', linewidth=1.5, alpha=0.8)
                switzerland.plot(ax=ax2, color='red', edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # Add overall title
            clean_scenario = scenario.replace('base_', '').replace('_', ' ').title()
            fig.suptitle(f'Cost Breakdown: Production vs Transport to Basel\n{year} - {clean_scenario}', 
                        fontsize=16, fontweight='bold')
            
            # Save figure
            filename = f'cost_breakdown_{year}_{scenario}.png'
            plt.savefig(
                output_path / filename,
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
            )
            plt.close()
            
            print(f"Saved: {filename}")

def create_summary_statistics(combined_df, output_dir='figures/delivered_costs'):
    """
    Create summary statistics and comparison plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary table
    summary_stats = combined_df.groupby(['year', 'scenario']).agg({
        'PEM_LCOH': ['mean', 'min', 'max', 'std'],
        'total_cost': ['mean', 'min', 'max', 'std'],
        'total_delivered_cost': ['mean', 'min', 'max', 'std']
    }).round(3)
    
    # Save summary statistics
    summary_stats.to_excel(output_path / 'delivered_cost_summary_statistics.xlsx')
    
    # Create box plots comparing scenarios
    years = sorted(combined_df['year'].unique())
    
    for year in years:
        year_data = combined_df[combined_df['year'] == year]
        
        if len(year_data) == 0:
            continue
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Production costs
        sns.boxplot(data=year_data, x='scenario', y='PEM_LCOH', ax=ax1)
        ax1.set_title('PEM Production Costs')
        ax1.set_ylabel('Cost (EUR/kg H₂)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Transport costs
        sns.boxplot(data=year_data, x='scenario', y='total_cost', ax=ax2)
        ax2.set_title('Transport Costs')
        ax2.set_ylabel('Cost (EUR/kg H₂)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Total delivered costs
        sns.boxplot(data=year_data, x='scenario', y='total_delivered_cost', ax=ax3)
        ax3.set_title('Total Delivered Costs')
        ax3.set_ylabel('Cost (EUR/kg H₂)')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Cost Distribution Comparison - {year}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f'cost_comparison_boxplot_{year}.png'
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    print(f"Summary statistics and comparisons saved to {output_path}")

def main():
    """Main function to run the total delivered cost analysis."""
    
    # Define file paths
    lcox_path = 'output/lcox_results.xlsx'
    transport_path = 'output/Switzerland_Transport_Pathways.csv'
    geojson_path = 'input/world_by_iso_geo.json'
    output_dir = 'figures/delivered_costs'
    
    print("Starting Total Delivered Cost Analysis...")
    print("=" * 50)
    
    # Check if required files exist
    if not Path(lcox_path).exists():
        print(f"Error: LCOX results file not found at {lcox_path}")
        return
    
    if not Path(transport_path).exists():
        # Try Excel version
        transport_path_excel = transport_path.replace('.csv', '.xlsx')
        if not Path(transport_path_excel).exists():
            print(f"Error: Transport results file not found at {transport_path} or {transport_path_excel}")
            return
        transport_path = transport_path_excel
    
    if not Path(geojson_path).exists():
        print(f"Error: GeoJSON file not found at {geojson_path}")
        return
    
    try:
        # Load data
        print("\n1. Loading PEM production costs...")
        production_costs = load_pem_production_costs(lcox_path)
        
        print("\n2. Loading transport costs...")
        transport_costs = load_transport_costs(transport_path)
        
        print("\n3. Loading world map data...")
        world_gdf = preprocess_geojson_safely(geojson_path)
        
        if not production_costs or transport_costs.empty:
            print("Error: Could not load required data")
            return
        
        # Combine costs
        print("\n4. Combining production and transport costs...")
        combined_df = combine_costs(production_costs, transport_costs)
        
        if combined_df.empty:
            print("Error: No data could be combined")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save combined data
        combined_df.to_excel(output_path / 'combined_delivered_costs.xlsx', index=False)
        combined_df.to_csv(output_path / 'combined_delivered_costs.csv', index=False)
        
        # Create visualizations
        print("\n5. Creating delivered cost maps...")
        create_delivered_cost_maps(combined_df, world_gdf, output_dir)
        
        print("\n6. Creating cost breakdown maps...")
        create_cost_breakdown_maps(combined_df, world_gdf, output_dir)
        
        print("\n7. Creating summary statistics...")
        create_summary_statistics(combined_df, output_dir)
        
        print(f"\n{'='*50}")
        print("Analysis completed successfully!")
        print(f"All outputs saved to: {output_dir}")
        print(f"Data files: combined_delivered_costs.xlsx/csv")
        print(f"Figures: total_delivered_cost_*.png, cost_breakdown_*.png, cost_comparison_*.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()