# This script is used to plot individual world maps for specific technologies
# Figure 2

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re
import numpy as np
from pathlib import Path

def preprocess_geojson_safely(geojson_path):
    """
    Load and preprocess GeoJSON, skipping invalid geometries and removing Antarctica.
    """
    import json
    with open(geojson_path, 'r') as file:
        raw_data = json.load(file)
    
    processed_features = []
    for feature in raw_data["features"]:
        try:
            geom = shape(feature["geometry"])
            # Remove Antarctica by ISO_A3 code
            if feature["properties"].get("iso") == "ATA":
                continue
            # Fix invalid geometries
            if not geom.is_valid:
                geom = geom.buffer(0)  # Attempt to fix
            # Simplify MultiPolygons
            if geom.geom_type == "MultiPolygon":
                geom = unary_union(geom)
            if geom.is_valid and not geom.is_empty:
                processed_features.append({
                    "geometry": geom,
                    "ISO_A3": feature["properties"].get("iso", None),
                    "name": feature["properties"].get("name", None)
                })
        except Exception as e:
            print(f"Skipping problematic feature: {e}")
    
    return gpd.GeoDataFrame(processed_features, crs="EPSG:4326")

def set_plot_style():
    """Set the style for the plots to match Nature journal aesthetics"""
    # Replace deprecated 'seaborn-whitegrid' with direct style settings
    plt.style.use('default')
    
    # Nature-style elements - VERY LARGE font sizes for subfigure use
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 28
    mpl.rcParams['axes.labelsize'] = 32
    mpl.rcParams['axes.titlesize'] = 36
    mpl.rcParams['xtick.labelsize'] = 26
    mpl.rcParams['ytick.labelsize'] = 26
    mpl.rcParams['legend.fontsize'] = 28
    mpl.rcParams['figure.titlesize'] = 40
    
    # Professional looking lines and markers
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['patch.linewidth'] = 0.5
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.edgecolor'] = '#555555'
    
    # Clean background
    mpl.rcParams['axes.facecolor'] = '#f8f8f8'
    mpl.rcParams['figure.facecolor'] = 'white'
    
    # Better margins
    mpl.rcParams['figure.constrained_layout.use'] = True

def plot_individual_world_maps(lcox_data, geojson_path, technologies, output_dir='figures'):
    """
    Create individual publication-quality world maps for specified technologies,
    with one map per scenario.
    """
    # Load and preprocess GeoJSON
    world = preprocess_geojson_safely(geojson_path)

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set plot style for publication quality
    set_plot_style()

    # Scenario mapping for better labels
    scenario_name_mapping = {
        'LCOX_Base24': 'Reference 2024',
        'LCOX_Base30': '2030 Business-as-usual',
        'LCOX_2deg30': '2030 2°C Scenario',
        'LCOX_15deg30': '2030 1.5°C Scenario',
        'LCOX_Base50': '2050 Business-as-usual',
        'LCOX_2deg50': '2050 2°C Scenario',
        'LCOX_15deg50': '2050 1.5°C Scenario'
    }

    # Get valid scenarios
    valid_scenarios = [s for s in scenario_name_mapping.keys() if s in lcox_data.keys()]
    
    # Group scenarios by year for consistent color scaling
    year_groups = {
        '2024': ['LCOX_Base24'],
        '2030': ['LCOX_Base30', 'LCOX_2deg30', 'LCOX_15deg30'],
        '2050': ['LCOX_Base50', 'LCOX_2deg50', 'LCOX_15deg50']
    }
    
    # Process each technology
    for tech in technologies:
        print(f"Processing individual world maps for {tech}")
        
        # Check if technology exists in data
        if tech not in next(iter(lcox_data.values())).columns:
            print(f"Warning: Technology {tech} not found in data")
            continue

        # Calculate color ranges for this technology
        # Combine 2022 and 2050 scenarios to use the same color bar
        year_ranges = {}
        
        # Calculate combined range for 2022 and 2050
        combined_2022_2050_values = []
        for year in ['2022', '2050']:
            scenarios = year_groups[year]
            for scenario in scenarios:
                if scenario in lcox_data:
                    data = lcox_data[scenario]
                    if tech in data.columns:
                        combined_2022_2050_values.extend(data[tech].dropna().values)
        
        if combined_2022_2050_values:
            vmin_combined, vmax_combined = np.percentile(combined_2022_2050_values, [5, 95])
            # Ensure there's some range for visualization
            if vmax_combined - vmin_combined < 0.01:
                vmax_combined = vmin_combined + 0.01
            # Assign the same range to both 2022 and 2050
            year_ranges['2022'] = (vmin_combined, vmax_combined)
            year_ranges['2050'] = (vmin_combined, vmax_combined)
            print(f"  2022 & 2050 combined range for {tech}: {vmin_combined:.3f} - {vmax_combined:.3f} EUR/kWh")
        
        # Calculate separate range for 2030
        year_tech_values_2030 = []
        for scenario in year_groups['2030']:
            if scenario in lcox_data:
                data = lcox_data[scenario]
                if tech in data.columns:
                    year_tech_values_2030.extend(data[tech].dropna().values)
        
        if year_tech_values_2030:
            vmin_2030, vmax_2030 = np.percentile(year_tech_values_2030, [5, 95])
            # Ensure there's some range for visualization
            if vmax_2030 - vmin_2030 < 0.01:
                vmax_2030 = vmin_2030 + 0.01
            year_ranges['2030'] = (vmin_2030, vmax_2030)
            print(f"  2030 range for {tech}: {vmin_2030:.3f} - {vmax_2030:.3f} EUR/kWh")

        # Process each scenario using year-specific color ranges
        for sheet_name in valid_scenarios:
            print(f"  Plotting {tech} - {sheet_name}")
            
            # Create individual figure for this scenario
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            
            data = lcox_data[sheet_name]
            tech_data = data[[tech]].copy()
            tech_data.index.name = 'ISO_A3'
            tech_data.reset_index(inplace=True)

            # Determine which year this scenario belongs to and get the appropriate range
            scenario_year = None
            for year, scenarios in year_groups.items():
                if sheet_name in scenarios:
                    scenario_year = year
                    break
            
            if scenario_year is None or scenario_year not in year_ranges:
                print(f"    No year range found for {sheet_name}")
                continue
                
            vmin, vmax = year_ranges[scenario_year]
            
            merged_data = world.merge(tech_data, on='ISO_A3', how='left')

            # Plot the map
            im = merged_data.plot(
                column=tech,
                ax=ax,
                legend=False,
                cmap='viridis_r',
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
                missing_kwds={'color': 'lightgrey', 'edgecolor': 'white', 'linewidth': 0.1}
            )
            
            # Style the map
            ax.axis('off')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 85)  # Exclude Antarctica region
            
            # Add title
            scenario_title = scenario_name_mapping[sheet_name]
            tech_display_name = tech.replace('_', '-')
            
            if tech == 'RWGS_FT_kerosene':
                tech_display_name = 'RWGS-FT Kerosene'
            
            ax.set_title(f'{tech_display_name} - {scenario_title}', 
                        fontsize=36, fontweight='bold', pad=20)

            # Add colorbar
            cbar = plt.colorbar(im.get_children()[0], ax=ax, 
                              orientation='horizontal', 
                              shrink=0.9,  # Increased from 0.8 to 0.9 for larger size
                              pad=0.02,    # Reduced from 0.05 to 0.02 to move higher
                              aspect=40)   # Reduced from 50 to 40 for slightly thicker bar
            
            # Set technology-specific colorbar label
            if tech == 'PEM':
                cbar_label = 'Levelized cost of hydrogen (PEM) [EUR/kWh]'
            elif tech == 'RWGS_FT_kerosene':
                cbar_label = 'Levelized cost of kerosene (PEM-DAC-RWGS-FT) [EUR/kWh]'
            else:
                cbar_label = 'Levelized Cost of Production [EUR/kWh]'  # Default for other technologies
            
            cbar.set_label(cbar_label, fontsize=30, labelpad=20)  # VERY LARGE font sizes for subfigures
            cbar.ax.tick_params(labelsize=26)  # VERY LARGE for subfigure use
            
            # Format colorbar ticks
            cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

            # Save plot with high resolution
            scenario_short = sheet_name.replace('LCOX_', '')
            filename = f'world_map_{tech}_{scenario_short}.png'
            plt.savefig(output_dir / filename,
                       bbox_inches='tight', 
                       dpi=300,
                       facecolor='white',
                       edgecolor='none')
            plt.close()
            print(f"    Saved figure to {output_dir}/{filename}")

def main():
    # Load your LCOX results
    results_file = 'output/lcox_results.xlsx'
    geojson_file = 'input/world_by_iso_geo.json'

    # Specify the technologies to plot
    technologies_to_plot = ['PEM', 'RWGS_FT_kerosene']

    # Read LCOX data
    try:
        lcox_xlsx = pd.ExcelFile(results_file)
        lcox_data = {
            sheet: pd.read_excel(results_file, sheet_name=sheet, index_col=0)
            for sheet in lcox_xlsx.sheet_names if sheet.startswith('LCOX_')
        }
        
        print(f"Loaded data for scenarios: {list(lcox_data.keys())}")
        print(f"Available technologies: {list(next(iter(lcox_data.values())).columns)}")
        
        # Generate individual plots
        plot_individual_world_maps(lcox_data, geojson_file, technologies_to_plot)
        
    except FileNotFoundError:
        print(f"Error: Could not find {results_file}")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    main() 