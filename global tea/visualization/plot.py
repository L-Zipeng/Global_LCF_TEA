# This script is used to plot the world maps for the LCOX values in supplementary figures

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
    
    # Nature-style elements
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['figure.titlesize'] = 12
    
    # Professional looking lines and markers
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['patch.linewidth'] = 0.5
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.edgecolor'] = '#555555'
    
    # Add grid for scientific appearance
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = ':'
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.alpha'] = 0.5
    
    # Elegant figure size for world maps
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['figure.dpi'] = 300
    
    # Clean background
    mpl.rcParams['axes.facecolor'] = '#f8f8f8'
    mpl.rcParams['figure.facecolor'] = 'white'
    
    # Better margins
    mpl.rcParams['figure.constrained_layout.use'] = True

def plot_world_map(lcox_data, geojson_path, output_dir='figures'):
    """
    Create publication-quality world maps for LCOX values with a 3x3 grid layout.
    """
    # Load and preprocess GeoJSON
    world = preprocess_geojson_safely(geojson_path)

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set plot style for publication quality
    set_plot_style()

    # Define technology groups for consistent scaling
    electrolysis_techs = ['PEM', 'AE', 'SOEC', 'HTSE', 'CuCl']
    fossil_techs = ['SMR_CCS', 'ATR_CCS', 'CLR']

    # Scenario mapping for better labels
    scenario_name_mapping = {
        'LCOX_Base24': 'Ref. 2024',
        'LCOX_Base30': '2030 BAU',
        'LCOX_2deg30': '2030 2°C',
        'LCOX_15deg30': '2030 1.5°C',
        'LCOX_Base50': '2050 BAU',
        'LCOX_2deg50': '2050 2°C',
        'LCOX_15deg50': '2050 1.5°C'
    }

    # Define specific layout positions for each scenario in a 3x3 grid
    scenario_positions = {
        'LCOX_Base24': (0, 0),  # Reference 2024 in top left
        # Positions (0,1) and (0,2) are left blank
        'LCOX_Base30': (1, 0),  # 2030 scenarios in second row
        'LCOX_2deg30': (1, 1),
        'LCOX_15deg30': (1, 2),
        'LCOX_Base50': (2, 0),  # 2050 scenarios in third row
        'LCOX_2deg50': (2, 1),
        'LCOX_15deg50': (2, 2)
    }

    # Get valid scenarios
    valid_scenarios = [s for s in scenario_positions.keys() if s in lcox_data.keys()]
    
    # Calculate global min and max for each technology group
    electrolysis_values = []
    fossil_values = []
    for data in lcox_data.values():
        for tech in electrolysis_techs:
            if tech in data.columns:
                electrolysis_values.extend(data[tech].dropna().values)
        for tech in fossil_techs:
            if tech in data.columns:
                fossil_values.extend(data[tech].dropna().values)

    electrolysis_vmin, electrolysis_vmax = np.percentile(electrolysis_values, [5, 95])
    fossil_vmin, fossil_vmax = np.percentile(fossil_values, [5, 95])
    
    # Process each technology
    for tech in next(iter(lcox_data.values())).columns:
        print(f"Processing world map for {tech}")

        # Set appropriate scale based on technology group
        if tech in electrolysis_techs:
            vmin, vmax = electrolysis_vmin, electrolysis_vmax
        elif tech in fossil_techs:
            vmin, vmax = fossil_vmin, fossil_vmax
        else:
            tech_values = []
            for data in lcox_data.values():
                tech_values.extend(data[tech].dropna().values)
            vmin, vmax = np.percentile(tech_values, [5, 95])

        # Create figure with 3x3 grid layout - increased width for larger maps
        fig = plt.figure(figsize=(20, 12))  # Significantly wider figure
        
        # Create gridspec with 3 rows and 3 columns - tighter spacing
        gs = fig.add_gridspec(3, 3,
                            hspace=0.05,   # Minimal space between rows
                            wspace=0.01,   # Minimal space between columns
                            left=0.01,     # Minimal left margin
                            right=0.99,    # Maximal right margin
                            top=0.95,      # Larger top margin for title
                            bottom=0.05)   # Small bottom margin

        # Plot each scenario
        for sheet_name in valid_scenarios:
            row, col = scenario_positions[sheet_name]
            print(f"Plotting {sheet_name} at position ({row}, {col})")

            ax = fig.add_subplot(gs[row, col])
            
            data = lcox_data[sheet_name]
            tech_data = data[[tech]]
            tech_data.index.name = 'ISO_A3'
            tech_data.reset_index(inplace=True)

            merged_data = world.merge(tech_data, on='ISO_A3', how='left')

            merged_data.plot(
                column=tech,
                ax=ax,
                legend=False,
                cmap='viridis_r',
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
                missing_kwds={'color': 'lightgrey'}
            )
            
            ax.axis('off')
            ax.set_title(scenario_name_mapping[sheet_name], 
                        fontsize=12, fontweight='bold', pad=3)  # Further reduced padding

            # Add scenario year annotation in the corner of each map
            year = "2024" if sheet_name == 'LCOX_Base24' else "2030" if '30' in sheet_name else "2050"
            ax.annotate(f"{year}", xy=(0.05, 0.05), xycoords='axes fraction', 
                       fontsize=9, fontweight='bold', 
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Add colorbar at the bottom
        cax = fig.add_axes([0.15, 0.01, 0.7, 0.03])  # [left, bottom, width, height] - wider and taller
        cax.set_visible(True)
        norm = plt.Normalize(vmin=0, vmax=vmax)  # Start from 0
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis_r')
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        
        # Set appropriate units based on technology
        if tech == 'DAC':
            cbar.set_label('Levelized Cost of Production [EUR/kg]', fontsize=12)
        else:
            cbar.set_label('Levelized Cost of Production [EUR/kWh]', fontsize=12)
            
        cbar.ax.tick_params(labelsize=12)  # Larger font size for numbers

        # Add main title - moved up slightly
        if tech == 'DAC':
            fig.suptitle(f'Global Direct Air Capture Cost Distribution - {tech}',
                        fontsize=14, fontweight='bold', 
                        y=0.97)  # Positioned higher
        else:
            fig.suptitle(f'Global Production Cost Distribution - {tech}',
                        fontsize=14, fontweight='bold', 
                        y=0.97)  # Positioned higher

        # Save plot with high resolution and tight layout
        plt.savefig(output_dir / f'world_map_{tech}_3x3.png',
                   bbox_inches='tight', 
                   dpi=600)
        plt.close()
        print(f"  Saved figure to {output_dir}/world_map_{tech}_3x3.png")

def main():
    # Load your LCOX results
    results_file = 'output/lcox_results.xlsx'
    geojson_file = 'input/world_by_iso_geo.json'

    # Read LCOX data
    lcox_xlsx = pd.ExcelFile(results_file)
    lcox_data = {
        sheet: pd.read_excel(results_file, sheet_name=sheet, index_col=0)
        for sheet in lcox_xlsx.sheet_names if sheet.startswith('LCOX_')
    }

    # Generate plots
    plot_world_map(lcox_data, geojson_file)

if __name__ == "__main__":
    main()
