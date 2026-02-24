"""
Visualize Delivered Cost Routes: PEM Production + Transport to Basel
===================================================================

This script creates the same style maps as the original best_routes visualizations
but uses total delivered cost (PEM LCOH + transport) instead of just transport cost.

Based on analysis/visualize_best_routes.py but modified to use combined costs.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
import os
from shapely.geometry import shape
from shapely.ops import unary_union
import json

def get_major_economies():
    """
    Get list of major economies to focus visualization on ~20-25 countries
    Based on 2024 GDP rankings and economic significance for hydrogen markets
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
    }
    return major_economies

# Transport configuration constants (from original script)
TRANSPORT_SPEEDS = {
    'ship': 800,      # km/day (around 33 km/h or 18 knots)
    'truck': 600,     # km/day (average speed 25 km/h including stops)
    'rail': 720,      # km/day (average speed 30 km/h including stops)
    'pipeline': 960,  # km/day (average speed 40 km/h)
    'barge': 480      # km/day (average speed 20 km/h)
}

CARRIER_CONVERSION_FACTORS = {
    'CH2': 1.0,      # 1 kg H2 = 1 kg CH2
    'LH2': 1.0,      # 1 kg H2 = 1 kg LH2
    'NH3': 5.9,      # 1 kg H2 = 5.9 kg NH3
    'MeOH': 8,       # 1 kg H2 = 8.0 kg MeOH
    'LOHC': 6        # 1 kg H2 = 6.0 kg LOHC
}

# Add capital city coordinates dictionary for plotting
CAPITAL_COORDINATES = {
    'AFG': (69.1761, 34.5228), 'ALB': (19.8172, 41.3317), 'DZA': (3.0588, 36.7538),
    'AGO': (13.2343, -8.8383), 'ARG': (-58.3816, -34.6037), 'AUS': (149.1300, -35.2809),
    'AUT': (16.3738, 48.2082), 'BHR': (50.5854, 26.2285), 'BGD': (90.4125, 23.8103),
    'BLR': (27.5534, 53.9045), 'BEL': (4.3517, 50.8503), 'BEN': (2.6323, 6.4779),
    'BOL': (-68.1193, -16.4897), 'BIH': (18.3564, 43.8563), 'BWA': (25.9201, -24.6282),
    'BRA': (-47.8645, -15.7942), 'BGR': (23.3219, 42.6977), 'BFA': (-1.5247, 12.3714),
    'KHM': (104.9225, 11.5564), 'CMR': (11.5021, 3.8480), 'CAN': (-75.6972, 45.4215),
    'CHL': (-70.6483, -33.4489), 'CHN': (116.4074, 39.9042), 'COL': (-74.0721, 4.7110),
    'CRI': (-84.0877, 9.9281), 'HRV': (15.9779, 45.8150), 'CUB': (-82.3666, 23.1136),
    'CYP': (33.3823, 35.1856), 'CZE': (14.4378, 50.0755), 'DNK': (12.5683, 55.6761),
    'DOM': (-69.9312, 18.4861), 'ECU': (-78.5243, -0.2295), 'EGY': (31.2357, 30.0444),
    'SLV': (-89.2090, 13.6929), 'EST': (24.7536, 59.4369), 'ETH': (38.7578, 9.0084),
    'FIN': (24.9384, 60.1699), 'FRA': (2.3522, 48.8566), 'GAB': (9.4496, 0.4162),
    'DEU': (13.4050, 52.5200), 'GHA': (-0.1870, 5.6037), 'GRC': (23.7275, 37.9838),
    'GTM': (-90.5133, 14.6349), 'GIN': (-13.6773, 9.5370), 'HND': (-87.1715, 14.0723),
    'HUN': (19.0402, 47.4979), 'ISL': (-21.8954, 64.1265), 'IND': (77.2090, 28.6139),
    'IDN': (106.8456, -6.2088), 'IRN': (51.3890, 35.6892), 'IRQ': (44.3661, 33.3152),
    'IRL': (-6.2603, 53.3498), 'ISR': (35.2137, 31.7683), 'ITA': (12.4964, 41.9028),
    'JPN': (139.7690, 35.6804), 'JOR': (35.9106, 31.9539), 'KAZ': (71.4704, 51.1605),
    'KEN': (36.8219, -1.2921), 'KWT': (47.9774, 29.3759), 'LVA': (24.1052, 56.9496),
    'LBN': (35.5018, 33.8938), 'LBY': (13.1875, 32.8872), 'LTU': (25.2797, 54.6872),
    'LUX': (6.1320, 49.6116), 'MYS': (101.6869, 3.1390), 'MLT': (14.5145, 35.8992),
    'MEX': (-99.1332, 19.4326), 'MDA': (28.8575, 47.0105), 'MNG': (106.9057, 47.8864),
    'MNE': (19.2636, 42.4304), 'MAR': (-6.8498, 34.0209), 'MOZ': (32.5732, -25.9692),
    'MMR': (96.1561, 19.7633), 'NAM': (17.0835, -22.5609), 'NPL': (85.3240, 27.7172),
    'NLD': (4.9041, 52.3676), 'NZL': (174.7762, -41.2866), 'NIC': (-86.2504, 12.1149),
    'NGA': (7.4898, 9.0765), 'NOR': (10.7522, 59.9139), 'OMN': (58.5922, 23.6139),
    'PAK': (73.0479, 33.6844), 'PAN': (-79.5342, 8.9824), 'PRY': (-57.3333, -25.2867),
    'PER': (-77.0428, -12.0464), 'PHL': (120.9842, 14.5995), 'POL': (21.0122, 52.2297),
    'PRT': (-9.1393, 38.7223), 'QAT': (51.5310, 25.2867), 'ROU': (26.1025, 44.4268),
    'RUS': (37.6173, 55.7558), 'SAU': (46.7219, 24.6877), 'SEN': (-17.4734, 14.7167),
    'SRB': (20.4612, 44.8125), 'SGP': (103.8198, 1.3521), 'SVK': (17.1077, 48.1486),
    'SVN': (14.5058, 46.0569), 'ZAF': (28.0473, -26.2041), 'KOR': (126.9780, 37.5665),
    'ESP': (-3.7038, 40.4168), 'LKA': (79.8612, 6.9271), 'SWE': (18.0686, 59.3293),
    'CHE': (7.4474, 46.9480), 'TWN': (121.5654, 25.0330), 'TJK': (68.7870, 38.5598),
    'TZA': (35.7382, -6.3690), 'THA': (100.5018, 13.7563), 'TUR': (32.8597, 39.9334),
    'TKM': (58.3794, 37.9601), 'UGA': (32.5825, 0.3476), 'UKR': (30.5234, 50.4501),
    'ARE': (54.3773, 24.2992), 'GBR': (-0.1278, 51.5074), 'USA': (-77.0369, 38.9072),
    'URY': (-56.1645, -34.9011), 'UZB': (69.2401, 41.2995), 'VEN': (-66.9036, 10.4806),
    'VNM': (105.8342, 21.0285), 'YEM': (44.2067, 15.3694), 'ZMB': (28.2833, -15.4167),
    'ZWE': (31.0335, -17.8252)
}

def preprocess_geojson_safely(geojson_path):
    """Load and preprocess GeoJSON, skipping invalid geometries and removing Antarctica."""
    with open(geojson_path, 'r') as file:
        raw_data = json.load(file)
    
    processed_features = []
    for feature in raw_data["features"]:
        try:
            geom = shape(feature["geometry"])
            # Skip Antarctica using both possible ISO codes
            if feature["properties"].get("iso_a3") == "ATA" or feature["properties"].get("iso") == "ATA":
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_valid and not geom.is_empty:
                processed_features.append({
                    "geometry": geom,
                    "ISO_A3": feature["properties"].get("iso_a3", feature["properties"].get("iso", None)),
                    "name": feature["properties"].get("name", None)
                })
        except Exception as e:
            print(f"Skipping problematic feature: {e}")
    
    return gpd.GeoDataFrame(processed_features, crs="EPSG:4326")

def load_world_map():
    """Load the world map data using geopandas built-in data or from local files."""
    # Try to find local geospatial files first
    potential_paths = [
        '../../data/world_by_iso_geo.json',
        '../../data/world_by_iso_topo.json',
        'input/world_by_iso_geo.json',
        'input/world_by_iso_topo.json'
    ]
    
    for path in potential_paths:
        if Path(path).exists():
            world = preprocess_geojson_safely(path)
            print(f"Loaded world map with {len(world)} features")
            return world
    
    # Fallback to geopandas built-in world data
    try:
        import geopandas as gpd
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        print(f"Using geopandas built-in world map with {len(world)} features")
        return world
    except Exception as e:
        raise FileNotFoundError(f"World map data not found locally and geopandas fallback failed: {e}")

def load_delivered_cost_data(delivered_cost_path='figures/delivered_costs/combined_delivered_costs.csv'):
    """
    Load the combined delivered cost data (PEM LCOH + transport cost).
    """
    try:
        if not Path(delivered_cost_path).exists():
            raise FileNotFoundError(f"Delivered cost data not found at {delivered_cost_path}")
        
        df = pd.read_csv(delivered_cost_path)
        
        # Use only the columns we need and avoid duplicates
        # Keep ISO_A3 as origin_country and use total_delivered_cost as total_cost
        df_clean = df[['ISO_A3', 'total_delivered_cost', 'year', 'scenario', 'carrier', 'transport_mode']].copy()
        
        # Rename columns
        df_clean = df_clean.rename(columns={
            'ISO_A3': 'origin_country',
            'total_delivered_cost': 'total_cost'
        })
        
        # Convert data types
        df_clean['year'] = df_clean['year'].astype(str)
        df_clean['origin_country'] = df_clean['origin_country'].astype(str)
        df_clean['total_cost'] = pd.to_numeric(df_clean['total_cost'], errors='coerce')
        
        # Remove any rows with NaN values
        df_clean = df_clean.dropna()
        
        print(f"Loaded delivered cost data: {len(df_clean)} routes")
        print(f"Years: {sorted(df_clean['year'].unique())}")
        print(f"Scenarios: {sorted(df_clean['scenario'].unique())}")
        print(f"Final columns: {df_clean.columns.tolist()}")
        
        return df_clean
    
    except Exception as e:
        print(f"Error loading delivered cost data: {e}")
        raise

def create_delivered_cost_route_maps(results_df, output_dir):
    """
    Create maps showing the best delivered cost routes to Switzerland.
    This function is adapted from the original create_best_route_maps but uses delivered costs.
    Matches the exact style of the original transport maps.
    """
    # Set plot style for scientific publication (exactly as in original)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'figure.figsize': (8, 4.5),
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.5,
        'lines.linewidth': 0.5,
    })
    
    # Create maps directory
    maps_dir = output_dir / 'maps'
    maps_dir.mkdir(exist_ok=True)
    
    # Load world map data
    world = load_world_map()
    
    # Coordinates for Basel, Switzerland (target city)
    basel_lon, basel_lat = 7.5886, 47.5596
    
    # Coordinates for Antwerp port (replaces Rotterdam)
    antwerp_lon, antwerp_lat = 4.4024, 51.2194
    
    # List all possible markers for different combinations
    all_markers = ['o', '^', 's', 'D', 'h', 'p', '*', '+', 'x', 'v', '<', '>', '|', '_']
    
    # Calculate global min/max for consistent color scaling across all figures
    global_min = results_df['total_cost'].min()
    global_max = results_df['total_cost'].max()
    
    # Add some padding to the global range
    cost_range = global_max - global_min
    global_vmin = max(0, global_min - 0.05 * cost_range)
    global_vmax = global_max + 0.05 * cost_range
    
    print(f"Using consistent color scale: {global_vmin:.4f} - {global_vmax:.4f} EUR/kWh")
    print(f"Equivalent to: {global_vmin * 1.1 * 33.33:.1f} - {global_vmax * 1.1 * 33.33:.1f} USD/kg H2")
    
    # Group results by scenario, year
    for scenario in results_df['scenario'].unique():
        for year in results_df['year'].unique():
            # Get best (lowest delivered cost) options for this scenario and year
            scenario_data = results_df[(results_df['scenario'] == scenario) & 
                                      (results_df['year'] == year)]
            
            # Remove any rows with NaN values in critical columns
            scenario_data = scenario_data.dropna(subset=['origin_country', 'total_cost'])
            
            # Skip if no data available for this scenario/year
            if scenario_data.empty:
                print(f"No data available for scenario: {scenario}, year: {year}")
                continue
            
            # Get the minimum delivered cost option for each country
            min_indices = scenario_data.groupby('origin_country')['total_cost'].idxmin()
            min_indices = min_indices.dropna()
            best_options = scenario_data.loc[min_indices]
            
            # Verify that the routes follow proper logic
            print(f"Scenario: {scenario}, Year: {year}")
            print(f"Found {len(best_options)} best routes")
            print(f"Transport modes: {best_options['transport_mode'].unique()}")
            print(f"Carriers: {best_options['carrier'].unique()}")
            
            # Find unique carrier-transport combinations present in the data
            unique_combinations = set([(row['carrier'], row['transport_mode']) 
                                       for _, row in best_options.iterrows()])
            
            print(f"Unique carrier-transport combinations: {unique_combinations}")
            
            # Create mapping for marker shapes based on the actual combinations used
            carrier_transport_markers = {}
            for i, combo in enumerate(sorted(unique_combinations)):
                marker_idx = i % len(all_markers)
                carrier_transport_markers[combo] = all_markers[marker_idx]
            
            # Create figure with white background (exact style as original)
            fig = plt.figure(figsize=(8, 4.5), facecolor='white')
            ax_main = fig.add_subplot(111)
            
            # Plot world map
            world.plot(ax=ax_main, color='lightgrey', edgecolor='white', linewidth=0.2)
            
            # Custom colormap - blue for low costs, green for high costs (same as original)
            cmap = plt.cm.YlGnBu  # Blue for low costs, green/yellow for high costs
            
            # Use global color scale for consistency across all figures
            norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
            
            # Plot Switzerland - use colored polygon (exactly as original)
            switzerland = world[world['ISO_A3'] == 'CHE']
            if not switzerland.empty:
                switzerland.plot(
                    ax=ax_main,
                    color='#E63946',  # Use a distinct red color
                    edgecolor='black',
                    linewidth=0.3,
                    zorder=10
                )
            else:
                print("Warning: Switzerland not found in world data")
            
            # Label Basel as target city
            ax_main.annotate(
                'Basel',
                xy=(basel_lon, basel_lat),
                xytext=(0, 0),
                textcoords='offset points',
                fontsize=8,
                color='white',
                weight='bold',
                ha='center',
                va='center',
                zorder=11
            )
            
            # Define European countries for inset map
            european_countries = {'AUT', 'BEL', 'BGR', 'HRV', 'CZE', 'DNK', 'EST', 
                                'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 
                                'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 
                                'SVN', 'ESP', 'SWE', 'GBR', 'NOR'}
            
            # Plot points and routes for each country
            for idx, row in best_options.iterrows():
                country_code = row['origin_country']
                if country_code in CAPITAL_COORDINATES:
                    lon, lat = CAPITAL_COORDINATES[country_code]
                    mode = row['transport_mode']
                    carrier = row['carrier']
                    
                    # Get marker for this carrier-transport combination
                    combo_marker = carrier_transport_markers.get((carrier, mode), 'o')
                    
                    # Plot point with marker and color based on cost
                    ax_main.scatter(
                        lon, lat,
                        color=cmap(norm(row['total_cost'])),  # Use colormap for cost
                        marker=combo_marker,
                        s=20, 
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.9,
                        zorder=4
                    )
                    
                    # Draw route lines based on transport type - use thin gray lines (as original)
                    # For delivered cost maps, we'll use simplified routing: 
                    # - Direct land routes for pipeline/truck/rail
                    # - Maritime routes via Antwerp for ship
                    if mode in ['pipeline', 'truck', 'rail']:
                        # Direct land route to Basel
                        ax_main.plot(
                            [lon, basel_lon], 
                            [lat, basel_lat],
                            color='#aaaaaa',  # Light gray
                            linewidth=0.3,
                            linestyle='-',
                            alpha=0.5,
                            zorder=2
                        )
                    else:  # ship - maritime route
                        # Route from origin to Antwerp port
                        ax_main.plot(
                            [lon, antwerp_lon], 
                            [lat, antwerp_lat],
                            color='#aaaaaa',  # Light gray
                            linewidth=0.3,
                            linestyle='-',
                            alpha=0.5,
                            zorder=2
                        )
                        
                        # Route from Antwerp port to Basel
                        ax_main.plot(
                            [antwerp_lon, basel_lon],
                            [antwerp_lat, basel_lat],
                            color='#aaaaaa',  # Light gray
                            linewidth=0.3,
                            linestyle='-',
                            alpha=0.5,
                            zorder=2
                        )
            
            # Add Antwerp port to the map
            ax_main.scatter(
                antwerp_lon, 
                antwerp_lat,
                marker='s',
                s=20,
                color='black',
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8,
                zorder=5
            )
            
            # Add colorbar for cost (exactly as original with dual units)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_main, orientation='vertical', pad=0.05, fraction=0.02)
            
            # Remove default label and add custom labels on both sides
            cbar.ax.tick_params(size=3, width=0.5, labelsize=8)
            
            # Add EUR/kWh label on the left side (data is already in EUR/kWh)
            cbar.ax.text(-3, 0.5, 'EUR/kWh', transform=cbar.ax.transAxes, 
                        rotation=90, va='center', ha='center', fontsize=9, fontweight='bold')
            
            # Add EUR/kg H2 values on the right side of the colorbar
            # Conversion factor: 1 kg H2 = 33.33 kWh (LHV)
            h2_to_kwh_factor = 33.33
            
            # Get the tick values from the left side (EUR/kWh)
            tick_values_kwh = cbar.get_ticks()
            tick_values_kg = tick_values_kwh * h2_to_kwh_factor
            
            # Create secondary y-axis for EUR/kg H2
            cbar_right = cbar.ax.twinx()
            cbar_right.set_ylim(cbar.ax.get_ylim())
            cbar_right.set_yticks(tick_values_kwh)
            cbar_right.set_yticklabels([f'{val:.1f}' for val in tick_values_kg])
            cbar_right.set_ylabel('EUR/kg H$_2$', fontsize=9, fontweight='bold')
            cbar_right.tick_params(size=3, width=0.5, labelsize=8)
            
            # Create legend for carrier-transport combinations
            legend_elements = []
            
            # Create legend elements for each used combination
            for (carrier, mode), marker in sorted(carrier_transport_markers.items()):
                legend_elements.append(
                    Line2D([0], [0], 
                           marker=marker, 
                           color='w',
                           markerfacecolor='gray',
                           markeredgecolor='black',
                           markersize=5, 
                           label=f"{carrier}-{mode.capitalize()}")
                )
            
            # Add combined legend - moved further to the right (as original)
            combined_legend = ax_main.legend(
                handles=legend_elements, 
                title='Carrier-Transport Mode',
                bbox_to_anchor=(1.25, 1.0),  # Move legend further right
                loc='upper left',
                frameon=True,
                framealpha=0.9,
                fontsize=5,
                ncol=2,  # Use 2 columns for more compact legend
                handletextpad=0.5,
                columnspacing=1.0
            )
            
            combined_legend.get_title().set_fontweight('bold')
            combined_legend.get_title().set_fontsize(8)
            
            # Create inset for Europe - moved to avoid overlap with legend (adjusted width)
            inset_ax = fig.add_axes([0.05, 0.05, 0.3, 0.3], frame_on=True)
            inset_ax.set_facecolor('white')
            
            # Add frame to the inset map
            for spine in inset_ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)
            
            # Plot all countries but set the view bounds to Europe
            world.plot(ax=inset_ax, color='lightgrey', edgecolor='white', linewidth=0.2)
            
            # Add Switzerland to European inset - colored polygon
            if not switzerland.empty:
                switzerland.plot(
                    ax=inset_ax,
                    color='#E63946',  # Same distinct red color
                    edgecolor='black',
                    linewidth=0.3,
                    zorder=5
                )
            else:
                print("Warning: Switzerland not found for inset map")
            
            # Plot Antwerp port on inset
            inset_ax.scatter(
                antwerp_lon,
                antwerp_lat,
                marker='s',
                s=20,
                color='black',
                edgecolor='black',
                linewidth=0.5,
                zorder=5
            )
            
            # Plot European countries on inset map
            for idx, row in best_options.iterrows():
                country_code = row['origin_country']
                if country_code in CAPITAL_COORDINATES and country_code in european_countries:
                    lon, lat = CAPITAL_COORDINATES[country_code]
                    mode = row['transport_mode']
                    carrier = row['carrier']
                    
                    # Get marker for this carrier-transport combination
                    combo_marker = carrier_transport_markers.get((carrier, mode), 'o')
                    
                    # Plot point with appropriate marker and color based on cost
                    inset_ax.scatter(
                        lon, lat,
                        color=cmap(norm(row['total_cost'])),  # Use colormap for cost
                        marker=combo_marker,
                        s=20,  # Reduced marker size from 60
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.9,
                        zorder=4
                    )
                    
                    # Draw route lines on inset map
                    if mode in ['pipeline', 'truck', 'rail']:
                        # Direct land route to Basel
                        inset_ax.plot(
                            [lon, basel_lon],
                            [lat, basel_lat],
                            color='#aaaaaa',  # Light gray
                            linewidth=0.3,
                            linestyle='-',
                            alpha=0.5,
                            zorder=2
                        )
                    else:  # ship - maritime route
                        # Route from origin to Antwerp
                        inset_ax.plot(
                            [lon, antwerp_lon],
                            [lat, antwerp_lat],
                            color='#aaaaaa',  # Light gray
                            linewidth=0.3,
                            linestyle='-',
                            alpha=0.5,
                            zorder=2
                        )
                        
                        # Route from Antwerp to Basel
                        inset_ax.plot(
                            [antwerp_lon, basel_lon],
                            [antwerp_lat, basel_lat],
                            color='#aaaaaa',  # Light gray
                            linewidth=0.3,
                            linestyle='-',
                            alpha=0.5,
                            zorder=2
                        )
            
            # Set European bounds for inset map (exactly as original)
            inset_ax.set_xlim(-10, 20)  # Longitude range for Europe
            inset_ax.set_ylim(35, 65)   # Latitude range for Europe
            inset_ax.set_aspect('equal')  # Maintain proper aspect ratio for Europe
            inset_ax.axis('off')  # Hide axis
            
            # Title and axis settings - use more concise scientific style (as original)
            scenario_title = scenario.replace("_", " ").title()
            # Use 2024 in title instead of data year
            display_year = year
            ax_main.set_title(f'Hydrogen Delivered Cost Routes to Basel ({display_year})\n{scenario_title}',
                              fontsize=11, fontweight='bold')
            ax_main.set_xlim(-180, 180)
            ax_main.set_ylim(-60, 80)  # This cuts off Antarctica
            ax_main.set_aspect('equal')  # Maintain proper aspect ratio
            ax_main.axis('off')
            
            # Add caption-like text at the bottom of the figure (as original)
            fig.text(0.5, 0.01, 
                     "Color represents total delivered cost (production + transport). Markers indicate carrier-transport mode combinations.",
                     ha='center', fontsize=8, style='italic')
            
            # Save figure (using exact same naming as original)
            filename = f'delivered_cost_routes_{scenario}_{year}.png'
            plt.savefig(
                maps_dir / filename,
                dpi=600,
                bbox_inches='tight'
            )
            plt.close()
            
            print(f"Saved: {filename}")
    
    print(f"All delivered cost route maps saved to {maps_dir}")

def main():
    """Main function to create delivered cost route maps."""
    
    print("Creating Delivered Cost Route Maps...")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load delivered cost data
        print("1. Loading delivered cost data...")
        delivered_cost_path = 'figures/delivered_costs/combined_delivered_costs.csv'
        results_df = load_delivered_cost_data(delivered_cost_path)
        
        # Debug: Check data quality
        print(f"   Total rows loaded: {len(results_df)}")
        print(f"   Unique scenarios: {results_df['scenario'].unique()}")
        print(f"   Unique years: {results_df['year'].unique()}")
        print(f"   Rows with NaN origin_country: {results_df['origin_country'].isna().sum()}")
        print(f"   Rows with NaN total_cost: {results_df['total_cost'].isna().sum()}")
        
        # Create delivered cost route maps
        print("\n2. Creating delivered cost route maps...")
        create_delivered_cost_route_maps(results_df, output_dir)
        
        print(f"\n{'='*50}")
        print("Maps saved to output/maps/")
        print("Generated files:")
        
        maps_dir = output_dir / 'maps'
        if maps_dir.exists():
            for file in sorted(maps_dir.glob('delivered_cost_routes_*.png')):
                print(f"  - {file.name}")
        
    except Exception as e:
        print(f"Error creating delivered cost route maps: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()