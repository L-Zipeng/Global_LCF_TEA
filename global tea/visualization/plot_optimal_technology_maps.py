"""
This script generates global maps showing competitive production technologies 
for each fuel category (Hydrogen, SAF, Diesel, Other Fuels) in every country.

The aim is to identify:
1. Best production locations for each fuel category
2. Which technologies are competitive (within cost margin) in which regions
3. Geographic patterns in technology competitiveness
4. Countries with multiple viable technology options

Key features:
- Configurable competitiveness margin (default: 10%)
- Visual distinction between single and multiple competitive technologies
- Hatching patterns for countries with multiple competitive options
- Comprehensive analysis of technology competitiveness
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
from pathlib import Path
import json
import seaborn as sns

def set_plot_style():
    """Set the style for publication-quality plots"""
    plt.style.use('default')
    
    # High-impact journal style elements
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['figure.titlesize'] = 18
    
    # Professional looking elements
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['patch.linewidth'] = 0.8
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '#333333'
    
    # Clean background
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['figure.facecolor'] = 'white'
    
    # Better margins
    mpl.rcParams['figure.constrained_layout.use'] = True

def preprocess_geojson_safely(geojson_path):
    """Load and preprocess GeoJSON, skipping invalid geometries and removing Antarctica"""
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
                geom = geom.buffer(0)
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

def load_lcox_data(excel_file):
    """Load LCOX data from Excel file"""
    print(f"Loading LCOX data from {excel_file}")
    xlsx = pd.ExcelFile(excel_file)
    
    # Get LCOX sheets
    lcox_sheets = [s for s in xlsx.sheet_names if s.startswith('LCOX_')]
    print(f"Found LCOX sheets: {lcox_sheets}")
    
    lcox_data = {}
    for sheet in lcox_sheets:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        # Set the first column as index (country codes)
        if 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
        lcox_data[sheet] = df.apply(pd.to_numeric, errors='coerce')
        print(f"Loaded {sheet}: {df.shape}")
    
    return lcox_data

def define_technology_categories():
    """Define technology categories as per aggregateplot.py"""
    categories = {
        'Hydrogen': {
            'Green H$_2$': ['PEM', 'AE', 'SOEC'],
            'Pink H$_2$': ['HTSE', 'CuCl'],
            'Blue H$_2$': ['SMR_CCS', 'ATR_CCS', 'CLR'],
            'Turquoise H$_2$': ['M_PYR'],
            'Biogenic H$_2$': ['TG_CCS']
        },
        'SAF': {
            'Solar SAF': ['SR_FT_kerosene', 'ST_FT_kerosene'],
            'Biogenic SAF': ['TG_FT_kerosene', 'HTL', 'B_PYR_kerosene'],
            'Power-to-Liquid SAF': ['RWGS_FT_kerosene', 'RWGS_MeOH_kerosene']
        },
        'Diesel': {
            'Solar Diesel': ['SR_FT_diesel', 'ST_FT_diesel'],
            'Biogenic Diesel': ['TG_FT_diesel', 'HVO_diesel', 'FAME'],
            'Power-to-Liquid Diesel': ['RWGS_FT_diesel']
        },
        'Other Fuels': {
            'Ammonia': ['HB'],
            'Methanol': ['RWGS_MeOH_methanol'],
            'DME': ['RWGS_MeOH_DME'],
            'Methane': ['PTM', 'AD']
        }
    }
    
    return categories

def define_color_palettes():
    """Define distinct color palettes for each fuel category"""
    palettes = {
        'Hydrogen': {
            'Green H$_2$': '#2E8B57',      # Sea Green
            'Pink H$_2$': '#FF69B4',       # Hot Pink
            'Blue H$_2$': '#4169E1',       # Royal Blue
            'Turquoise H$_2$': '#40E0D0',  # Turquoise
            'Biogenic H$_2$': '#8B4513'    # Saddle Brown
        },
        'SAF': {
            'Solar SAF': '#FFD700',          # Gold
            'Biogenic SAF': '#228B22',       # Forest Green
            'Power-to-Liquid SAF': '#9370DB' # Medium Purple
        },
        'Diesel': {
            'Solar Diesel': '#FF8C00',       # Dark Orange
            'Biogenic Diesel': '#32CD32',    # Lime Green
            'Power-to-Liquid Diesel': '#8A2BE2' # Blue Violet
        },
        'Other Fuels': {
            'Ammonia': '#FF6347',       # Tomato
            'Methanol': '#DA70D6',      # Orchid
            'DME': '#20B2AA',           # Light Sea Green
            'Methane': '#87CEEB'        # Sky Blue
        }
    }
    
    return palettes

def find_cheapest_technology_per_category(lcox_data, categories, competitiveness_margins=None):
    """Find all competitive technologies within margin of cheapest for each country
    
    Args:
        lcox_data: Dictionary of LCOX data by scenario
        categories: Technology categories definition
        competitiveness_margins: Dictionary of margins per fuel category (e.g., {'Hydrogen': 0.20, 'SAF': 0.50})
                                or single float for uniform margin across all categories
    """
    # Handle backward compatibility - if single margin provided, apply to all categories
    if competitiveness_margins is None:
        competitiveness_margins = 0.10
    
    if isinstance(competitiveness_margins, (int, float)):
        # Convert single margin to dictionary for all categories
        uniform_margin = competitiveness_margins
        competitiveness_margins = {category: uniform_margin for category in categories.keys()}
    results = {}
    
    for scenario, data in lcox_data.items():
        results[scenario] = {}
        
        for fuel_category, tech_groups in categories.items():
            # Get the margin for this fuel category
            category_margin = competitiveness_margins.get(fuel_category, 0.10)
            
            # Get all technologies in this category
            all_techs = []
            for group_techs in tech_groups.values():
                all_techs.extend(group_techs)
            
            # Filter to only available technologies
            available_techs = [tech for tech in all_techs if tech in data.columns]
            
            # CLR technology constraint: not available before 2030
            if 'CLR' in available_techs:
                if scenario in ['LCOX_Base24', 'LCOX_Base30', 'LCOX_2deg30', 'LCOX_15deg30']:
                    available_techs.remove('CLR')
                    print(f"  Excluded CLR from {scenario} (not commercially available before 2030)")
            
            if not available_techs:
                print(f"Warning: No technologies found for {fuel_category} in {scenario}")
                continue
            
            # Get cost data for these technologies
            category_data = data[available_techs]
            
            # Find the cheapest technology and cost for each country
            cheapest_cost = category_data.min(axis=1)
            cheapest_tech = category_data.idxmin(axis=1)
            
            # Find all competitive technologies within margin
            competitive_techs = {}
            competitive_groups = {}
            
            # Map technology to its group
            tech_to_group = {}
            for group_name, techs in tech_groups.items():
                for tech in techs:
                    tech_to_group[tech] = group_name
            
            for country in category_data.index:
                if pd.isna(cheapest_cost[country]):
                    continue
                    
                min_cost = cheapest_cost[country]
                threshold = min_cost * (1 + category_margin)
                
                # Find all technologies within threshold
                country_costs = category_data.loc[country]
                competitive_tech_list = country_costs[country_costs <= threshold].sort_values().index.tolist()
                
                # Map to groups
                competitive_group_list = [tech_to_group[tech] for tech in competitive_tech_list if tech in tech_to_group]
                # Remove duplicates while preserving order
                competitive_group_list = list(dict.fromkeys(competitive_group_list))
                
                competitive_techs[country] = competitive_tech_list
                competitive_groups[country] = competitive_group_list
            
            # Create results dataframe
            country_results = pd.DataFrame({
                'cheapest_technology': cheapest_tech,
                'cheapest_cost': cheapest_cost,
                'technology_group': cheapest_tech.map(tech_to_group),
                'competitive_technologies': pd.Series(competitive_techs),
                'competitive_groups': pd.Series(competitive_groups),
                'num_competitive': pd.Series({k: len(v) for k, v in competitive_groups.items()})
            })
            
            # Debug output for hydrogen
            if fuel_category == 'Hydrogen':
                print(f"Debug {scenario} - Hydrogen competitive analysis (margin: {category_margin*100}%):")
                multi_tech_countries = country_results[country_results['num_competitive'] > 1]
                print(f"  Countries with multiple competitive technologies: {len(multi_tech_countries)}")
                if len(multi_tech_countries) > 0:
                    print(f"  Example: {multi_tech_countries.iloc[0].name} has {multi_tech_countries.iloc[0]['competitive_groups']}")
            
            results[scenario][fuel_category] = country_results
    
    return results

def create_optimal_technology_maps(world_gdf, optimal_tech_data, color_palettes, output_dir, competitiveness_margins=None):
    """Create maps showing optimal technology and competitive alternatives for each fuel category
    
    Args:
        world_gdf: World geographic data
        optimal_tech_data: Technology optimization results
        color_palettes: Color schemes for each fuel category
        output_dir: Output directory for figures
        competitiveness_margins: Dictionary of margins per fuel category or single margin for all
    """
    # Handle backward compatibility
    if competitiveness_margins is None:
        competitiveness_margins = 0.10
    
    if isinstance(competitiveness_margins, (int, float)):
        uniform_margin = competitiveness_margins
        # We'll determine the format for display purposes
        margin_display = f"±{uniform_margin*100:.0f}%"
    else:
        # Format display for multiple margins
        margin_parts = []
        for category, margin in competitiveness_margins.items():
            margin_parts.append(f"{category}: ±{margin*100:.0f}%")
        margin_display = " | ".join(margin_parts)
    
    # Define hatching patterns for secondary technologies
    hatch_patterns = ['///', '\\\\\\', '|||', '---', '+++', 'xxx', '...', 'ooo']
    
    # Scenario mapping for better labels
    scenario_names = {
        'LCOX_Base24': 'Ref 2024',
        'LCOX_Base30': 'BAU 2030',
        'LCOX_2deg30': '2°C 2030',
        'LCOX_15deg30': '1.5°C 2030',
        'LCOX_Base50': 'BAU 2050',
        'LCOX_2deg50': '2°C 2050',
        'LCOX_15deg50': '1.5°C 2050'
    }
    
    # Create maps for each fuel category
    for fuel_category in optimal_tech_data[list(optimal_tech_data.keys())[0]].keys():
        print(f"Creating competitive technology maps for {fuel_category}")
        
        # Get unique technology groups across all scenarios
        all_groups = set()
        for scenario_data in optimal_tech_data.values():
            if fuel_category in scenario_data:
                groups = scenario_data[fuel_category]['technology_group'].dropna().unique()
                all_groups.update(groups)
        
        # Get colors for this category
        category_colors = color_palettes[fuel_category]
        
        # Create figure with subplots for different scenarios
        # Use a 2x4 grid but ensure proper aspect ratio management
        fig, axes = plt.subplots(2, 4, figsize=(28, 14))
        axes = axes.flatten()
        
        # Plot each scenario
        for idx, (scenario, scenario_data) in enumerate(optimal_tech_data.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            if fuel_category not in scenario_data:
                ax.set_title(f"{scenario_names.get(scenario, scenario)}\nNo data", fontsize=14)
                ax.axis('off')
                continue
            
            # Merge with world data
            plot_data = world_gdf.copy()
            country_data = scenario_data[fuel_category].reset_index()
            country_data.columns = ['ISO_A3'] + list(country_data.columns[1:])
            plot_data = plot_data.merge(
                country_data, 
                on='ISO_A3', 
                how='left'
            )
            
            # Plot base map (countries with no data)
            plot_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
            
            # Plot countries with single competitive technology (solid colors)
            single_tech_data = plot_data[plot_data['num_competitive'] == 1]
            for group in all_groups:
                if group in category_colors:
                    group_data = single_tech_data[single_tech_data['technology_group'] == group]
                    if not group_data.empty:
                        group_data.plot(
                            ax=ax, 
                            color=category_colors[group], 
                            edgecolor='white', 
                            linewidth=0.5
                        )
            
            # Plot countries with multiple competitive technologies (base color + hatching)
            multi_tech_data = plot_data[plot_data['num_competitive'] > 1]
            
            for _, country in multi_tech_data.iterrows():
                # Get competitive_groups safely
                try:
                    competitive_groups = country['competitive_groups']
                    
                    # Handle various NaN representations
                    if competitive_groups is None:
                        continue
                    if isinstance(competitive_groups, float) and pd.isna(competitive_groups):
                        continue
                    if not competitive_groups or len(competitive_groups) == 0:
                        continue
                        
                except (TypeError, ValueError):
                    # Skip if we can't process this entry
                    continue
                    
                primary_group = country['technology_group']  # Cheapest technology
                
                # Plot base color (primary/cheapest technology)
                if primary_group in category_colors:
                    country_geom = gpd.GeoDataFrame([country], crs=plot_data.crs)
                    country_geom.plot(
                        ax=ax,
                        color=category_colors[primary_group],
                        edgecolor='white',
                        linewidth=0.5
                    )
                
                # Add hatching for secondary competitive technologies
                secondary_groups = [g for g in competitive_groups if g != primary_group]
                for i, secondary_group in enumerate(secondary_groups):
                    if secondary_group in category_colors and i < len(hatch_patterns):
                        country_geom.plot(
                            ax=ax,
                            color='none',
                            edgecolor=category_colors[secondary_group],
                            linewidth=1.5,
                            hatch=hatch_patterns[i],
                            alpha=0.8
                        )
            
            # Style the map - ensure consistent aspect ratio for all subplots
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 85)
            ax.set_aspect('equal')  # This ensures proper aspect ratio
            ax.axis('off')
            ax.set_title(f"{scenario_names.get(scenario, scenario)}", fontsize=14, fontweight='bold')
        
        # Remove empty subplots properly to maintain grid structure
        for idx in range(len(optimal_tech_data), len(axes)):
            axes[idx].set_visible(False)  # Hide instead of deleting to maintain layout
        
        # Create comprehensive legend
        legend_elements = []
        
        # Solid colors for ALL defined technologies (including non-competitive ones)
        for group, color in category_colors.items():
            if group in all_groups:
                # Technology appears in data and is competitive
                legend_elements.append(
                    mpatches.Patch(color=color, label=f'{group} (primary)')
                )
            else:
                # Technology is defined but not competitive - show with faded color
                legend_elements.append(
                    mpatches.Patch(color=color, alpha=0.3, label=f'{group} (not competitive)')
                )
        
        # Hatching patterns for secondary competitive technologies
        legend_elements.append(
            mpatches.Patch(color='white', label='─────────────')  # Separator
        )
        
        for i, pattern in enumerate(hatch_patterns[:3]):  # Show first few patterns as examples
            legend_elements.append(
                mpatches.Patch(
                    facecolor='lightblue', 
                    edgecolor='black',
                    hatch=pattern,
                    alpha=0.7,
                    label=f'Secondary tech {i+1}'
                )
            )
        
        legend_elements.append(
            mpatches.Patch(color='lightgray', label='No data')
        )
        
        # Add legend to the right side
        fig.legend(handles=legend_elements, 
                  loc='center right', 
                  bbox_to_anchor=(0.98, 0.5),
                  fontsize=11,
                  frameon=True,
                  fancybox=True,
                  shadow=True)
        
        # Add main title with margin information
        fig.suptitle(f'Competitive {fuel_category} Production Technologies by Country\n' + 
                    f'({margin_display} cost margin | Hatching = secondary competitive options)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Adjust layout to accommodate legend
        plt.subplots_adjust(left=0.02, right=0.85, top=0.88, bottom=0.05, wspace=0.05, hspace=0.15)
        
        # Save figure
        filename = f'competitive_{fuel_category.lower().replace(" ", "_")}_technologies_map.png'
        plt.savefig(output_dir / filename, 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none')
        
        print(f"Saved: {filename}")
        plt.close()
        
        # Create a summary plot showing competitiveness statistics
        create_competitiveness_summary(optimal_tech_data, fuel_category, competitiveness_margins, output_dir)

def create_cost_distribution_analysis(optimal_tech_data, color_palettes, output_dir):
    """Create additional analysis plots for cost distributions and technology dominance"""
    
    # Technology dominance analysis
    dominance_data = {}
    
    for scenario, scenario_data in optimal_tech_data.items():
        dominance_data[scenario] = {}
        
        for fuel_category, data in scenario_data.items():
            # Count technology groups
            group_counts = data['technology_group'].value_counts()
            dominance_data[scenario][fuel_category] = group_counts
    
    # Create dominance visualization for each fuel category
    for fuel_category in optimal_tech_data[list(optimal_tech_data.keys())[0]].keys():
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        scenario_names = {
            'LCOX_Base24': 'Ref. 2022',
            'LCOX_Base30': 'BAU 2030',
            'LCOX_2deg30': '2°C 2030',
            'LCOX_15deg30': '1.5°C 2030',
            'LCOX_Base50': 'BAU 2050',
            'LCOX_2deg50': '2°C 2050',
            'LCOX_15deg50': '1.5°C 2050'
        }
        
        for idx, scenario in enumerate(optimal_tech_data.keys()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            if fuel_category in dominance_data[scenario]:
                data = dominance_data[scenario][fuel_category]
                
                # Create bar chart with consistent category colors
                if not data.empty:
                    # Get colors based on technology groups using the category palette
                    category_palette = color_palettes[fuel_category]
                    colors = [category_palette.get(group, '#cccccc') for group in data.index]
                    bars = ax.bar(range(len(data)), data.values, color=colors)
                    
                    # Customize the chart
                    ax.set_xticks(range(len(data)))
                    ax.set_xticklabels(data.index, rotation=45, ha='right')
                    ax.set_ylabel('Number of Countries')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, data.values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{value}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f"{scenario_names.get(scenario, scenario)}", fontsize=12, fontweight='bold')
        
        # Remove empty subplots
        for idx in range(len(optimal_tech_data), len(axes)):
            fig.delaxes(axes[idx])
        
        # Add main title
        fig.suptitle(f'{fuel_category} Technology Dominance by Scenario\n(Number of Countries per Technology Group)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{fuel_category.lower().replace(" ", "_")}_technology_dominance.png'
        plt.savefig(output_dir / filename, 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white')
        
        print(f"Saved: {filename}")
        plt.close()

def create_summary_statistics(optimal_tech_data, output_dir):
    """Create summary statistics and export to Excel"""
    
    summary_data = {}
    
    for scenario, scenario_data in optimal_tech_data.items():
        summary_data[scenario] = {}
        
        for fuel_category, data in scenario_data.items():
            summary_stats = {
                'Total_Countries': len(data.dropna()),
                'Average_Cost': data['cheapest_cost'].mean(),
                'Median_Cost': data['cheapest_cost'].median(),
                'Min_Cost': data['cheapest_cost'].min(),
                'Max_Cost': data['cheapest_cost'].max(),
                'Cost_StdDev': data['cheapest_cost'].std()
            }
            
            # Technology group distribution
            tech_distribution = data['technology_group'].value_counts()
            for tech, count in tech_distribution.items():
                summary_stats[f'{tech}_Count'] = count
                summary_stats[f'{tech}_Percentage'] = (count / len(data.dropna())) * 100
            
            summary_data[scenario][fuel_category] = summary_stats
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame.from_dict({
        (scenario, fuel): stats 
        for scenario, fuel_data in summary_data.items() 
        for fuel, stats in fuel_data.items()
    }, orient='index')
    
    summary_df.to_excel(output_dir / 'optimal_technology_summary_statistics.xlsx')
    print("Saved: optimal_technology_summary_statistics.xlsx")

def create_competitiveness_summary(optimal_tech_data, fuel_category, competitiveness_margins, output_dir):
    """Create summary visualization of technology competitiveness"""
    
    # Extract the margin for this specific fuel category
    if isinstance(competitiveness_margins, dict):
        category_margin = competitiveness_margins.get(fuel_category, 0.10)
    else:
        category_margin = competitiveness_margins
    
    # Collect competitiveness data across scenarios
    competitiveness_data = []
    scenario_names = {
        'LCOX_Base24': 'Ref 2024',
        'LCOX_Base30': 'BAU 2030',
        'LCOX_2deg30': '2°C 2030',
        'LCOX_15deg30': '1.5°C 2030',
        'LCOX_Base50': 'BAU 2050',
        'LCOX_2deg50': '2°C 2050',
        'LCOX_15deg50': '1.5°C 2050'
    }
    
    for scenario, scenario_data in optimal_tech_data.items():
        if fuel_category not in scenario_data:
            continue
            
        data = scenario_data[fuel_category]
        scenario_name = scenario_names.get(scenario, scenario)
        
        # Count countries by number of competitive technologies
        competitiveness_counts = data['num_competitive'].value_counts().sort_index()
        
        for num_tech, count in competitiveness_counts.items():
            competitiveness_data.append({
                'Scenario': scenario_name,
                'Competitive_Technologies': f'{num_tech} tech{"s" if num_tech > 1 else ""}',
                'Number_of_Countries': count,
                'Percentage': (count / len(data.dropna())) * 100
            })
    
    if not competitiveness_data:
        return
    
    # Create visualization
    comp_df = pd.DataFrame(competitiveness_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart of competitiveness distribution
    pivot_df = comp_df.pivot(index='Scenario', columns='Competitive_Technologies', values='Number_of_Countries')
    pivot_df = pivot_df.fillna(0)
    
    # Create stacked bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(pivot_df.columns)))
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=colors, width=0.8)
    
    ax1.set_title(f'{fuel_category}: Technology Competitiveness Distribution\n' + 
                  f'(±{category_margin*100:.0f}% cost margin)', fontweight='bold')
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Number of Countries')
    ax1.legend(title='Competitive Technologies', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Percentage of countries with multiple competitive options
    multi_tech_data = comp_df[comp_df['Competitive_Technologies'] != '1 tech'].groupby('Scenario')['Percentage'].sum().reset_index()
    
    bars = ax2.bar(multi_tech_data['Scenario'], multi_tech_data['Percentage'], 
                   color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, multi_tech_data['Percentage']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title(f'{fuel_category}: Countries with Multiple\nCompetitive Technologies', fontweight='bold')
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Percentage of Countries (%)')
    ax2.set_ylim(0, max(multi_tech_data['Percentage']) * 1.15 if len(multi_tech_data) > 0 else 100)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{fuel_category.lower().replace(" ", "_")}_competitiveness_analysis.png'
    plt.savefig(output_dir / filename, 
               dpi=300, 
               bbox_inches='tight', 
               facecolor='white')
    
    print(f"Saved: {filename}")
    plt.close()

def validate_competitiveness_analysis(optimal_tech_data, competitiveness_margins):
    """Validate and summarize the competitiveness analysis results"""
    
    if isinstance(competitiveness_margins, dict):
        margins_text = " | ".join([f"{cat}: ±{margin*100:.0f}%" for cat, margin in competitiveness_margins.items()])
        print(f"\n=== Competitiveness Analysis Summary ({margins_text}) ===")
    else:
        print(f"\n=== Competitiveness Analysis Summary (±{competitiveness_margins*100:.0f}%) ===")
    
    for scenario, scenario_data in optimal_tech_data.items():
        print(f"\n{scenario}:")
        
        for fuel_category, data in scenario_data.items():
            total_countries = len(data.dropna())
            multi_tech_countries = len(data[data['num_competitive'] > 1])
            multi_tech_percentage = (multi_tech_countries / total_countries) * 100 if total_countries > 0 else 0
            
            max_competitive = data['num_competitive'].max() if len(data) > 0 else 0
            
            # Get the margin for this category
            if isinstance(competitiveness_margins, dict):
                category_margin = competitiveness_margins.get(fuel_category, 0.10)
                margin_text = f"(±{category_margin*100:.0f}%)"
            else:
                margin_text = f"(±{competitiveness_margins*100:.0f}%)"
            
            print(f"  {fuel_category} {margin_text}:")
            print(f"    - Total countries: {total_countries}")
            print(f"    - Countries with multiple competitive techs: {multi_tech_countries} ({multi_tech_percentage:.1f}%)")
            print(f"    - Max competitive technologies in single country: {max_competitive}")

def main():
    """Main execution function"""
    set_plot_style()
    
    # ===== CONFIGURATION PARAMETERS =====
    # Define different competitiveness margins for each fuel category
    competitiveness_margins = {
        'Hydrogen': 0.20,      # 20% margin for hydrogen technologies
        'SAF': 0.80,          # 50% margin for SAF (kerosene) technologies  
        'Diesel': 0.20,       # 20% margin for diesel technologies
        'Other Fuels': 0.50   # 50% margin for other fuel technologies
    }
    
    # Alternative: Use single margin for all categories
    # competitiveness_margins = 0.20  # Single margin for all categories
    # =====================================
    
    # Define paths
    geojson_path = 'input/world_by_iso_geo.json'
    excel_file = 'output/lcox_results.xlsx'
    output_dir = Path('figures/optimal_technology_maps')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=== Competitive Technology Mapping Analysis ===")
    print(f"Using category-specific competitiveness margins:")
    
    # Load data
    print("\n1. Loading geographic data...")
    world_gdf = preprocess_geojson_safely(geojson_path)
    print(f"Loaded world map with {len(world_gdf)} countries")
    
    print("\n2. Loading LCOX data...")
    lcox_data = load_lcox_data(excel_file)
    
    print("\n3. Defining technology categories...")
    categories = define_technology_categories()
    color_palettes = define_color_palettes()
    
    # Print available technologies per category
    for fuel_category, tech_groups in categories.items():
        print(f"\n{fuel_category}:")
        for group, techs in tech_groups.items():
            available_techs = []
            for scenario_data in lcox_data.values():
                available_techs.extend([tech for tech in techs if tech in scenario_data.columns])
            available_techs = list(set(available_techs))
            print(f"  {group}: {available_techs}")
    
    print(f"\n4. Finding competitive technologies with category-specific margins...")
    print(f"   Hydrogen: ±{competitiveness_margins['Hydrogen']*100:.0f}%, SAF: ±{competitiveness_margins['SAF']*100:.0f}%, Diesel: ±{competitiveness_margins['Diesel']*100:.0f}%, Other Fuels: ±{competitiveness_margins['Other Fuels']*100:.0f}%")
    optimal_tech_data = find_cheapest_technology_per_category(lcox_data, categories, competitiveness_margins)
    
    print("\n5. Creating competitive technology maps...")
    create_optimal_technology_maps(world_gdf, optimal_tech_data, color_palettes, output_dir, competitiveness_margins)
    
    print("\n6. Creating technology dominance analysis...")
    create_cost_distribution_analysis(optimal_tech_data, color_palettes, output_dir)
    
    print("\n7. Creating summary statistics...")
    create_summary_statistics(optimal_tech_data, output_dir)
    
    print(f"\n=== Analysis Complete ===")
    print(f"All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("- 4 competitive technology maps (one per fuel category)")
    print("- 4 competitiveness analysis charts (one per fuel category)")
    print("- 4 technology dominance charts")
    print("- 1 summary statistics Excel file")
    print(f"\nNote: Countries with multiple competitive technologies (within category-specific margins) are shown with hatched patterns")
    print("Solid colors = single competitive technology or cheapest technology")
    print("Hatched overlays = additional competitive technologies")
    
    validate_competitiveness_analysis(optimal_tech_data, competitiveness_margins)

if __name__ == '__main__':
    main()

"""
=== USAGE EXAMPLES ===

To change the competitiveness margin, modify the competitiveness_margin variable in main():

# For a stricter analysis (5% margin):
competitiveness_margin = 0.05

# For a more lenient analysis (15% margin):
competitiveness_margin = 0.15

# For a very strict analysis (2% margin):
competitiveness_margin = 0.02

=== OUTPUT FILES ===

The script generates the following files in figures/optimal_technology_maps/:

1. competitive_hydrogen_technologies_map.png
   - World map showing hydrogen technologies
   - Solid colors = single competitive technology
   - Hatched patterns = multiple competitive technologies

2. competitive_saf_technologies_map.png
   - World map showing SAF technologies
   - Same visualization scheme as hydrogen

3. competitive_diesel_technologies_map.png
   - World map showing diesel technologies
   - Same visualization scheme as hydrogen

4. competitive_other_fuels_technologies_map.png
   - World map showing other fuel technologies
   - Same visualization scheme as hydrogen

5. [fuel]_competitiveness_analysis.png (4 files)
   - Bar charts showing competitiveness statistics
   - Distribution of countries by number of competitive technologies

6. [fuel]_technology_dominance.png (4 files)
   - Technology dominance analysis across scenarios

7. optimal_technology_summary_statistics.xlsx
   - Comprehensive statistics in Excel format

=== INTERPRETATION ===

- Solid colors: Only one technology is competitive within the margin
- Hatched overlays: Multiple technologies are competitive
- Different hatch patterns represent different secondary technologies
- The legend shows both primary (solid) and secondary (hatched) patterns

Countries with hatched patterns indicate regions where decision-makers have 
multiple viable technology options within the specified cost margin.
""" 