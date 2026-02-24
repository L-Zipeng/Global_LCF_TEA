import pandas as pd
import numpy as np
from pathlib import Path

def export_capex_format(input_file='output/future_capex_results.csv', 
                       output_file='output/capex_export_format.csv'):
    """
    Export CAPEX data in the specific format requested:
    - tech column with technology names
    - Scenario columns: Base_2024, Base_2030, 2degree_2030, 1.5degree_2030, Base_2050, 2degree_2050, 1.5degree_2050
    """
    
    print("Reading CAPEX results...")
    
    try:
        # Read the detailed results
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records")
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        print("Please run calculate_future_capex.py first to generate the results.")
        return None
    
    # Create scenario-year combinations
    df['scenario_year'] = df['Scenario'] + '_' + df['Year'].astype(str)
    
    # Create wide format pivot
    wide_df = df.pivot_table(
        index='Technology',
        columns='scenario_year',
        values='CAPEX_future',
        aggfunc='first'
    ).reset_index()
    
    # Rename Technology column to 'tech'
    wide_df = wide_df.rename(columns={'Technology': 'tech'})
    
    # Get base CAPEX (2024 values) for Base_2024 column
    base_capex = df[df['Year'] == 2024].groupby('Technology')['CAPEX_EUR_per_kW'].first().reset_index()
    base_capex = base_capex.rename(columns={'Technology': 'tech', 'CAPEX_EUR_per_kW': 'Base_2024'})
    
    # Merge base CAPEX
    result_df = wide_df.merge(base_capex[['tech', 'Base_2024']], on='tech', how='left')
    
    # Define the desired column order
    desired_columns = [
        'tech',
        'Base_2024',      # Base CAPEX (using 2024 values as reference)
        'Base_2030',      # Base scenario 2030
        '2degree_2030',   # 2 degree scenario 2030
        '1.5degree_2030', # 1.5 degree scenario 2030
        'Base_2050',      # Base scenario 2050
        '2degree_2050',   # 2 degree scenario 2050
        '1.5degree_2050'  # 1.5 degree scenario 2050
    ]
    
    # Check which columns exist and create missing ones with NaN
    available_columns = []
    for col in desired_columns:
        if col in result_df.columns:
            available_columns.append(col)
        else:
            result_df[col] = np.nan
            available_columns.append(col)
            print(f"Warning: Column '{col}' not found, filled with NaN")
    
    # Select and reorder columns
    export_df = result_df[desired_columns].copy()
    
    # Round CAPEX values to 2 decimal places
    numeric_cols = export_df.select_dtypes(include=[np.number]).columns
    export_df[numeric_cols] = export_df[numeric_cols].round(2)
    
    # Define custom technology ordering as requested by user
    tech_order = [
        'PEM',
        'AE', 
        'SOEC',
        'HTSE',
        'CuCl',
        'SMR_CCS',
        'ATR_CCS',
        'CLR',
        'M_PYR',
        'TG_CCS',
        'SR_FT',
        'ST_FT',
        'TG_FT',
        'RWGS_FT',
        'RWGS_MeOH',
        'HTL',
        'HVO',
        'B_PYR',
        'PTM',
        'AD',
        'HB',
        'DAC'
    ]
    
    # Create a mapping for sorting
    tech_rank = {tech: i for i, tech in enumerate(tech_order)}
    
    # Add rank column for sorting
    export_df['tech_rank'] = export_df['tech'].map(tech_rank)
    
    # Handle any technologies not in the predefined order (put them at the end)
    max_rank = len(tech_order)
    export_df['tech_rank'] = export_df['tech_rank'].fillna(max_rank)
    
    # Sort by the custom technology ranking
    export_df = export_df.sort_values('tech_rank').reset_index(drop=True)
    
    # Remove the ranking column before export
    export_df = export_df.drop('tech_rank', axis=1)
    
    # Save the formatted export
    print(f"Saving formatted CAPEX export to {output_file}...")
    export_df.to_csv(output_file, index=False)
    
    # Also create an Excel version for better formatting
    excel_file = output_file.replace('.csv', '.xlsx')
    export_df.to_excel(excel_file, index=False, sheet_name='CAPEX_by_Scenario')
    print(f"Also saved Excel version to {excel_file}")
    
    # Print summary
    print("\nExport Summary:")
    print(f"Technologies: {len(export_df)}")
    print(f"Columns: {list(export_df.columns)}")
    
    # Show the data structure
    print(f"\nFirst 10 rows of exported data (in custom order):")
    print(export_df.head(10).to_string(index=False))
    
    # Check for any missing data
    missing_data = export_df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\nMissing data by column:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"  {col}: {count} missing values")
    
    # Create summary statistics
    print(f"\nCAPEX Statistics by Scenario:")
    for col in desired_columns[1:]:  # Skip 'tech' column
        if col in export_df.columns:
            values = export_df[col].dropna()
            if len(values) > 0:
                print(f"  {col}:")
                print(f"    Mean: {values.mean():.2f} EUR/kW")
                print(f"    Range: {values.min():.2f} - {values.max():.2f} EUR/kW")
    
    print(f"\nTechnology order used:")
    for i, tech in enumerate(export_df['tech'], 1):
        print(f"  {i:2d}. {tech}")
    
    return export_df

def create_formatted_table_view(export_df, output_file='output/capex_table_view.txt'):
    """Create a nicely formatted table view for easy copying."""
    
    print(f"\nCreating formatted table view...")
    
    # Create a more readable format
    with open(output_file, 'w') as f:
        # Write header
        f.write("CAPEX by Technology and Scenario (EUR/kW)\n")
        f.write("=" * 80 + "\n\n")
        
        # Write column headers
        headers = export_df.columns.tolist()
        f.write(f"{'tech':<15}")
        for header in headers[1:]:
            f.write(f"{header:>12}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        
        # Write data rows
        for _, row in export_df.iterrows():
            f.write(f"{row['tech']:<15}")
            for col in headers[1:]:
                value = row[col]
                if pd.isna(value):
                    f.write(f"{'N/A':>12}")
                else:
                    f.write(f"{value:>12.2f}")
            f.write("\n")
    
    print(f"Formatted table saved to {output_file}")

def create_technology_list(export_df, output_file='output/technology_list.txt'):
    """Create a simple list of technologies for reference."""
    
    print(f"\nCreating technology list...")
    
    with open(output_file, 'w') as f:
        f.write("Technology List:\n")
        f.write("=" * 30 + "\n")
        for i, tech in enumerate(export_df['tech'], 1):
            f.write(f"{i:2d}. {tech}\n")
    
    print(f"Technology list saved to {output_file}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    Path('output').mkdir(exist_ok=True)
    
    # Export the CAPEX data in the requested format
    export_df = export_capex_format()
    
    if export_df is not None:
        # Create additional formatted outputs
        create_formatted_table_view(export_df)
        create_technology_list(export_df)
        
        print("\n" + "="*60)
        print("EXPORT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Files created:")
        print("  - output/capex_export_format.csv (CSV format)")
        print("  - output/capex_export_format.xlsx (Excel format)")
        print("  - output/capex_table_view.txt (Formatted table view)")
        print("  - output/technology_list.txt (Technology list)")
    else:
        print("Export failed. Please check that the input file exists.") 