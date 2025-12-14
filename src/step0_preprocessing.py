"""
Step 0: Data Preprocessing Module

This module handles:
1. Loading raw production data from CSV files
2. Handling NaN values and shut-in times
3. Removing or keeping zeros based on configuration
4. Generating cumulative production (Gp)
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional
from src.utils.file_handler import get_file_info
from src.utils.visualization import plot_preprocessing


def process_csv_files(config: Dict) -> Dict[str, any]:
    """
    Process CSV files to handle NaN values and zeros.
    
    Args:
        config: Configuration dictionary containing:
            - input_directory: Path to input CSV files
            - output_directory: Path to save processed files
            - drop_zeros: Boolean to drop or keep zeros
            - save_intermediate: Boolean to save intermediate results
            - wells_to_process: List of well IDs or ["all"]
    
    Returns:
        Dictionary with processing statistics
    """
    input_directory = config['input_directory']
    output_directory = config['output_directory']
    drop_zeros = config.get('drop_zeros', True)
    save_intermediate = config.get('save_intermediate', True)
    wells_to_process = config.get('wells_to_process', ['all'])
    visualize_wells = config.get('visualize_wells', [])
    figure_format = config.get('figure_format', 'png')
    figure_dpi = config.get('figure_dpi', 300)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Create figures directory if visualization is enabled
    figures_dir = None
    if visualize_wells:
        figures_dir = os.path.join(output_directory, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
    
    # Filter out files that already have processing suffixes
    processing_suffixes = ['_clear', '_all', '_lof', '_knn', '_abod', '_cof', '_cluster', '_iforest', 
                          '_interpolated', '_smoothed', '_models']
    csv_files = [f for f in csv_files if not any(suffix in os.path.basename(f) for suffix in processing_suffixes)]
    
    # Filter files based on wells_to_process
    if wells_to_process != ['all']:
        csv_files = [f for f in csv_files if any(well_id in os.path.basename(f) for well_id in wells_to_process)]
    
    stats = {
        'total_files': len(csv_files),
        'processed_files': [],
        'errors': []
    }
    
    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Drop the second row if it exists (often contains units or metadata)
            if len(df) > 1:
                df = df.drop(df.index[1])
            
            # Rename columns to standard names
            df.columns = ['t', 'q_actual']
            
            # Delete rows with infinite values
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['t', 'q_actual'])
            
            # Fill NaN or empty cells in 'q_actual' with zero
            df['q_actual'] = df['q_actual'].fillna(0)
            
            # Convert to numeric
            df['t'] = pd.to_numeric(df['t'], errors='coerce')
            df['q_actual'] = pd.to_numeric(df['q_actual'], errors='coerce')
            
            # Remove any remaining NaN rows
            df = df.dropna()
            
            # Initial statistics
            initial_length = len(df)
            zero_count_before = (df['q_actual'] == 0).sum()
            
            # Apply drop_zeros option
            if drop_zeros:
                df = df[df['q_actual'] != 0]
                dropped_zero_count = zero_count_before
            else:
                dropped_zero_count = 0
            
            # Final statistics
            final_length = len(df)
            zero_count_after = (df['q_actual'] == 0).sum()
            
            # Prepare output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            new_file_name = f"{base_name}_clear.csv"
            new_file_path = os.path.join(output_directory, new_file_name)
            
            # Save or return data
            if save_intermediate:
                df.to_csv(new_file_path, index=False)
                print(f"Processed: {new_file_name}")
                print(f"  Initial length: {initial_length}, Final length: {final_length}")
                print(f"  Zeros dropped: {dropped_zero_count}")
            
            # Generate visualization if enabled
            if visualize_wells and figures_dir:
                # Check if this well should be visualized
                should_visualize = (visualize_wells == ['all'] or 
                                  any(well_id in base_name for well_id in visualize_wells))
                
                if should_visualize:
                    # Determine well type from filename
                    file_info = get_file_info(base_name)
                    well_type = file_info.get('well_type', 'oil')
                    well_type_upper = 'OIL' if well_type == 'oil' else 'GAS'
                    
                    # Calculate Gp for visualization
                    df_viz = df.copy()
                    df_viz['Gp_actual'] = df_viz['q_actual'].cumsum()
                    
                    # Create figure filename
                    fig_filename = os.path.splitext(new_file_name)[0] + f'.{figure_format}'
                    fig_path = os.path.join(figures_dir, fig_filename)
                    
                    # Generate plot
                    plot_preprocessing(df_viz, 
                                     title=f"Production Data for {new_file_name}",
                                     save_path=fig_path,
                                     show=False,
                                     well_type=well_type_upper)
            
            stats['processed_files'].append({
                'filename': new_file_name,
                'initial_length': initial_length,
                'final_length': final_length,
                'zeros_dropped': dropped_zero_count,
                'data': df if not save_intermediate else None
            })
            
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
            print(error_msg)
            stats['errors'].append(error_msg)
    
    return stats


def generate_cumulative(config: Dict) -> Dict[str, any]:
    """
    Generate cumulative production (Gp) from rate data.
    
    Args:
        config: Configuration dictionary containing:
            - input_directory: Path to input CSV files (with _clear suffix)
            - output_directory: Path to save Gp files
            - save_intermediate: Boolean to save intermediate results
            - wells_to_process: List of well IDs or ["all"]
    
    Returns:
        Dictionary with processing statistics
    """
    input_directory = config.get('input_directory', config.get('output_directory'))
    output_directory = config['output_directory']
    save_intermediate = config.get('save_intermediate', True)
    wells_to_process = config.get('wells_to_process', ['all'])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all CSV files with 'clear' in the name
    csv_files = glob.glob(os.path.join(input_directory, '*_clear.csv'))
    
    # Filter files based on wells_to_process
    if wells_to_process != ['all']:
        csv_files = [f for f in csv_files if any(well_id in f for well_id in wells_to_process)]
    
    stats = {
        'total_files': len(csv_files),
        'processed_files': [],
        'errors': []
    }
    
    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Calculate cumulative sum
            df['Gp_actual'] = df['q_actual'].cumsum()
            
            # Select only t and Gp_actual columns
            gp_df = df[['t', 'Gp_actual']].copy()
            
            # Prepare output filename (replace 'rates' with 'Gp')
            base_name = os.path.basename(file_path)
            new_file_name = base_name.replace('rates', 'Gp')
            new_file_path = os.path.join(output_directory, new_file_name)
            
            # Save or return data
            if save_intermediate:
                gp_df.to_csv(new_file_path, index=False)
                print(f"Generated Gp: {new_file_name}")
            
            stats['processed_files'].append({
                'filename': new_file_name,
                'final_gp': gp_df['Gp_actual'].iloc[-1] if len(gp_df) > 0 else 0,
                'data': gp_df if not save_intermediate else None
            })
            
        except Exception as e:
            error_msg = f"Error generating Gp for {os.path.basename(file_path)}: {str(e)}"
            print(error_msg)
            stats['errors'].append(error_msg)
    
    return stats


def run_preprocessing(config: Dict) -> Dict[str, any]:
    """
    Main function to run the complete preprocessing step.
    
    Args:
        config: Configuration dictionary for step 0
    
    Returns:
        Combined statistics from both processing and Gp generation
    """
    print("=" * 60)
    print("STEP 0: PREPROCESSING")
    print("=" * 60)
    
    # Ensure global settings are available
    if 'figure_format' not in config:
        config['figure_format'] = config.get('figure_format', 'png')
    if 'figure_dpi' not in config:
        config['figure_dpi'] = config.get('figure_dpi', 300)
    
    # Step 0.1: Process CSV files (handle NaN and zeros)
    print("\n[0.1] Processing CSV files (handling NaN and zeros)...")
    process_stats = process_csv_files(config)
    
    # Step 0.2: Generate cumulative production
    print(f"\n[0.2] Generating cumulative production (Gp)...")
    gp_stats = generate_cumulative(config)
    
    # Combined results
    results = {
        'step': 'step0_preprocessing',
        'processing_stats': process_stats,
        'gp_stats': gp_stats,
        'total_files_processed': process_stats['total_files'],
        'total_errors': len(process_stats['errors']) + len(gp_stats['errors'])
    }
    
    print(f"\n{'=' * 60}")
    print(f"STEP 0 COMPLETE")
    print(f"Files processed: {results['total_files_processed']}")
    print(f"Total errors: {results['total_errors']}")
    print(f"{'=' * 60}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    config = {
        'input_directory': 'Original/',
        'output_directory': 'output/step0/',
        'drop_zeros': True,
        'save_intermediate': True,
        'wells_to_process': ['all']
    }
    
    results = run_preprocessing(config)
