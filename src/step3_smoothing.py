"""
Step 3: Smoothing Module

This module handles:
1. Applying various smoothing filters to production data
2. Gaussian filter
3. Savitzky-Golay filter
4. Spline smoothing
5. LOWESS smoothing
"""

import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
import glob
from typing import Dict, List, Tuple, Optional
from src.utils.file_handler import get_file_info
from src.utils.visualization import plot_smoothing


def apply_filters(t: np.ndarray, q_actual: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply various smoothing filters to production data.
    
    Args:
        t: Time array
        q_actual: Production rate array
    
    Returns:
        Dictionary with filter names as keys and smoothed data as values
    """
    filters = {}
    data_len = len(q_actual)
    std_dev = np.std(q_actual)
    q_range = np.max(q_actual) - np.min(q_actual)
    
    # Gaussian Filter
    try:
        sigma_gaussian = max(0.5, min(2, std_dev / q_range)) if q_range > 0 else 1.0
        filters['Gaussian'] = gaussian_filter(q_actual, sigma=sigma_gaussian)
    except Exception as e:
        print(f"  Warning: Gaussian filter failed: {e}")
        filters['Gaussian'] = q_actual.copy()
    
    # Savitzky-Golay Filter
    try:
        window_length_sg = max(5, min(11, (data_len // 10) | 1))
        if window_length_sg % 2 == 0:  # Must be odd
            window_length_sg += 1
        polyorder_sg = min(3, (window_length_sg - 1) // 2)
        
        if window_length_sg <= data_len:
            filters['Savitzky-Golay'] = savgol_filter(q_actual, window_length=window_length_sg, 
                                                      polyorder=polyorder_sg)
        else:
            filters['Savitzky-Golay'] = q_actual.copy()
    except Exception as e:
        print(f"  Warning: Savitzky-Golay filter failed: {e}")
        filters['Savitzky-Golay'] = q_actual.copy()
    
    # Spline Smoothing
    try:
        if data_len > 3:
            spline_s = std_dev * 0.5
            spline = UnivariateSpline(t, q_actual, s=spline_s)
            filters['Spline'] = spline(t)
        else:
            filters['Spline'] = q_actual.copy()
    except Exception as e:
        print(f"  Warning: Spline filter failed: {e}")
        filters['Spline'] = q_actual.copy()
    
    # LOWESS
    try:
        frac_lowess = max(0.05, min(0.3, 0.1 * std_dev / q_range)) if q_range > 0 else 0.1
        lowess_filtered = lowess(q_actual, t, frac=frac_lowess)
        filters['Lowess'] = np.array([point[1] for point in lowess_filtered])
    except Exception as e:
        print(f"  Warning: LOWESS filter failed: {e}")
        filters['Lowess'] = q_actual.copy()
    
    return filters


def process_and_smooth_csv_files(directory: str, config: Dict) -> Dict[str, any]:
    """
    Process CSV files and apply smoothing filters.
    
    Args:
        directory: Input directory path
        config: Configuration dictionary
    
    Returns:
        Dictionary with processing results
    """
    output_directory = config.get('output_directory', directory)
    save_intermediate = config.get('save_intermediate', True)
    filters_to_apply = config.get('filters', ['Gaussian', 'Savitzky-Golay', 'Spline', 'Lowess'])
    wells_to_process = config.get('wells_to_process', ['all'])
    visualize_wells = config.get('visualize_wells', [])
    figure_format = config.get('figure_format', 'png')
    figure_dpi = config.get('figure_dpi', 300)
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Create figures directory if visualization is enabled
    figures_dir = None
    if visualize_wells:
        figures_dir = os.path.join(output_directory, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
    
    # Find files to process - accept any CSV that hasn't been smoothed yet (output from THIS step)
    files_to_process = []
    for filename in os.listdir(directory):
        # Skip files that start with 'smoothed_' (output from THIS step)
        if filename.endswith(".csv") and not filename.startswith('smoothed_'):
            # Filter by well IDs if specified
            if wells_to_process == ['all'] or any(well_id in filename for well_id in wells_to_process):
                files_to_process.append(filename)
    
    results = {
        'step': 'step3_smoothing',
        'total_files': len(files_to_process),
        'processed_files': [],
        'errors': []
    }
    
    for filename in files_to_process:
        try:
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            
            # Check if file has necessary columns
            if 't' not in data.columns or data.shape[1] < 2:
                print(f"File {filename} does not have the necessary columns for processing.")
                continue
            
            t = data['t'].values
            q_actual = data.iloc[:, 1].values
            
            print(f"\nProcessing: {filename}")
            print(f"  Data points: {len(t)}")
            
            # Apply filters
            filters = apply_filters(t, q_actual)
            
            # Add filtered data to DataFrame
            for filter_name, filtered_data in filters.items():
                if filter_name in filters_to_apply:
                    # Ensure length matches
                    filtered_series = pd.Series(filtered_data[:len(data)], 
                                              index=data.index[:len(filtered_data)])
                    data[filter_name] = filtered_series
            
            # Create output filename with smoothed prefix
            base_name = os.path.splitext(filename)[0]
            output_filename = f"smoothed_{base_name}.csv"
            
            # Save if requested
            if save_intermediate:
                output_file_path = os.path.join(output_directory, output_filename)
                data.to_csv(output_file_path, index=False)
                print(f"  Saved: {output_filename}")
            
            # Generate visualization if enabled
            if visualize_wells and figures_dir:
                # Check if this well should be visualized
                base_name = os.path.splitext(filename)[0]
                should_visualize = (visualize_wells == ['all'] or 
                                  any(well_id in base_name for well_id in visualize_wells))
                
                if should_visualize:
                    # Create figure filename
                    fig_filename = f"{os.path.splitext(output_filename)[0]}_smoothing.{figure_format}"
                    fig_path = os.path.join(figures_dir, fig_filename)
                    
                    # Get filter columns that were actually applied
                    filter_cols = [f for f in filters_to_apply if f in data.columns]
                    
                    # Generate plot
                    well_name = base_name.split('.')[0] if '.' in base_name else base_name
                    plot_smoothing(data, filter_cols,
                                 title=f"Well: {well_name}",
                                 save_path=fig_path,
                                 show=False)
            
            results['processed_files'].append({
                'filename': filename,
                'output_filename': output_filename if save_intermediate else None,
                'filters_applied': list(filters.keys()),
                'data': data if not save_intermediate else None
            })
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f"  {error_msg}")
            results['errors'].append(error_msg)
    
    return results


def run_smoothing(config: Dict) -> Dict[str, any]:
    """
    Main function to run the smoothing step.
    
    Args:
        config: Configuration dictionary for step 3
    
    Returns:
        Statistics and results from smoothing
    """
    print("=" * 60)
    print("STEP 3: SMOOTHING")
    print("=" * 60)
    
    input_directory = config['input_directory']
    
    results = process_and_smooth_csv_files(input_directory, config)
    
    print(f"\n{'=' * 60}")
    print(f"STEP 3 COMPLETE")
    print(f"Files processed: {len(results['processed_files'])}")
    print(f"Total errors: {len(results['errors'])}")
    print(f"{'=' * 60}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    config = {
        'input_directory': 'output/step2/',
        'output_directory': 'output/step3/',
        'filters': ['Gaussian', 'Savitzky-Golay', 'Spline', 'Lowess'],
        'save_intermediate': True,
        'wells_to_process': ['all']
    }
    
    results = run_smoothing(config)
