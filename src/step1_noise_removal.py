"""
Step 1: Noise Removal Module

This module handles:
1. Window-based outlier detection
2. Multiple anomaly detection algorithms (PyCaret)
3. Ensemble outlier detection
"""

import os
import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
from src.utils.file_handler import get_file_info
from src.utils.visualization import plot_noise_removal

# Import PyCaret for anomaly detection
try:
    import sys
    # Check Python version compatibility
    if sys.version_info >= (3, 12):
        raise ImportError("PyCaret does not support Python 3.12+")
    from pycaret.anomaly import setup, create_model, assign_model
    PYCARET_AVAILABLE = True
except ImportError as e:
    PYCARET_AVAILABLE = False
    print(f"Warning: PyCaret not available ({str(e)}). Only window-based outlier detection will work.")


def setup_pycaret(data: pd.DataFrame, normalize: bool = True, 
                  transformation: bool = False, transformation_method: str = 'quantile',
                  numeric_imputation: str = 'mean') -> any:
    """
    Initialize PyCaret for anomaly detection.
    
    Args:
        data: Input DataFrame
        normalize: Whether to normalize data
        transformation: Whether to apply transformation
        transformation_method: Transformation method to use
        numeric_imputation: Method for handling missing values
    
    Returns:
        PyCaret setup object
    """
    if not PYCARET_AVAILABLE:
        raise ImportError("PyCaret is required for anomaly detection algorithms")
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=786)
    
    # Initialize PyCaret setup
    return setup(data, 
                 normalize=normalize, 
                 transformation=transformation, 
                 numeric_imputation=numeric_imputation,
                 transformation_method=transformation_method,
                 session_id=786,
                 verbose=False)


def window_outliers(data: pd.Series, window_size: int = 10, step_size: int = 1,
                    method: str = 'lowest_quantile', num_lowest_points: int = 1) -> List[int]:
    """
    Detect outliers using a rolling window approach.
    
    Args:
        data: Input data series
        window_size: Size of the rolling window
        step_size: Step size for moving the window
        method: Detection method ('lowest_quantile', 'both_quantiles', 'lowest_points')
        num_lowest_points: Number of lowest points to remove (for 'lowest_points' method)
    
    Returns:
        List of outlier indices
    """
    outliers = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size]
        
        if method == 'lowest_quantile':
            q1 = window.quantile(0.25)
            outlier_indices = window[window < q1].index
        elif method == 'both_quantiles':
            q1 = window.quantile(0.25)
            q3 = window.quantile(0.75)
            outlier_indices = window[(window < q1) | (window > q3)].index
        elif method == 'lowest_points':
            outlier_indices = window.nsmallest(num_lowest_points).index
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outliers.extend(outlier_indices)
    
    return list(set(outliers))


def count_repeated_files(directory: str) -> Counter:
    """
    Count files with repeated well IDs in a directory.
    
    Args:
        directory: Directory path
    
    Returns:
        Counter object with ID counts
    """
    pattern = re.compile(r'(?:gas_|oil_)([^_]+)_')
    counts = Counter()
    
    for file in os.listdir(directory):
        match = pattern.search(file)
        if match:
            counts[match.group(1)] += 1
    
    return counts


def detect_and_remove_outliers(directory: str, algorithm_choice: str,
                               hyperparameters: Dict = None, use_window: bool = True,
                               window_size: int = 10, step_size: int = 1,
                               drop_repeated: bool = True, window_method: str = 'lowest_quantile',
                               num_lowest_points: int = 1,
                               wells_to_process: List[str] = None) -> List[Tuple]:
    """
    Main function to detect and remove outliers from CSV files.
    
    Args:
        directory: Input directory containing CSV files
        algorithm_choice: Algorithm to use ('knn', 'lof', 'abod', 'cof', 'cluster', 'iforest', 'all', 'window')
        hyperparameters: Dictionary of hyperparameters for algorithms
        use_window: Whether to use window-based outlier detection first
        window_size: Window size for outlier detection
        step_size: Step size for window movement
        drop_repeated: Whether to drop files with repeated well IDs
        window_method: Window detection method
        num_lowest_points: Number of lowest points to detect
        wells_to_process: List of well IDs to process
    
    Returns:
        List of tuples containing results for each file
    """
    # Count repeated files
    counts = count_repeated_files(directory)
    repeated_ids = {key for key, count in counts.items() if count > 1}
    
    # Identify processed keywords (files already processed by THIS step)
    # These are output suffixes from THIS step, so we should skip them
    processed_keywords = ['_lof', '_abod', '_cluster', '_knn', '_cof', '_iforest', '_window', '_all']
    
    # Find files to process - accept any CSV, but skip already processed ones
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Skip files that have already been processed by noise removal
    # Check if filename ends with these patterns (not just contains them)
    files = [f for f in all_files 
             if not any(keyword in os.path.splitext(f)[0] for keyword in processed_keywords)]
    
    # Filter by well IDs if specified
    if wells_to_process and wells_to_process != ['all']:
        files = [f for f in files if any(well_id in f for well_id in wells_to_process)]
    
    # Drop repeated files if requested
    if drop_repeated:
        files = [f for f in files if not any(repeated_id in f for repeated_id in repeated_ids)]
    
    results = []
    
    for file in files:
        file_path = os.path.join(directory, file)
        data = pd.read_csv(file_path)
        
        if data.shape[1] < 2:
            print(f"Skipping {file} - less than 2 columns")
            continue
        
        target = data.columns[1]
        print(f"Processing file: {file}")
        print(f"Data shape: {data.shape}")
        
        window_outliers_data = pd.DataFrame(columns=data.columns)
        
        # Apply window-based outlier detection
        if use_window:
            outlier_indices = window_outliers(data[target], window_size, step_size, 
                                            window_method, num_lowest_points)
            window_outliers_data = data.loc[outlier_indices]
            data = data.drop(index=outlier_indices)
            print(f"{file} - Window: Removed {len(outlier_indices)} outliers")
        
        # If only window method, return results
        if algorithm_choice == 'window':
            cleaned_data = data
            results.append((file, data, cleaned_data, window_outliers_data, 
                          data.columns[0], target, 'window', 
                          len(outlier_indices) / len(data) * 100 if use_window else 0))
        else:
            # Apply PyCaret anomaly detection
            if not PYCARET_AVAILABLE:
                print(f"Warning: PyCaret not available - falling back to window method for {file}")
                # Fall back to window method
                cleaned_data = data
                results.append((file, data, cleaned_data, window_outliers_data, 
                              data.columns[0], target, 'window', 
                              len(outlier_indices) / len(data) * 100 if use_window else 0))
                continue
            
            exp = setup_pycaret(data)
            selected_algorithms = ['lof', 'abod', 'cluster', 'knn', 'cof', 'iforest'] \
                                if algorithm_choice == 'all' else [algorithm_choice]
            
            all_outliers_indices = []
            
            for algo in selected_algorithms:
                try:
                    params = hyperparameters.get(algo, {}) if algorithm_choice == 'all' \
                            else (hyperparameters or {})
                    
                    model = create_model(algo, **params)
                    result = assign_model(model)
                    outliers = result[result['Anomaly'] == 1]
                    outlier_indices = outliers.index
                    filtered_outliers_set = set(outlier_indices)
                    all_outliers_indices.append(filtered_outliers_set)
                    print(f"{file} - {algo}: Found {len(outlier_indices)} outliers")
                except Exception as e:
                    print(f"Error processing {file} with {algo}: {e}")
            
            # Process results based on algorithm choice
            if algorithm_choice == 'all':
                if all_outliers_indices:
                    common_outliers = set.intersection(*all_outliers_indices)
                    outlier_percent = len(common_outliers) / len(data) * 100
                    cleaned_data = data.drop(index=common_outliers)
                    results.append((file, data, cleaned_data, window_outliers_data,
                                  data.columns[0], target, 'all', outlier_percent))
            else:
                for algo, outlier_indices in zip(selected_algorithms, all_outliers_indices):
                    cleaned_data = data.drop(index=outlier_indices)
                    results.append((file, data, cleaned_data, window_outliers_data,
                                  data.columns[0], target, algo,
                                  len(outlier_indices) / len(data) * 100))
    
    return results


def save_results(results: List[Tuple], output_directory: str, 
                 visualize_wells: List = None, figure_format: str = 'png', 
                 figure_dpi: int = 300) -> Dict[str, any]:
    """
    Save cleaned data to CSV files and generate visualizations if enabled.
    
    Args:
        results: List of result tuples from detect_and_remove_outliers
        output_directory: Output directory path
        visualize_wells: List of well IDs to visualize or ['all'] for all
        figure_format: Format for saved figures
        figure_dpi: DPI for saved figures
    
    Returns:
        Dictionary with save statistics
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Create figures directory if visualization is enabled
    figures_dir = None
    if visualize_wells:
        figures_dir = os.path.join(output_directory, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
    
    stats = {
        'files_saved': 0,
        'errors': []
    }
    
    for file, original_data, cleaned_data, window_outliers_data, x_col, y_col, algo, outlier_percent in results:
        try:
            cleaned_file_name = f"{os.path.splitext(file)[0]}_{algo}.csv"
            cleaned_file_path = os.path.join(output_directory, cleaned_file_name)
            cleaned_data.to_csv(cleaned_file_path, index=False)
            print(f"Saved: {cleaned_file_name} - Removed {outlier_percent:.2f}% outliers")
            stats['files_saved'] += 1
            
            # Generate visualization if enabled
            if visualize_wells and figures_dir:
                # Check if this well should be visualized
                base_name = os.path.splitext(file)[0]
                should_visualize = (visualize_wells == ['all'] or 
                                  any(well_id in base_name for well_id in visualize_wells))
                
                if should_visualize:
                    # Create figure filename
                    fig_filename = f"{os.path.splitext(cleaned_file_name)[0]}.{figure_format}"
                    fig_path = os.path.join(figures_dir, fig_filename)
                    
                    # Generate plot
                    plot_noise_removal(original_data, cleaned_data, window_outliers_data,
                                     title=f'Outlier Removal using {algo}',
                                     save_path=fig_path,
                                     show=False,
                                     algorithm=algo)
                    
        except Exception as e:
            error_msg = f"Error saving {file}: {str(e)}"
            print(error_msg)
            stats['errors'].append(error_msg)
    
    return stats


def run_noise_removal(config: Dict) -> Dict[str, any]:
    """
    Main function to run the noise removal step.
    
    Args:
        config: Configuration dictionary for step 1
    
    Returns:
        Statistics and results from noise removal
    """
    print("=" * 60)
    print("STEP 1: NOISE REMOVAL")
    print("=" * 60)
    
    input_directory = config['input_directory']
    output_directory = config['output_directory']
    
    # Count repeated files
    counts = count_repeated_files(input_directory)
    total_files = len([f for f in os.listdir(input_directory) 
                      if 'clear' in f and f.endswith('.csv')])
    total_repeated = sum(count for count in counts.values() if count > 1)
    
    print(f"\nTotal files: {total_files}")
    print(f"Repeated files: {total_repeated}")
    
    # Detect and remove outliers
    print("\nDetecting and removing outliers...")
    results = detect_and_remove_outliers(
        directory=input_directory,
        algorithm_choice=config.get('algorithms', ['all'])[0] if isinstance(config.get('algorithms', ['all']), list) else config.get('algorithms', 'all'),
        hyperparameters=config.get('hyperparameters', {}),
        use_window=config.get('use_window', True),
        window_size=config.get('window_size', 15),
        step_size=config.get('step_size', 5),
        drop_repeated=config.get('drop_repeated_files', True),
        window_method=config.get('window_method', 'lowest_points'),
        num_lowest_points=config.get('num_lowest_points', 3),
        wells_to_process=config.get('wells_to_process', ['all'])
    )
    
    # Save results if requested
    if config.get('save_intermediate', True):
        print(f"\nSaving cleaned files to {output_directory}...")
        visualize_wells = config.get('visualize_wells', [])
        figure_format = config.get('figure_format', 'png')
        figure_dpi = config.get('figure_dpi', 300)
        save_stats = save_results(results, output_directory, 
                                 visualize_wells=visualize_wells,
                                 figure_format=figure_format,
                                 figure_dpi=figure_dpi)
    else:
        save_stats = {'files_saved': 0, 'errors': []}
    
    summary = {
        'step': 'step1_noise_removal',
        'total_files_processed': len(results),
        'files_saved': save_stats['files_saved'],
        'errors': save_stats['errors'],
        'results': results
    }
    
    print(f"\n{'=' * 60}")
    print(f"STEP 1 COMPLETE")
    print(f"Files processed: {summary['total_files_processed']}")
    print(f"Files saved: {summary['files_saved']}")
    print(f"{'=' * 60}\n")
    
    return summary


if __name__ == "__main__":
    # Example usage
    config = {
        'input_directory': 'output/step0/',
        'output_directory': 'output/step1/',
        'use_window': True,
        'window_size': 15,
        'step_size': 5,
        'window_method': 'lowest_points',
        'num_lowest_points': 3,
        'algorithms': ['all'],
        'hyperparameters': {
            'knn': {'fraction': 0.20, 'n_neighbors': 20},
            'lof': {'fraction': 0.20, 'n_neighbors': 20},
            'abod': {'fraction': 0.20, 'n_neighbors': 3},
            'cof': {'fraction': 0.20, 'n_neighbors': 3},
            'cluster': {'fraction': 0.20, 'n_clusters': 3},
            'iforest': {'fraction': 0.20, 'n_estimators': 200}
        },
        'drop_repeated_files': True,
        'save_intermediate': True,
        'wells_to_process': ['all']
    }
    
    results = run_noise_removal(config)
