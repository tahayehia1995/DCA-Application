"""
Step 2: Interpolation Module

This module handles:
1. Interpolating missing time points in production data
2. Filling gaps in production history
3. Identifying fitting periods for decline curve analysis
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import glob
from src.utils.file_handler import get_file_info
from src.utils.visualization import plot_interpolation


def add_missing_months(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the DataFrame's 't' column has a continuous range without gaps.
    Missing months are added by reindexing the DataFrame over the complete range.
    
    Args:
        df: Input DataFrame containing a 't' column
    
    Returns:
        DataFrame reindexed to include all months in the range with NaNs where data is missing
    """
    # Create a range of months from the minimum to maximum value in 't'
    t_range = pd.Series(range(int(df['t'].min()), int(df['t'].max()) + 1))
    # Reindex the DataFrame to ensure 't' has no gaps, and reset the index
    df = df.set_index('t').reindex(t_range).reset_index().rename(columns={'index': 't'})
    return df


def nonlinear_interpolation(df: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
    """
    Fills missing values in the 'q_actual' column using non-linear interpolation.
    Matches original notebook algorithm exactly.
    
    Args:
        df: DataFrame containing a 'q_actual' column with missing values
    
    Returns:
        Tuple of (interpolated 'q_actual' series, boolean flags indicating interpolated values)
    """
    # Copy the 'q_actual' column for processing
    q_actual = df['q_actual'].copy()
    # Initialize a boolean array to track interpolated values
    interpolated_flags = np.zeros(len(q_actual), dtype=bool)

    # Identify indices of missing values in 'q_actual'
    missing_indices = q_actual[q_actual.isna()].index

    for idx in missing_indices:
        prev_idx = idx - 1  # Previous index
        next_idx = idx + 1  # Next index

        # Skip interpolation if the missing value is within the last 3 points
        if idx >= len(q_actual) - 3:
            continue

        # Find the previous non-missing value
        while prev_idx >= 0 and pd.isna(q_actual[prev_idx]):
            prev_idx -= 1

        # Find the next non-missing value
        while next_idx < len(q_actual) and pd.isna(q_actual[next_idx]):
            next_idx += 1

        # Get previous and next non-missing values, if available
        prev_val = q_actual[prev_idx] if prev_idx >= 0 else None
        next_val = q_actual[next_idx] if next_idx < len(q_actual) else None
        after_next_idx = next_idx + 1 if next_idx < len(q_actual) else None
        after_next_val = q_actual[after_next_idx] if after_next_idx is not None and after_next_idx < len(q_actual) and not pd.isna(q_actual[after_next_idx]) else None

        # Case 1: Single missing value between two known values
        if prev_val is not None and next_val is not None and next_idx - prev_idx == 2:
            if next_val > prev_val:
                # Leave the gap empty if there's an increasing trend
                continue
            else:
                # Find the closest value after the next known value that is less than the previous value
                closest_val = None
                closest_diff = float('inf')
                search_idx = after_next_idx

                while search_idx is not None and search_idx < len(q_actual):
                    search_val = q_actual[search_idx]
                    if not pd.isna(search_val) and search_val < prev_val:
                        diff = abs(search_val - prev_val)
                        if diff < closest_diff:
                            closest_val = search_val
                            closest_diff = diff
                    search_idx += 1
                    if search_idx >= len(q_actual):
                        search_idx = None

                # Interpolate based on closest_val if found, otherwise use prev_val and next_val
                if closest_val is not None:
                    interpolated_value = (prev_val + closest_val) / 2
                else:
                    interpolated_value = (prev_val + next_val) / 2
                q_actual[idx] = interpolated_value
                interpolated_flags[idx] = True
        
        # Case 2: Gap of multiple missing values
        elif prev_val is not None and next_val is not None and next_idx - prev_idx > 2:
            if next_val > prev_val:
                # Leave the gap empty if there's an increasing trend
                continue
            else:
                # Find the closest value after the next known value that is less than the previous value
                closest_val = None
                closest_diff = float('inf')
                search_idx = after_next_idx

                while search_idx is not None and search_idx < len(q_actual):
                    search_val = q_actual[search_idx]
                    if not pd.isna(search_val) and search_val < prev_val:
                        diff = abs(search_val - prev_val)
                        if diff < closest_diff:
                            closest_val = search_val
                            closest_diff = diff
                    search_idx += 1
                    if search_idx >= len(q_actual):
                        search_idx = None

                # Perform non-linear interpolation with closest_val or next_val
                if closest_val is not None:
                    # Logarithmic interpolation with closest_val
                    log_prev_val = np.log(prev_val)
                    log_closest_val = np.log(closest_val)
                    for gap_idx in range(prev_idx + 1, next_idx):
                        fraction = (gap_idx - prev_idx) / (next_idx - prev_idx)
                        log_interpolated_value = log_prev_val + fraction * (log_closest_val - log_prev_val)
                        interpolated_value = np.exp(log_interpolated_value)
                        q_actual[gap_idx] = interpolated_value
                        interpolated_flags[gap_idx] = True
                else:
                    # Logarithmic interpolation with next_val
                    log_prev_val = np.log(prev_val)
                    log_next_val = np.log(next_val)
                    for gap_idx in range(prev_idx + 1, next_idx):
                        fraction = (gap_idx - prev_idx) / (next_idx - prev_idx)
                        log_interpolated_value = log_prev_val + fraction * (log_next_val - log_prev_val)
                        interpolated_value = np.exp(log_interpolated_value)
                        q_actual[gap_idx] = interpolated_value
                        interpolated_flags[gap_idx] = True

    return q_actual, interpolated_flags


def interpolate_data(df: pd.DataFrame, method: str = 'linear') -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Interpolate missing data points in production data using original algorithm.
    
    Args:
        df: DataFrame with 't' and 'q_actual' columns
        method: Interpolation method (only 'linear' supported for original algorithm)
    
    Returns:
        Tuple of (interpolated DataFrame, boolean flags indicating interpolated points)
    """
    if len(df) < 2:
        return df, np.zeros(len(df), dtype=bool)
    
    # Ensure we have 't' and 'q_actual' columns
    if 't' not in df.columns:
        t_col = df.columns[0]
        df = df.rename(columns={t_col: 't'})
    if 'q_actual' not in df.columns:
        q_col = df.columns[1]
        df = df.rename(columns={q_col: 'q_actual'})
    
    # Step 1: Add missing months (creates NaNs for missing time points)
    df_with_gaps = add_missing_months(df.copy())
    
    # Step 2: Perform non-linear interpolation (tracks interpolated points)
    q_interpolated, interpolated_flags = nonlinear_interpolation(df_with_gaps.copy())
    
    # Step 3: Create final dataframe
    df_interpolated = pd.DataFrame({
        't': df_with_gaps['t'].values,
        'q_actual': q_interpolated.values
    })
    
    # Step 4: Remove rows with NaN values (but keep track of which were interpolated)
    # Create a mask for non-NaN rows BEFORE dropping
    non_nan_mask = ~df_interpolated.isna().any(axis=1)
    
    # Filter flags to only include rows that will remain after dropping NaNs
    final_interpolated_flags = interpolated_flags[non_nan_mask]
    
    # Remove NaN rows
    df_interpolated = df_interpolated[non_nan_mask].reset_index(drop=True)
    
    # Ensure flags match dataframe length
    if len(final_interpolated_flags) != len(df_interpolated):
        # This shouldn't happen, but handle it just in case
        if len(final_interpolated_flags) > len(df_interpolated):
            final_interpolated_flags = final_interpolated_flags[:len(df_interpolated)]
        else:
            final_interpolated_flags = np.pad(final_interpolated_flags, (0, len(df_interpolated) - len(final_interpolated_flags)), 
                                             constant_values=False)
    
    return df_interpolated, final_interpolated_flags


def identify_fitting_periods(df: pd.DataFrame, min_period_length: int = 30) -> List[Tuple[int, int]]:
    """
    Identify potential fitting periods for decline curve analysis.
    
    Args:
        df: DataFrame with production data
        min_period_length: Minimum length of a valid fitting period (months)
    
    Returns:
        List of tuples (start_time, end_time) for fitting periods
    """
    t_col = df.columns[0]
    q_col = df.columns[1]
    
    fitting_periods = []
    
    # Find periods where production is relatively stable or declining
    # Calculate rolling mean and std to identify stable periods
    window_size = min(20, len(df) // 4)
    if window_size < 3:
        window_size = 3
    
    df['rolling_mean'] = df[q_col].rolling(window=window_size, center=True).mean()
    df['rolling_std'] = df[q_col].rolling(window=window_size, center=True).std()
    
    # Find the peak production
    peak_idx = df[q_col].idxmax()
    peak_time = df.loc[peak_idx, t_col]
    
    # Main fitting period: from peak to end (or until production becomes very low)
    end_idx = len(df) - 1
    
    # Find where production drops below 10% of peak
    peak_value = df.loc[peak_idx, q_col]
    low_production_mask = df[q_col] < (0.1 * peak_value)
    
    if low_production_mask.any():
        first_low_idx = low_production_mask.idxmax()
        if low_production_mask.loc[first_low_idx]:
            end_idx = first_low_idx
    
    # Check if period is long enough
    period_length = df.loc[end_idx, t_col] - peak_time
    if period_length >= min_period_length:
        fitting_periods.append((int(peak_time), int(df.loc[end_idx, t_col])))
    
    # Clean up temporary columns
    df.drop(['rolling_mean', 'rolling_std'], axis=1, inplace=True, errors='ignore')
    
    return fitting_periods if fitting_periods else [(int(df[t_col].min()), int(df[t_col].max()))]


def process_file_interpolation(file_path: str, config: Dict) -> Tuple[pd.DataFrame, List[Tuple], str, pd.DataFrame, np.ndarray]:
    """
    Process a single file for interpolation.
    
    Args:
        file_path: Path to input CSV file
        config: Configuration dictionary
    
    Returns:
        Tuple of (interpolated_df, fitting_periods, filename, original_df, interpolated_flags)
    """
    method = config.get('interpolation_method', 'linear')
    min_period = config.get('min_fitting_period', 30)
    
    # Read file
    df_original = pd.read_csv(file_path)
    
    # Interpolate
    df_interpolated, interpolated_flags = interpolate_data(df_original.copy(), method=method)
    
    # Identify fitting periods
    fitting_periods = identify_fitting_periods(df_interpolated.copy(), min_period_length=min_period)
    
    filename = os.path.basename(file_path)
    
    return df_interpolated, fitting_periods, filename, df_original, interpolated_flags


def run_interpolation(config: Dict) -> Dict[str, any]:
    """
    Main function to run the interpolation step.
    
    Args:
        config: Configuration dictionary for step 2
    
    Returns:
        Statistics and results from interpolation
    """
    print("=" * 60)
    print("STEP 2: INTERPOLATION & FITTING PERIODS")
    print("=" * 60)
    
    input_directory = config['input_directory']
    output_directory = config['output_directory']
    save_intermediate = config.get('save_intermediate', True)
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
    
    # Find all CSV files - accept any CSV that hasn't been interpolated yet
    all_csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
    
    # Skip files that have already been interpolated (output from THIS step)
    csv_files = [f for f in all_csv_files if '_interpolated' not in os.path.splitext(os.path.basename(f))[0]]
    
    # Filter by well IDs if specified
    if wells_to_process and wells_to_process != ['all']:
        csv_files = [f for f in csv_files if any(well_id in f for well_id in wells_to_process)]
    
    results = {
        'step': 'step2_interpolation',
        'total_files': len(csv_files),
        'processed_files': [],
        'errors': []
    }
    
    print(f"\nFound {len(csv_files)} files to process")
    
    for file_path in csv_files:
        try:
            print(f"\nProcessing: {os.path.basename(file_path)}")
            
            df_interpolated, fitting_periods, filename, df_original, interpolated_flags = process_file_interpolation(file_path, config)
            
            print(f"  Original points: {len(df_original)}")
            print(f"  Interpolated points: {df_interpolated.shape[0]}")
            print(f"  Fitting periods: {fitting_periods}")
            
            # Save if requested
            if save_intermediate:
                output_filename = filename.replace('.csv', '_interpolated.csv')
                output_path = os.path.join(output_directory, output_filename)
                df_interpolated.to_csv(output_path, index=False)
                print(f"  Saved: {output_filename}")
            
            # Generate visualization if enabled
            if visualize_wells and figures_dir:
                # Check if this well should be visualized
                base_name = os.path.splitext(filename)[0]
                should_visualize = (visualize_wells == ['all'] or 
                                  any(well_id in base_name for well_id in visualize_wells))
                
                if should_visualize:
                    # Interpolation plot
                    # Note: Original notebook plots interpolated data twice - once as "Original" (all points)
                    # and once as "Interpolated" (only flagged points). We need to pass the same
                    # interpolated dataframe but use flags to distinguish.
                    fig_filename_interp = f"{os.path.splitext(output_filename)[0]}_interpolation.{figure_format}"
                    fig_path_interp = os.path.join(figures_dir, fig_filename_interp)
                    
                    # Create a version of original data that matches interpolated data structure
                    # The original data has gaps, interpolated data has continuous time
                    # For visualization, we plot interpolated data as "original" (all points in blue)
                    # and flagged points as "interpolated" (red)
                    plot_interpolation(df_interpolated, df_interpolated, 
                                     interpolated_flags=interpolated_flags,
                                     title=f'Semi-log Plot of {filename}',
                                     save_path=fig_path_interp,
                                     show=False)
            
            results['processed_files'].append({
                'filename': filename,
                'output_filename': output_filename if save_intermediate else None,
                'original_points': len(df_original),
                'interpolated_points': df_interpolated.shape[0],
                'fitting_periods': fitting_periods,
                'data': df_interpolated if not save_intermediate else None
            })
            
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
            print(f"  {error_msg}")
            results['errors'].append(error_msg)
    
    print(f"\n{'=' * 60}")
    print(f"STEP 2 COMPLETE")
    print(f"Files processed: {len(results['processed_files'])}")
    print(f"Total errors: {len(results['errors'])}")
    print(f"{'=' * 60}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    config = {
        'input_directory': 'output/step1/',
        'output_directory': 'output/step2/',
        'interpolation_method': 'linear',
        'min_fitting_period': 30,
        'save_intermediate': True,
        'wells_to_process': ['all']
    }
    
    results = run_interpolation(config)
