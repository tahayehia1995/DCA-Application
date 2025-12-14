"""
Batch processing utility for running all workflow steps on all wells
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st

from streamlit_app.utils.session_state import (
    get_well_data, set_step_result, get_step_result
)
from streamlit_app.config.default_config import get_default_config
from src.utils.file_handler import get_file_info


def process_step0_preprocessing(well_id: str, well_data: pd.DataFrame, config: Dict) -> Dict:
    """Process Step 0: Preprocessing"""
    df = well_data.copy()
    
    # Ensure correct column names
    if 't' not in df.columns or 'q_actual' not in df.columns:
        if len(df.columns) >= 2:
            df.columns = ['t', 'q_actual'] + list(df.columns[2:])
    
    # Convert to numeric
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    df['q_actual'] = pd.to_numeric(df['q_actual'], errors='coerce')
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['t', 'q_actual'])
    
    # Fill NaN with zero
    df['q_actual'] = df['q_actual'].fillna(0)
    
    # Store original for comparison
    original_df = df.copy()
    initial_length = len(df)
    zero_count_before = (df['q_actual'] == 0).sum()
    
    # Apply zero removal
    drop_zeros = config.get('drop_zeros', True)
    zeros_removed = None
    if drop_zeros:
        zeros_mask = df['q_actual'] == 0
        zeros_removed = df[zeros_mask].copy()
        df = df[~zeros_mask]
    
    final_length = len(df)
    zeros_dropped = zero_count_before if drop_zeros else 0
    
    # Calculate cumulative production
    df['Gp_actual'] = df['q_actual'].cumsum()
    
    return {
        'processed_df': df,
        'original_df': original_df,
        'zeros_removed': zeros_removed,
        'initial_length': initial_length,
        'final_length': final_length,
        'zeros_dropped': zeros_dropped,
        'drop_zeros': drop_zeros
    }


def process_step1_noise_removal(well_id: str, input_df: pd.DataFrame, config: Dict) -> Dict:
    """Process Step 1: Noise Removal - Simplified for batch processing"""
    # For batch processing, use a simplified approach
    # Full implementation would use detect_and_remove_outliers from step1_noise_removal
    # but that requires file I/O, so we'll do a simple pass-through for now
    # Users can re-run Step 1 individually with full algorithms if needed
    
    cleaned_df = input_df.copy()
    
    # Simple window-based outlier removal if enabled
    if config.get('use_window', True):
        window_size = config.get('window_size', 15)
        step_size = config.get('step_size', 5)
        window_method = config.get('window_method', 'lowest_points')
        num_lowest_points = config.get('num_lowest_points', 3)
        
        t = input_df['t'].values
        q = input_df['q_actual'].values
        
        outliers_mask = np.zeros(len(input_df), dtype=bool)
        for i in range(0, max(1, len(input_df) - window_size + 1), step_size):
            end_idx = min(i + window_size, len(input_df))
            window_q = q[i:end_idx]
            if len(window_q) > num_lowest_points:
                threshold_idx = np.argsort(window_q)[num_lowest_points]
                threshold = window_q[threshold_idx]
                outliers_mask[i:end_idx] |= (window_q <= threshold)
        
        window_outliers = input_df[outliers_mask].copy()
        cleaned_df = input_df[~outliers_mask].copy()
    else:
        window_outliers = None
    
    # Use cleaned data as default output
    cleaned_data_dict = {'default': cleaned_df}
    
    return {
        'original_df': input_df.copy(),  # Store original for visualization
        'cleaned_data_dict': cleaned_data_dict,
        'window_outliers': window_outliers,
        'output_df': cleaned_df,
        'algorithm_names': ['default'],  # For compatibility with page display
        'parameters': {
            'use_window': config.get('use_window', True),
            'window_size': config.get('window_size', 15),
            'step_size': config.get('step_size', 5),
            'window_method': config.get('window_method', 'lowest_points'),
            'algorithms': ['default'],
            'hyperparameters': {}
        }
    }


def process_step2_interpolation(well_id: str, input_df: pd.DataFrame, config: Dict) -> Dict:
    """Process Step 2: Interpolation"""
    from step2_interpolation import interpolate_data, identify_fitting_periods
    
    # Interpolate data
    interpolation_method = config.get('interpolation_method', 'linear')
    df_interpolated, interpolated_flags = interpolate_data(input_df.copy(), method=interpolation_method)
    
    # Identify fitting periods
    min_period_length = config.get('min_fitting_period', 30)
    fitting_periods = identify_fitting_periods(df_interpolated.copy(), min_period_length=min_period_length)
    
    return {
        'original_df': input_df,
        'interpolated_df': df_interpolated,
        'fitting_periods': fitting_periods,
        'interpolated_flags': interpolated_flags,
        'output_df': df_interpolated
    }


def process_step3_smoothing(well_id: str, input_df: pd.DataFrame, config: Dict) -> Dict:
    """Process Step 3: Smoothing"""
    from step3_smoothing import apply_filters
    
    t = input_df['t'].values
    q_actual = input_df['q_actual'].values
    
    # Apply filters
    filters_dict = apply_filters(t, q_actual)
    
    # Create output DataFrame
    output_df = input_df.copy()
    
    # Add filtered columns
    selected_filters = config.get('filters', ['Gaussian', 'Savitzky-Golay', 'Spline', 'Lowess'])
    applied_filters = []
    for filter_name in selected_filters:
        if filter_name in filters_dict:
            output_df[filter_name] = filters_dict[filter_name]
            applied_filters.append(filter_name)
    
    return {
        'input_df': input_df,
        'output_df': output_df,
        'filters_dict': {k: v for k, v in filters_dict.items() if k in applied_filters},
        'applied_filters': applied_filters
    }


def process_step4_fitting(well_id: str, input_df: pd.DataFrame, config: Dict, well_type: str) -> Dict:
    """Process Step 4: DCA Fitting for batch processing"""
    from step4_q_fitting import (
        fit_decline_model, calculate_error_metrics,
        find_peaks_and_troughs, validate_fitting_period
    )
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    t = pd.Series(input_df['t'].values)
    q_actual = input_df['q_actual'].values
    
    # Get configuration parameters
    lowess_frac = config.get('lowess_frac', 0.30)
    min_peak_rel_height = config.get('min_peak_rel_height', 0.2)
    min_months = config.get('min_months', 30)
    fitting_method = config.get('fitting_method', 1)
    lmfit_method = config.get('lmfit_method', 'leastsq')
    fitting_accuracy_threshold = config.get('fitting_accuracy_threshold', 90)
    prediction_accuracy_threshold = config.get('prediction_accuracy_threshold', 80)
    minimum_production_history = config.get('minimum_production_history', 3)
    forecast_method = config.get('forecast_method', 1)
    forecast_end_time = config.get('forecast_end_time', 400)
    forecast_end_flow_rate_oil = config.get('forecast_end_flow_rate_oil', 3.0)
    forecast_end_flow_rate_gas = config.get('forecast_end_flow_rate_gas', 100.0)
    selected_models = config.get('models', ['Arps_Hyperbolic', 'Logistic_Growth', 'Stretched_Exponential', 
                                            'Power_Law', 'Duong', 'Wang', 'VDMA'])
    
    # Apply LOWESS smoothing for peak detection
    q_smoothed_lowess = lowess(q_actual, t, frac=lowess_frac, return_sorted=False)
    
    # Find peaks and troughs
    peaks, troughs = find_peaks_and_troughs(q_smoothed_lowess, min_peak_rel_height)
    
    # Validate fitting period
    best_start, best_end = validate_fitting_period(
        t, q_smoothed_lowess, q_actual, peaks, troughs, min_months
    )
    
    start_idx = t[t == best_start].index[0]
    end_idx = t[t == best_end].index[0]
    
    # Determine which smoothed columns to use
    smoothed_columns = [col for col in input_df.columns if col not in ['t', 'q_actual']]
    smoothing_methods_to_fit = smoothed_columns if smoothed_columns else ['raw']
    
    # Model functions
    from step4_q_fitting import (
        arps_hyperbolic, logistic_growth, stretched_exponential, power_law,
        duong, wang, vdma
    )
    
    models = {
        'Arps_Hyperbolic': arps_hyperbolic,
        'Logistic_Growth': logistic_growth,
        'Stretched_Exponential': stretched_exponential,
        'Power_Law': power_law,
        'Duong': duong,
        'Wang': wang,
        'VDMA': vdma,
    }
    
    # Fit models on all smoothing methods
    well_results = []
    all_combined_curves = {}
    
    # Loop through each smoothing method
    for smoothing_method in smoothing_methods_to_fit:
        # Get smoothed data for this method
        if smoothing_method == 'raw':
            q_smoothed_values = q_actual
        else:
            if smoothing_method in input_df.columns:
                q_smoothed_values = input_df[smoothing_method].values
            else:
                continue  # Skip if column doesn't exist
        
        # Prepare fitting data for this smoothing method
        fitting_t = t[start_idx:end_idx+1]
        fitting_data = q_smoothed_values[start_idx:end_idx+1]
        
        # Split into train/test (80/20)
        split_index = int(len(fitting_t) * 0.8)
        train_t = fitting_t[:split_index]
        train_data = fitting_data[:split_index]
        test_t = fitting_t[split_index:]
        test_data = fitting_data[split_index:]
        
        if len(train_data) < minimum_production_history:
            continue  # Skip this smoothing method if insufficient data
        
        qi_initial = np.max(fitting_data)
        
        # Model bounds (recalculated per smoothing method)
        default_bounds = {
            'Arps_Hyperbolic': [(qi_initial*0.5, qi_initial*5), (0.0001, 0.1), (0.0001, 4)],
            'Logistic_Growth': [(qi_initial*1000, qi_initial*15000), (10, 1000), (0.00001, 1.5)],
            'Stretched_Exponential': [(qi_initial*0.5, qi_initial*1.5), (0.001, 2), (0.001, 2)],
            'Power_Law': [(qi_initial*0.5, qi_initial*1.5), (0.0001, 2), (0.0001, 2)],
            'Duong': [(qi_initial*0.5, qi_initial*1.5), (0.01, 10), (0.5, 1.5)],
            'Wang': [(qi_initial*0.5, qi_initial*1.5), (0.0001, 10)],
            'VDMA': [(qi_initial*0.5, qi_initial*1.5), (0.00001, 2), (0.1, 2)],
        }
        
        # Fit all selected models on this smoothing method
        for model_name in selected_models:
            if model_name not in models:
                continue
            
            model_func = models[model_name]
            bounds = default_bounds.get(model_name)
            
            try:
                params = fit_decline_model(
                    model_func, train_t.values, train_data, bounds, fitting_method, lmfit_method
                )
                
                if params is None:
                    continue
                
                # Calculate training metrics
                train_mae, train_mse, train_rmse, train_r_squared, train_adj_r_squared, train_mean_abs_acc = calculate_error_metrics(
                    model_func, train_t.values, train_data, params
                )
                
                # Check fitting accuracy
                if train_adj_r_squared < (fitting_accuracy_threshold / 100):
                    continue
                
                # Calculate test metrics
                test_mae = test_mse = test_rmse = test_r_squared = test_adj_r_squared = test_mean_abs_acc = None
                if len(test_data) > 0:
                    test_mae, test_mse, test_rmse, test_r_squared, test_adj_r_squared, test_mean_abs_acc = calculate_error_metrics(
                        model_func, test_t.values, test_data, params
                    )
                    
                    if test_mean_abs_acc < prediction_accuracy_threshold:
                        continue
                
                # Calculate forecast
                start_forecast_time = t.iloc[end_idx] + 1
                if forecast_method == 1:
                    forecast_t_range = np.arange(start_forecast_time, forecast_end_time + 1)
                    forecast_q = model_func(forecast_t_range, *params)
                else:
                    forecast_t_range = []
                    forecast_q = []
                    current_time = start_forecast_time
                    threshold = forecast_end_flow_rate_oil if well_type == 'oil' else forecast_end_flow_rate_gas
                    while True:
                        forecast_value = model_func(current_time, *params)
                        forecast_t_range.append(current_time)
                        forecast_q.append(forecast_value)
                        
                        if forecast_value <= threshold or current_time >= forecast_end_time:
                            break
                        current_time += 1
                    
                    forecast_t_range = np.array(forecast_t_range)
                    forecast_q = np.array(forecast_q)
                
                # Combine fitted and forecasted
                combined_t = np.concatenate([train_t.values, test_t.values, forecast_t_range])
                combined_q = np.concatenate([
                    model_func(train_t.values, *params),
                    model_func(test_t.values, *params),
                    forecast_q
                ])
                
                # Store combined curve with smoothing method in key
                key = f'{smoothing_method} {model_name}'
                all_combined_curves[key] = {
                    'combined_t': combined_t,
                    'combined_q': combined_q,
                    'forecast_t': forecast_t_range,
                    'forecast_q': forecast_q
                }
                
                # Calculate EUR
                eur = np.trapz(forecast_q, forecast_t_range) if len(forecast_q) > 0 else None
                
                well_results.append({
                    'smoothing_method': smoothing_method,
                    'model': model_name,
                    'params': params,
                    'train_mae': train_mae,
                    'train_mse': train_mse,
                    'train_rmse': train_rmse,
                    'train_r_squared': train_r_squared,
                    'train_adj_r_squared': train_adj_r_squared,
                    'train_mean_abs_acc': train_mean_abs_acc,
                    'test_mae': test_mae,
                    'test_mse': test_mse,
                    'test_rmse': test_rmse,
                    'test_r_squared': test_r_squared,
                    'test_adj_r_squared': test_adj_r_squared,
                    'test_mean_abs_acc': test_mean_abs_acc,
                    'eur': eur,
                    'forecast_t': forecast_t_range,
                    'forecast_q': forecast_q
                })
            except Exception as e:
                # Skip this model if fitting fails
                continue
    
    # Calculate percentiles
    percentiles_data = None
    if len(well_results) > 0 and forecast_method == 1:
        forecast_arrays = []
        start_forecast_time = t.iloc[end_idx] + 1
        max_length = max(len(r['forecast_q']) for r in well_results) if well_results else 0
        
        if max_length > 0:
            common_t_forecast = np.arange(start_forecast_time, start_forecast_time + max_length)
            
            for model_result in well_results:
                forecast_q = model_result['forecast_q']
                if len(forecast_q) < max_length:
                    forecast_q = np.pad(forecast_q, (0, max_length - len(forecast_q)), 
                                      constant_values=forecast_q[-1] if len(forecast_q) > 0 else 0)
                elif len(forecast_q) > max_length:
                    forecast_q = forecast_q[:max_length]
                forecast_arrays.append(forecast_q)
            
            if len(forecast_arrays) > 0:
                forecast_matrix = np.array(forecast_arrays)
                p10 = np.percentile(forecast_matrix, 10, axis=0)
                p50 = np.percentile(forecast_matrix, 50, axis=0)
                p90 = np.percentile(forecast_matrix, 90, axis=0)
                average = np.mean(forecast_matrix, axis=0)
                percentiles_data = (p10, p50, p90, average, common_t_forecast)
    
    return {
        'well_id': well_id,
        't': t.values,
        'q_actual': q_actual,
        'fitting_period': (best_start, best_end),
        'peaks': peaks,
        'troughs': troughs,
        'q_smoothed_lowess': q_smoothed_lowess,
        'well_results': well_results,
        'forecast_curves': all_combined_curves,
        'percentiles': percentiles_data,
        'parameters': {
            'models': selected_models,
            'fitting_method': fitting_method,
            'forecast_method': forecast_method,
            'forecast_end_time': forecast_end_time
        }
    }


def run_batch_processing_all_wells(progress_bar=None, status_text=None) -> Dict[str, any]:
    """
    Run all workflow steps for all wells using default configurations.
    
    Args:
        progress_bar: Optional Streamlit progress bar
        status_text: Optional Streamlit status text element
    
    Returns:
        Dictionary with processing results and statistics
    """
    if not st.session_state.well_list:
        return {'error': 'No wells loaded'}
    
    default_config = get_default_config()
    results = {
        'wells_processed': [],
        'wells_failed': [],
        'step_results': {}
    }
    
    total_wells = len(st.session_state.well_list)
    total_steps = 5
    
    for well_idx, well_id in enumerate(st.session_state.well_list):
        try:
            if status_text:
                status_text.text(f"Processing well {well_idx + 1}/{total_wells}: {well_id}")
            
            # Get well data
            well_data = get_well_data(well_id)
            if well_data is None:
                results['wells_failed'].append({'well_id': well_id, 'error': 'No data found'})
                continue
            
            # Get well type
            file_info = get_file_info(well_id)
            well_type = file_info.get('well_type') or 'oil'
            
            # Step 0: Preprocessing
            step0_config = default_config.get('step0_preprocessing', {})
            step0_result = process_step0_preprocessing(well_id, well_data, step0_config)
            set_step_result('step0', well_id, step0_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 1) / (total_wells * total_steps))
            
            # Step 1: Noise Removal
            step1_config = default_config.get('step1_noise_removal', {})
            step1_result = process_step1_noise_removal(well_id, step0_result['processed_df'], step1_config)
            set_step_result('step1', well_id, step1_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 2) / (total_wells * total_steps))
            
            # Step 2: Interpolation
            step2_config = default_config.get('step2_interpolation', {})
            # Get input from step1 (use cleaned data dict or fallback to output_df)
            if 'cleaned_data_dict' in step1_result and step1_result['cleaned_data_dict']:
                step2_input = list(step1_result['cleaned_data_dict'].values())[0]
            else:
                step2_input = step1_result.get('output_df', step0_result['processed_df'])
            step2_result = process_step2_interpolation(well_id, step2_input, step2_config)
            set_step_result('step2', well_id, step2_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 3) / (total_wells * total_steps))
            
            # Step 3: Smoothing
            step3_config = default_config.get('step3_smoothing', {})
            step3_result = process_step3_smoothing(well_id, step2_result['interpolated_df'], step3_config)
            set_step_result('step3', well_id, step3_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 4) / (total_wells * total_steps))
            
            # Step 4: Fitting
            step4_config = default_config.get('step4_q_fitting', {})
            step4_result = process_step4_fitting(well_id, step3_result['output_df'], step4_config, well_type)
            set_step_result('step4', well_id, step4_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 5) / (total_wells * total_steps))
            
            results['wells_processed'].append(well_id)
            
        except Exception as e:
            results['wells_failed'].append({
                'well_id': well_id,
                'error': str(e)
            })
    
    if status_text:
        status_text.text("Batch processing completed!")
    
    return results


def run_batch_processing_selected_wells(
    well_ids: List[str],
    progress_bar=None, 
    status_text=None
) -> Dict[str, any]:
    """
    Run all workflow steps for selected wells using default configurations.
    
    Args:
        well_ids: List of well IDs to process
        progress_bar: Optional Streamlit progress bar
        status_text: Optional Streamlit status text element
    
    Returns:
        Dictionary with processing results and statistics
    """
    if not well_ids:
        return {'error': 'No wells selected'}
    
    default_config = get_default_config()
    results = {
        'wells_processed': [],
        'wells_failed': [],
        'step_results': {}
    }
    
    total_wells = len(well_ids)
    total_steps = 5
    
    for well_idx, well_id in enumerate(well_ids):
        try:
            if status_text:
                status_text.text(f"Processing well {well_idx + 1}/{total_wells}: {well_id}")
            
            # Get well data
            well_data = get_well_data(well_id)
            if well_data is None:
                results['wells_failed'].append({'well_id': well_id, 'error': 'No data found'})
                continue
            
            # Get well type
            file_info = get_file_info(well_id)
            well_type = file_info.get('well_type') or 'oil'
            
            # Step 0: Preprocessing
            step0_config = default_config.get('step0_preprocessing', {})
            step0_result = process_step0_preprocessing(well_id, well_data, step0_config)
            set_step_result('step0', well_id, step0_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 1) / (total_wells * total_steps))
            
            # Step 1: Noise Removal
            step1_config = default_config.get('step1_noise_removal', {})
            step1_result = process_step1_noise_removal(well_id, step0_result['processed_df'], step1_config)
            set_step_result('step1', well_id, step1_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 2) / (total_wells * total_steps))
            
            # Step 2: Interpolation
            step2_config = default_config.get('step2_interpolation', {})
            # Get input from step1 (use cleaned data dict or fallback to output_df)
            if 'cleaned_data_dict' in step1_result and step1_result['cleaned_data_dict']:
                step2_input = list(step1_result['cleaned_data_dict'].values())[0]
            else:
                step2_input = step1_result.get('output_df', step0_result['processed_df'])
            step2_result = process_step2_interpolation(well_id, step2_input, step2_config)
            set_step_result('step2', well_id, step2_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 3) / (total_wells * total_steps))
            
            # Step 3: Smoothing
            step3_config = default_config.get('step3_smoothing', {})
            step3_result = process_step3_smoothing(well_id, step2_result['interpolated_df'], step3_config)
            set_step_result('step3', well_id, step3_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 4) / (total_wells * total_steps))
            
            # Step 4: Fitting
            step4_config = default_config.get('step4_q_fitting', {})
            step4_result = process_step4_fitting(well_id, step3_result['output_df'], step4_config, well_type)
            set_step_result('step4', well_id, step4_result)
            
            if progress_bar:
                progress_bar.progress((well_idx * total_steps + 5) / (total_wells * total_steps))
            
            results['wells_processed'].append(well_id)
            
        except Exception as e:
            results['wells_failed'].append({
                'well_id': well_id,
                'error': str(e)
            })
    
    if status_text:
        status_text.text("Batch processing completed!")
    
    return results
