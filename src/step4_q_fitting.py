"""
Step 4: Decline Curve Analysis (DCA) and Q-Fitting Module

This module handles:
1. Fitting various DCA models to production data
2. Forecasting future production
3. Calculating EUR (Estimated Ultimate Recovery)
4. Computing P10/P50/P90 percentiles
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution, dual_annealing
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
import re
import glob
from typing import Dict, List, Tuple, Optional
from src.utils.file_handler import get_file_info
from src.utils.visualization import plot_decline_curves, plot_fitting_period


# =============================================================================
# DCA MODEL DEFINITIONS
# =============================================================================

def arps_hyperbolic(t, qi, Di, b):
    """Arps Hyperbolic decline model"""
    with np.errstate(invalid='ignore', over='ignore'):
        result = qi / ((1 + b * Di * t) ** (1 / b))
        result = np.where(np.isfinite(result), result, 0)
        return result


def logistic_growth(t, qi, aLGM, nLGM):
    """Logistic Growth Model"""
    with np.errstate(invalid='ignore', over='ignore'):
        result = qi * (nLGM - aLGM * t ** (nLGM - 1)) / ((aLGM + t ** nLGM) ** 2)
        return np.where(np.isfinite(result), result, 0)


def stretched_exponential(t, qi, tau, n):
    """Stretched Exponential Production Decline model"""
    with np.errstate(invalid='ignore', over='ignore'):
        result = qi * np.exp(-(t / tau) ** n)
        return np.where(np.isfinite(result), result, 0)


def power_law(t, qi, Di, beta):
    """Power Law Exponential model"""
    with np.errstate(invalid='ignore', over='ignore'):
        result = qi * (1 + beta * Di * t) ** (-1 / beta)
        return np.where(np.isfinite(result), result, 0)


def duong(t, qi, aD, mD):
    """Duong model"""
    with np.errstate(invalid='ignore', over='ignore'):
        result = qi * t ** (-mD) * np.exp((aD / (1 - mD)) * (t ** (1 - mD) - 1))
        return np.where(np.isfinite(result), result, 0)


def wang(t, qi, lambda_W):
    """Wang model"""
    with np.errstate(invalid='ignore', over='ignore'):
        result = qi * np.exp(-lambda_W * (np.log(t)) ** 2)
        return np.where(np.isfinite(result), result, 0)


def vdma(t, qi, Di, n_VDMA):
    """Variable Decline Modified Arps (VDMA) model"""
    with np.errstate(invalid='ignore', over='ignore'):
        result = qi * np.exp(-Di * t ** (1 - n_VDMA))
        return np.where(np.isfinite(result), result, 0)


# =============================================================================
# FITTING AND OPTIMIZATION FUNCTIONS
# =============================================================================

def sum_of_squared_errors(parameter_tuple, model_func, t, flow_rate):
    """Calculate sum of squared errors for optimization"""
    warnings.filterwarnings("ignore")
    val = model_func(t, *parameter_tuple)
    return np.sum((flow_rate - val) ** 2.0)


def generate_initial_parameters(model_func, t, flow_rate, bounds):
    """Generate initial parameters using Differential Evolution"""
    result = differential_evolution(sum_of_squared_errors, bounds, 
                                   args=(model_func, t, flow_rate), seed=3)
    return result.x


def fit_decline_model(model_func, time, flow_rate, initial_bounds, method, lmfit_method=None):
    """
    Fit decline curve model to data.
    
    Methods:
        1: curve_fit with differential evolution
        2: dual_annealing
        3: Gaussian Process minimization (requires scikit-optimize)
        4: Particle Swarm Optimization (requires pyswarm)
        5: LMFIT library
    """
    if len(flow_rate) == 0:
        return None
    
    try:
        if method == 1:
            initial_guess = generate_initial_parameters(model_func, time, flow_rate, initial_bounds)
            params, _ = curve_fit(model_func, time, flow_rate, p0=initial_guess, maxfev=20000)
        elif method == 2:
            result = dual_annealing(sum_of_squared_errors, bounds=initial_bounds, 
                                  args=(model_func, time, flow_rate))
            params = result.x
        elif method == 3:
            try:
                from skopt import gp_minimize
                result = gp_minimize(lambda x: sum_of_squared_errors(x, model_func, time, flow_rate), 
                                   initial_bounds)
                params = result.x
            except ImportError:
                print("scikit-optimize not available, falling back to method 1")
                return fit_decline_model(model_func, time, flow_rate, initial_bounds, 1)
        elif method == 4:
            try:
                from pyswarm import pso
                lb, ub = zip(*initial_bounds)
                params, _ = pso(sum_of_squared_errors, lb, ub, args=(model_func, time, flow_rate))
            except ImportError:
                print("pyswarm not available, falling back to method 1")
                return fit_decline_model(model_func, time, flow_rate, initial_bounds, 1)
        elif method == 5:
            try:
                from lmfit import Model
                model = Model(model_func)
                params_obj = model.make_params()
                result = model.fit(flow_rate, params_obj, t=time, method=lmfit_method or 'leastsq')
                params = list(result.best_values.values())
            except ImportError:
                print("lmfit not available, falling back to method 1")
                return fit_decline_model(model_func, time, flow_rate, initial_bounds, 1)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        return params
    except Exception as e:
        print(f"Error fitting model {model_func.__name__}: {e}")
        return None


def calculate_adjusted_r_squared(model_func, time, flow_rate, params):
    """Calculate Adjusted R-squared value"""
    residuals = flow_rate - model_func(time, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((flow_rate - np.mean(flow_rate))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    n = len(flow_rate)
    p = len(params)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else 0
    return adjusted_r_squared


def mean_abs_metric(y_true, y_pred, eps=0.001):
    """Calculate mean absolute percentage accuracy"""
    percentage_diff_list = []
    for idx in range(len(y_true)):
        diff = np.abs(y_true[idx] - y_pred[idx])
        error = np.clip(diff / np.maximum(y_true[idx] + y_pred[idx], eps), 0, 1)
        accuracy = 1 - error
        percentage_diff_list.append(accuracy)
    mean_abs_perc_accur = int(np.round(np.mean(percentage_diff_list) * 100))
    return mean_abs_perc_accur


def calculate_error_metrics(model_func, time, actual, params):
    """Calculate comprehensive error metrics"""
    predicted = model_func(time, *params)
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    n = len(actual)
    k = len(params)
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1)) if n > k + 1 else 0
    
    mean_abs_perc_accur = mean_abs_metric(actual, predicted)
    
    return mae, mse, rmse, r_squared, adj_r_squared, mean_abs_perc_accur


# =============================================================================
# FITTING PERIOD IDENTIFICATION
# =============================================================================

def find_peaks_and_troughs(smoothed_values, min_peak_rel_height=0.2):
    """Find peaks and troughs in production data"""
    peaks, _ = find_peaks(smoothed_values)
    valid_peaks = []
    
    if len(peaks) > 0:
        valid_peaks.append(peaks[0])
        for i in range(1, len(peaks)):
            if smoothed_values[peaks[i]] >= min_peak_rel_height * smoothed_values[valid_peaks[-1]]:
                valid_peaks.append(peaks[i])
    
    max_value_idx = np.argmax(smoothed_values)
    if max_value_idx not in valid_peaks:
        valid_peaks.append(max_value_idx)
    
    troughs, _ = find_peaks(-smoothed_values)
    
    if len(smoothed_values) >= 5:
        last_5_points = smoothed_values[-5:]
        min_last_5_idx = np.argmin(last_5_points) + (len(smoothed_values) - 5)
        if min_last_5_idx not in troughs:
            troughs = np.append(troughs, min_last_5_idx)
    
    if smoothed_values[-1] == np.min(smoothed_values) and len(smoothed_values) > 1:
        if len(troughs) == 0 or troughs[-1] != len(smoothed_values) - 1:
            troughs = np.append(troughs, len(smoothed_values) - 1)
    
    return np.array(valid_peaks), np.array(troughs)


def identify_best_fitting_period(t, smoothed_values, peaks, troughs, min_months=30):
    """Identify the best fitting period for DCA"""
    fitting_periods = []
    peaks = np.sort(peaks)
    
    for i in range(len(peaks) - 1, -1, -1):
        peak_idx = peaks[i]
        following_troughs = troughs[troughs > peak_idx]
        if len(following_troughs) > 0:
            trough_idx = following_troughs[0]
            if (t.iloc[trough_idx] - t.iloc[peak_idx]) > min_months:
                fitting_periods.append((t.iloc[peak_idx], t.iloc[trough_idx]))
    
    if not fitting_periods and len(peaks) > 1:
        for i in range(len(peaks) - 2, -1, -1):
            peak_idx = peaks[i]
            following_troughs = troughs[troughs > peak_idx]
            if len(following_troughs) > 0:
                trough_idx = following_troughs[0]
                fitting_periods.append((t.iloc[peak_idx], t.iloc[trough_idx]))
    
    if not fitting_periods:
        last_peak_idx = peaks[-1] if len(peaks) > 0 else 0
        last_trough_idx = troughs[-1] if len(troughs) > 0 else len(t) - 1
        return t.iloc[last_peak_idx], t.iloc[last_trough_idx]
    
    return max(fitting_periods, key=lambda period: period[1] - period[0])


def validate_fitting_period(t, smoothed_values, q_actual, peaks, troughs, min_months=30):
    """Validate and refine the fitting period"""
    best_start, best_end = identify_best_fitting_period(t, smoothed_values, peaks, troughs, min_months)
    
    attempts = 0
    while not ((best_start in t.iloc[peaks].values) and (best_end in t.iloc[troughs].values)):
        if len(peaks) < 2 or attempts > 10:
            break
        peaks = peaks[:-1]
        best_start, best_end = identify_best_fitting_period(t, smoothed_values, peaks, troughs, min_months)
        attempts += 1
    
    best_start_idx = t[t == best_start].index[0]
    pre_range = q_actual[max(0, best_start_idx - 10):best_start_idx]
    post_range = q_actual[best_start_idx+1:min(len(q_actual), best_start_idx + 11)]
    
    pre_max = pre_range.max() if pre_range.size > 0 else -np.inf
    post_max = post_range.max() if post_range.size > 0 else -np.inf
    
    if q_actual[best_start_idx] < pre_max or q_actual[best_start_idx] < post_max:
        if pre_max > post_max:
            best_start = t.iloc[pre_range.argmax() + max(0, best_start_idx - 10)]
        else:
            best_start = t.iloc[post_range.argmax() + best_start_idx + 1]
    
    return best_start, best_end


# =============================================================================
# PERCENTILE CALCULATIONS
# =============================================================================

def calculate_percentiles(forecasted_curves):
    """Calculate P10, P50, P90 percentiles"""
    forecasted_values = np.array(list(forecasted_curves.values()))
    if forecasted_values.size == 0:
        raise ValueError("No forecasted values available")
    
    p10 = np.percentile(forecasted_values, 10, axis=0)
    p50 = np.percentile(forecasted_values, 50, axis=0)
    p90 = np.percentile(forecasted_values, 90, axis=0)
    
    return p10, p50, p90


def calculate_average_curve(forecasted_curves):
    """Calculate average forecasted curve"""
    forecasted_values = np.array(list(forecasted_curves.values()))
    if forecasted_values.size == 0:
        raise ValueError("No forecasted values available")
    return np.mean(forecasted_values, axis=0)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def run_q_fitting(config: Dict) -> Dict[str, any]:
    """
    Main function to run DCA fitting and forecasting.
    
    Args:
        config: Configuration dictionary for step 4
    
    Returns:
        Results dictionary with fitted models and forecasts
    """
    print("=" * 60)
    print("STEP 4: DECLINE CURVE ANALYSIS & Q-FITTING")
    print("=" * 60)
    
    # Extract configuration
    input_directory = config['input_directory']
    output_directory = config['output_directory']
    save_intermediate = config.get('save_intermediate', True)
    
    # Forecast settings
    forecast_method = config.get('forecast_method', 1)
    forecast_end_time = config.get('forecast_end_time', 400)
    forecast_end_flow_rate_gas = config.get('forecast_end_flow_rate_gas', 100)
    forecast_end_flow_rate_oil = config.get('forecast_end_flow_rate_oil', 3)
    
    # Fitting settings
    fitting_method = config.get('fitting_method', 1)
    lmfit_method = config.get('lmfit_method', 'leastsq')
    
    # Model selection
    model_names = config.get('models', ['Arps_Hyperbolic', 'Logistic_Growth', 'Stretched_Exponential', 
                                       'Power_Law', 'Duong', 'Wang', 'VDMA'])
    
    # Thresholds
    fitting_accuracy_threshold = config.get('fitting_accuracy_threshold', 90) / 100
    prediction_accuracy_threshold = config.get('prediction_accuracy_threshold', 80)
    minimum_production_history = config.get('minimum_production_history', 3)
    
    # Peak/trough detection parameters
    min_peak_rel_height = config.get('min_peak_rel_height', 0.2)
    lowess_frac = config.get('lowess_frac', 0.30)
    min_months = config.get('min_months', 30)
    
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
    
    # Define DCA models
    models = {
        'Arps_Hyperbolic': arps_hyperbolic,
        'Logistic_Growth': logistic_growth,
        'Stretched_Exponential': stretched_exponential,
        'Power_Law': power_law,
        'Duong': duong,
        'Wang': wang,
        'VDMA': vdma,
    }
    
    # Find all CSV files to process - accept any CSV
    all_csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
    
    # Skip files that are output from THIS step
    csv_files = [f for f in all_csv_files if '_dca_models' not in os.path.splitext(os.path.basename(f))[0]]
    
    # Filter by wells if specified
    if wells_to_process and wells_to_process != ['all']:
        csv_files = [f for f in csv_files if any(well_id in f for well_id in wells_to_process)]
    
    print(f"\nFound {len(csv_files)} files to process")
    
    results = {
        'step': 'step4_q_fitting',
        'total_files': len(csv_files),
        'wells_with_valid_model': 0,
        'processed_files': [],
        'summary_data': [],
        'eur_data': [],
        'errors': []
    }
    
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            print(f"\n{'='*60}")
            print(f"Processing: {filename}")
            
            data = pd.read_csv(file_path)
            
            if 't' not in data.columns or 'q_actual' not in data.columns:
                print(f"  Skipping - missing required columns")
                continue
            
            t = pd.Series(data['t'].values)
            q_actual = data['q_actual'].values
            
            # Apply LOWESS smoothing for peak detection
            q_smoothed_lowess = lowess(q_actual, t, frac=lowess_frac, return_sorted=False)
            
            # Find peaks and troughs
            peaks, troughs = find_peaks_and_troughs(q_smoothed_lowess, min_peak_rel_height)
            
            # Validate fitting period
            best_start, best_end = validate_fitting_period(t, q_smoothed_lowess, q_actual, 
                                                          peaks, troughs, min_months)
            
            start_idx = t[t == best_start].index[0]
            end_idx = t[t == best_end].index[0]
            
            print(f"  Fitting period: t={best_start} to t={best_end}")
            
            # Generate fitting period plot if visualization is enabled (before processing models)
            if visualize_wells and figures_dir:
                should_visualize = (visualize_wells == ['all'] or 
                                  any(well_id in filename for well_id in visualize_wells))
                
                if should_visualize:
                    base_filename = os.path.splitext(filename)[0]
                    file_info = get_file_info(filename)
                    well_type = file_info.get('well_type', 'oil')
                    
                    # Create fitting period plot
                    fig_filename_fitting = f"{base_filename}_fitting_period.{figure_format}"
                    fig_path_fitting = os.path.join(figures_dir, fig_filename_fitting)
                    
                    plot_fitting_period(t, pd.Series(q_actual), q_smoothed_lowess,
                                      peaks, troughs, best_start, best_end,
                                      title=f'Best Fitting Period for {filename}',
                                      save_path=fig_path_fitting,
                                      show=False)
            
            # Process each smoothed column
            smoothed_columns = [col for col in data.columns if col not in ['t', 'q_actual']]
            well_has_valid_model = False
            
            well_results_data = []
            forecast_curves = {}  # Store forecast curves for visualization
            all_combined_curves = {}  # Store combined fitted+forecast curves
            
            for col in smoothed_columns[:1]:  # Process first smoothed column for now
                data_values = data[col].values
                fitting_t = t[start_idx:end_idx+1]
                fitting_data = data_values[start_idx:end_idx+1]
                
                # Split into train/test
                split_index = int(len(fitting_t) * 0.8)
                train_t = fitting_t[:split_index]
                train_data = fitting_data[:split_index]
                test_t = fitting_t[split_index:]
                test_data = fitting_data[split_index:]
                
                if len(train_data) < minimum_production_history:
                    print(f"  Insufficient data for {col}")
                    continue
                
                qi_initial = np.max(fitting_data)
                
                # Model bounds - use default bounds calculated from data
                # Note: config's model_bounds are for documentation only
                default_bounds = {
                    'Arps_Hyperbolic': [(qi_initial*0.5, qi_initial*5), (0.0001, 0.1), (0.0001, 4)],
                    'Logistic_Growth': [(qi_initial*1000, qi_initial*15000), (10, 1000), (0.00001, 1.5)],
                    'Stretched_Exponential': [(qi_initial*0.5, qi_initial*1.5), (0.001, 2), (0.001, 2)],
                    'Power_Law': [(qi_initial*0.5, qi_initial*1.5), (0.0001, 2), (0.0001, 2)],
                    'Duong': [(qi_initial*0.5, qi_initial*1.5), (0.01, 10), (0.5, 1.5)],
                    'Wang': [(qi_initial*0.5, qi_initial*1.5), (0.0001, 10)],
                    'VDMA': [(qi_initial*0.5, qi_initial*1.5), (0.00001, 2), (0.1, 2)],
                }
                
                # Fit models
                for model_name in model_names:
                    if model_name not in models:
                        continue
                    
                    model_func = models[model_name]
                    # Use default bounds (always calculated from data)
                    bounds = default_bounds.get(model_name, default_bounds['Arps_Hyperbolic'])
                    
                    params = fit_decline_model(model_func, train_t.values, train_data, 
                                              bounds, fitting_method, lmfit_method)
                    
                    if params is None:
                        continue
                    
                    # Calculate training metrics (goodness of fit)
                    train_mae, train_mse, train_rmse, train_r_squared, train_adj_r_squared, train_mean_abs_acc = calculate_error_metrics(
                        model_func, train_t.values, train_data, params)
                    
                    # Check fitting accuracy
                    if train_adj_r_squared < fitting_accuracy_threshold:
                        continue
                    
                    # Calculate test metrics (prediction accuracy) if test data available
                    test_mae = test_mse = test_rmse = test_r_squared = test_adj_r_squared = test_mean_abs_acc = None
                    if len(test_data) > 0:
                        test_mae, test_mse, test_rmse, test_r_squared, test_adj_r_squared, test_mean_abs_acc = calculate_error_metrics(
                            model_func, test_t.values, test_data, params)
                        
                        if test_mean_abs_acc < prediction_accuracy_threshold:
                            continue
                    
                    print(f"  [OK] {col} {model_name}: R2={train_adj_r_squared:.3f}")
                    well_has_valid_model = True
                    
                    # Calculate forecast
                    start_forecast_time = t.iloc[end_idx] + 1
                    if forecast_method == 1:
                        forecast_t_range = np.arange(start_forecast_time, forecast_end_time + 1)
                        forecast_q = model_func(forecast_t_range, *params)
                    elif forecast_method == 2:
                        forecast_t_range = []
                        forecast_q = []
                        current_time = start_forecast_time
                        while True:
                            forecast_value = model_func(current_time, *params)
                            forecast_t_range.append(current_time)
                            forecast_q.append(forecast_value)
                            
                            # Check stopping condition
                            if 'oil' in filename.lower():
                                if forecast_value <= forecast_end_flow_rate_oil or current_time >= forecast_end_time:
                                    break
                            elif 'gas' in filename.lower():
                                if forecast_value <= forecast_end_flow_rate_gas or current_time >= forecast_end_time:
                                    break
                            else:
                                if current_time >= forecast_end_time:
                                    break
                            
                            current_time += 1
                        
                        forecast_t_range = np.array(forecast_t_range)
                        forecast_q = np.array(forecast_q)
                    
                    # Combine fitted and forecasted data
                    combined_t = np.concatenate([train_t.values, test_t.values, forecast_t_range])
                    combined_q = np.concatenate([
                        model_func(train_t.values, *params),
                        model_func(test_t.values, *params),
                        forecast_q
                    ])
                    
                    # Store combined curve for visualization
                    key = f'{col} {model_name}'
                    all_combined_curves[key] = {
                        'combined_t': combined_t,
                        'combined_q': combined_q,
                        'forecast_t': forecast_t_range,
                        'forecast_q': forecast_q
                    }
                    
                    # Calculate EUR (sum of forecast to end time)
                    if len(forecast_q) > 0:
                        eur = np.trapz(forecast_q, forecast_t_range)  # Trapezoidal integration
                    else:
                        eur = None
                    
                    # Store model info with EUR and all metrics
                    well_results_data.append({
                        'model': model_name,
                        'params': params,
                        # Training metrics (goodness of fit)
                        'train_mae': train_mae,
                        'train_mse': train_mse,
                        'train_rmse': train_rmse,
                        'train_r_squared': train_r_squared,
                        'train_adj_r_squared': train_adj_r_squared,
                        'train_mean_abs_acc': train_mean_abs_acc,
                        # Test metrics (prediction accuracy) - may be None if no test data
                        'test_mae': test_mae,
                        'test_mse': test_mse,
                        'test_rmse': test_rmse,
                        'test_r_squared': test_r_squared,
                        'test_adj_r_squared': test_adj_r_squared,
                        'test_mean_abs_acc': test_mean_abs_acc,
                        # Legacy field for backward compatibility
                        'adj_r_squared': train_adj_r_squared,
                        'eur': eur,
                        'forecast_t': forecast_t_range,
                        'forecast_q': forecast_q
                    })
            
            if well_has_valid_model:
                results['wells_with_valid_model'] += 1
                results['processed_files'].append({
                    'filename': filename,
                    'models': well_results_data
                })
                
                # Save results to CSV if save_intermediate is True
                if save_intermediate:
                    # Create results dataframe with historical data
                    results_df = pd.DataFrame({
                        't': t.values,
                        'q_actual': q_actual
                    })
                    
                    # Add fitted curves for each model (historical only)
                    for model_result in well_results_data:
                        model_name = model_result['model']
                        params = model_result['params']
                        model_func = models[model_name]
                        
                        # Generate fitted curve for historical data
                        q_fitted = model_func(t.values, *params)
                        results_df[f'q_fitted_{model_name}'] = q_fitted
                    
                    # Add forecast if configured
                    if forecast_end_time and forecast_end_time > t.max():
                        forecast_t = np.arange(t.max() + 1, forecast_end_time + 1)
                        
                        # Create forecast dataframe
                        forecast_data = {'t': forecast_t, 'q_actual': np.nan}
                        
                        # Add forecast for each model
                        for model_result in well_results_data:
                            model_name = model_result['model']
                            params = model_result['params']
                            model_func = models[model_name]
                            q_forecast = model_func(forecast_t, *params)
                            forecast_data[f'q_fitted_{model_name}'] = q_forecast
                        
                        forecast_df = pd.DataFrame(forecast_data)
                        
                        # Concatenate historical and forecast
                        results_df = pd.concat([results_df, forecast_df], ignore_index=True)
                    
                    # Save to output directory
                    base_filename = os.path.splitext(filename)[0]
                    output_filename = f"{base_filename}_dca_models.csv"
                    output_path = os.path.join(output_directory, output_filename)
                    results_df.to_csv(output_path, index=False)
                    print(f"  Saved: {output_filename}")
                    
                    # Generate visualization if enabled
                    if visualize_wells and figures_dir:
                        # Check if this well should be visualized
                        should_visualize = (visualize_wells == ['all'] or 
                                          any(well_id in base_filename for well_id in visualize_wells))
                        
                        if should_visualize:
                            # Determine well type
                            file_info = get_file_info(filename)
                            well_type = file_info.get('well_type', 'oil')
                            
                            # Calculate percentiles if we have forecasts
                            percentiles_data = None
                            if len(all_combined_curves) > 0 and forecast_method == 1 and len(well_results_data) > 0:
                                # Collect all forecast curves - use forecast_q from well_results_data
                                forecast_arrays = []
                                common_start = start_forecast_time
                                
                                # Find max forecast length
                                max_length = 0
                                for model_result in well_results_data:
                                    if 'forecast_q' in model_result and len(model_result['forecast_q']) > max_length:
                                        max_length = len(model_result['forecast_q'])
                                
                                if max_length > 0:
                                    common_t_forecast = np.arange(common_start, common_start + max_length)
                                    
                                    for model_result in well_results_data:
                                        if 'forecast_q' in model_result:
                                            forecast_q = model_result['forecast_q']
                                            # Pad or trim to match common length
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
                            
                            # Create decline curve plot
                            fig_filename = f"{base_filename}_decline_curves.{figure_format}"
                            fig_path = os.path.join(figures_dir, fig_filename)
                            
                            plot_decline_curves(t.values, q_actual,
                                              fitted_models=None,
                                              fitting_period=(best_start, best_end),
                                              forecast_curves=all_combined_curves,
                                              percentiles=percentiles_data,
                                              title=f'Decline Curve Forecasts for {filename}',
                                              well_type=well_type,
                                              save_path=fig_path,
                                              show=False)
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f"  {error_msg}")
            results['errors'].append(error_msg)
    
    print(f"\n{'=' * 60}")
    print(f"STEP 4 COMPLETE")
    print(f"Wells with valid models: {results['wells_with_valid_model']}/{results['total_files']}")
    print(f"Total errors: {len(results['errors'])}")
    print(f"{'=' * 60}\n")
    
    # Save summary file if requested
    save_summary = config.get('save_summary', True)
    summary_metrics = config.get('summary_metrics', ['r_squared'])
    if save_summary and len(results['processed_files']) > 0:
        print("Generating summary statistics...")
        summary_rows = []
        
        # Map config metric names to data keys
        # Note: 'r_squared' is handled specially for backward compatibility (maps to train_adj_r_squared)
        metric_mapping = {
            'mae': ('train_mae', 'test_mae'),
            'mse': ('train_mse', 'test_mse'),
            'rmse': ('train_rmse', 'test_rmse'),
            'adj_r_squared': ('train_adj_r_squared', 'test_adj_r_squared'),
            'mean_abs_acc': ('train_mean_abs_acc', 'test_mean_abs_acc'),
        }
        
        for well_data in results['processed_files']:
            filename = well_data['filename']
            well_id = os.path.splitext(filename)[0]
            
            # Collect all EUR values for P10/P50/P90 calculation
            eur_values = [m['eur'] for m in well_data['models'] if m.get('eur') is not None]
            
            # Add a row for each model
            for model_data in well_data['models']:
                model_name = model_data['model']
                params = model_data['params']
                eur = model_data.get('eur')
                
                row = {
                    'well_id': well_id,
                    'model': model_name,
                    'eur': eur if eur is not None else np.nan,
                }
                
                # Handle special case: 'r_squared' in config maps to train_adj_r_squared for backward compatibility
                if 'r_squared' in summary_metrics:
                    row['r_squared'] = model_data.get('train_adj_r_squared', np.nan)
                
                # Add requested metrics
                for metric_name in summary_metrics:
                    if metric_name == 'r_squared':
                        # Already handled above for backward compatibility
                        continue
                    elif metric_name in metric_mapping:
                        train_key, test_key = metric_mapping[metric_name]
                        # Add training metric
                        row[train_key] = model_data.get(train_key, np.nan)
                        # Add test metric
                        row[test_key] = model_data.get(test_key, np.nan)
                    else:
                        # Unknown metric name, skip
                        print(f"  Warning: Unknown metric '{metric_name}' in summary_metrics, skipping")
                
                # Add parameter values
                param_names = {
                    'Arps_Hyperbolic': ['qi', 'Di', 'b'],
                    'Logistic_Growth': ['qi', 'aLGM', 'nLGM'],
                    'Stretched_Exponential': ['qi', 'tau', 'n'],
                    'Power_Law': ['qi', 'Di', 'beta'],
                    'Duong': ['qi', 'aD', 'mD'],
                    'Wang': ['qi', 'lambda_W'],
                    'VDMA': ['qi', 'Di', 'n_VDMA'],
                }
                
                if model_name in param_names:
                    for i, param_name in enumerate(param_names[model_name]):
                        if i < len(params):
                            row[param_name] = params[i]
                
                summary_rows.append(row)
            
            # Add a summary row with P10/P50/P90 for the well
            if len(eur_values) > 0:
                p10 = np.percentile(eur_values, 10)
                p50 = np.percentile(eur_values, 50)
                p90 = np.percentile(eur_values, 90)
                
                stats_row = {
                    'well_id': well_id,
                    'model': 'STATISTICS',
                    'eur': p50,  # Median EUR
                    'eur_p10': p10,
                    'eur_p50': p50,
                    'eur_p90': p90,
                }
                
                # Handle special case: 'r_squared' in config maps to train_adj_r_squared for backward compatibility
                if 'r_squared' in summary_metrics:
                    stats_row['r_squared'] = np.nan
                
                # Add requested metrics (set to NaN for statistics row)
                for metric_name in summary_metrics:
                    if metric_name == 'r_squared':
                        # Already handled above for backward compatibility
                        continue
                    elif metric_name in metric_mapping:
                        train_key, test_key = metric_mapping[metric_name]
                        stats_row[train_key] = np.nan
                        stats_row[test_key] = np.nan
                
                summary_rows.append(stats_row)
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary file
        summary_path = os.path.join(output_directory, 'dca_summary_all_wells.csv')
        summary_df.to_csv(summary_path, index=False)
        print("Saved summary: dca_summary_all_wells.csv")
        print(f"  Total wells: {results['wells_with_valid_model']}")
        print(f"  Total model fits: {len(summary_rows) - results['wells_with_valid_model']}")  # Subtract stat rows
    
    return results


if __name__ == "__main__":
    # Example usage
    config = {
        'input_directory': 'output/step3/',
        'output_directory': 'output/step4/',
        'forecast_method': 1,
        'forecast_end_time': 400,
        'fitting_method': 1,
        'models': ['Arps_Hyperbolic', 'Power_Law', 'Duong'],
        'save_intermediate': True
    }
    
    results = run_q_fitting(config)
