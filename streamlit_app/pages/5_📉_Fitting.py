"""
Step 4: DCA Fitting
Fit decline curve models and forecast production
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from typing import Dict, List, Tuple

# Add project root and src to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from streamlit_app.utils.session_state import (
    get_current_well_data, get_step_result, set_step_result, has_step_result, get_step_result as get_prev_step_result,
    get_adjusted_fitting_period, set_adjusted_fitting_period, clear_adjusted_fitting_period
)
from streamlit_app.components.well_selector import render_well_selector
from streamlit_app.utils.streamlit_helpers import (
    create_plotly_dca_forecast, create_fitting_period_plot, export_dataframe
)
from streamlit_app.config.default_config import get_default_config
from src.utils.file_handler import get_file_info

# Import DCA functions
from step4_q_fitting import (
    arps_hyperbolic, logistic_growth, stretched_exponential, power_law,
    duong, wang, vdma, fit_decline_model, calculate_error_metrics,
    find_peaks_and_troughs, validate_fitting_period, identify_best_fitting_period,
    mean_abs_metric
)
from statsmodels.nonparametric.smoothers_lowess import lowess

# Page config
st.set_page_config(page_title="DCA Fitting", layout="wide")

st.title("📉 Step 4: Decline Curve Analysis & Fitting")
st.markdown("Fit DCA models, forecast production, and calculate EUR (Estimated Ultimate Recovery).")

# Check prerequisites
if not st.session_state.well_list:
    st.warning("⚠️ No wells loaded. Please upload CSV files from the main page.")
    st.stop()

# Well selector
selected_well = render_well_selector(st.session_state.well_list, key="step4_well_selector")

if not selected_well:
    st.stop()

# Check if Step 3 is completed (prefer Step 3, fallback to earlier steps)
step3_result = get_prev_step_result('step3', selected_well)
step2_result = get_prev_step_result('step2', selected_well)
step1_result = get_prev_step_result('step1', selected_well)
step0_result = get_prev_step_result('step0', selected_well)

# Get input data
input_df = None
smoothed_column = None

if step3_result:
    input_df = step3_result['output_df']
    # Get all smoothed columns
    smoothed_columns = [col for col in input_df.columns if col not in ['t', 'q_actual']]
    if smoothed_columns:
        smoothed_column = smoothed_columns[0]  # Keep for backward compatibility
    else:
        smoothed_columns = []  # Will use raw q_actual
elif step2_result:
    input_df = step2_result['interpolated_df']
elif step1_result:
    for algo_name, cleaned_df in step1_result['cleaned_data_dict'].items():
        input_df = cleaned_df
        break
elif step0_result:
    input_df = step0_result['processed_df']
else:
    st.warning("⚠️ Please complete Step 0 (Raw Data) first.")
    st.stop()

if input_df is None:
    st.error("No input data available")
    st.stop()

# Determine well type
file_info = get_file_info(selected_well)
well_type = file_info.get('well_type', 'oil')

# Parameter controls in sidebar
st.sidebar.header("⚙️ Fitting Period Detection")

default_config = get_default_config()
step4_config = default_config.get('step4_q_fitting', {})

min_months = st.sidebar.number_input(
    "Minimum Months",
    min_value=10,
    max_value=100,
    value=step4_config.get('min_months', 30),
    help="Minimum number of months for valid fitting period"
)

lowess_frac = st.sidebar.slider(
    "LOWESS Fraction",
    min_value=0.1,
    max_value=0.5,
    value=step4_config.get('lowess_frac', 0.30),
    step=0.01,
    help="Fraction of data for LOWESS smoothing in peak detection"
)

min_peak_rel_height = st.sidebar.slider(
    "Min Peak Relative Height",
    min_value=0.1,
    max_value=0.5,
    value=step4_config.get('min_peak_rel_height', 0.2),
    step=0.01,
    help="Minimum relative height for peak detection"
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Selection")

available_models = [
    'Arps_Hyperbolic', 'Logistic_Growth', 'Stretched_Exponential',
    'Power_Law', 'Duong', 'Wang', 'VDMA'
]

selected_models = st.sidebar.multiselect(
    "DCA Models",
    options=available_models,
    default=step4_config.get('models', available_models),
    help="Select decline curve models to fit"
)

st.sidebar.markdown("---")
st.sidebar.header("🔧 Fitting Configuration")

fitting_method = st.sidebar.selectbox(
    "Fitting Method",
    options=[1, 2, 3, 4, 5],
    index=0,
    format_func=lambda x: {
        1: "1: curve_fit + differential_evolution",
        2: "2: dual_annealing",
        3: "3: Bayesian (GP)",
        4: "4: PSO",
        5: "5: lmfit"
    }[x],
    help="Optimization algorithm for model fitting"
)

lmfit_method = None
if fitting_method == 5:
    lmfit_method = st.sidebar.selectbox(
        "LMFIT Method",
        options=['leastsq', 'nelder', 'powell', 'cg', 'bfgs', 'lbfgsb', 'tnc', 'cobyla', 'slsqp', 'differential_evolution'],
        index=0,
        help="Specific method for lmfit optimization"
    )

minimum_production_history = st.sidebar.number_input(
    "Minimum Production History",
    min_value=3,
    max_value=20,
    value=step4_config.get('minimum_production_history', 3),
    help="Minimum data points required for fitting"
)

fitting_accuracy_threshold = st.sidebar.slider(
    "Fitting Accuracy Threshold (%)",
    min_value=50,
    max_value=100,
    value=step4_config.get('fitting_accuracy_threshold', 90),
    help="Minimum adjusted R² for accepting model"
)

prediction_accuracy_threshold = st.sidebar.slider(
    "Prediction Accuracy Threshold (%)",
    min_value=50,
    max_value=100,
    value=step4_config.get('prediction_accuracy_threshold', 80),
    help="Minimum accuracy on test data"
)

st.sidebar.markdown("---")
st.sidebar.header("📈 Forecast Configuration")

forecast_method = st.sidebar.selectbox(
    "Forecast Method",
    options=[1, 2],
    index=0,
    format_func=lambda x: {
        1: "1: Forecast to fixed end time",
        2: "2: Forecast until flow rate threshold"
    }[x]
)

forecast_end_time = st.sidebar.number_input(
    "Forecast End Time (months)",
    min_value=100,
    max_value=1000,
    value=step4_config.get('forecast_end_time', 400)
)

forecast_end_flow_rate_oil = st.sidebar.number_input(
    "Oil Flow Rate Threshold (STB/D)",
    min_value=0.1,
    max_value=10.0,
    value=float(step4_config.get('forecast_end_flow_rate_oil', 3.0)),
    step=0.1
)

forecast_end_flow_rate_gas = st.sidebar.number_input(
    "Gas Flow Rate Threshold (MSCF/D)",
    min_value=10.0,
    max_value=500.0,
    value=float(step4_config.get('forecast_end_flow_rate_gas', 100.0)),
    step=10.0
)

# Check for existing results
has_results = has_step_result('step4', selected_well)
if has_results:
    st.sidebar.success("✅ Results available")
    if st.sidebar.button("🔄 Re-run Fitting"):
        has_results = False

# Process button
process_btn = st.sidebar.button("▶️ Run DCA Fitting", type="primary", use_container_width=True)

# Main processing
if process_btn and selected_models:
    with st.spinner("Fitting DCA models... This may take several minutes."):
        try:
            t = pd.Series(input_df['t'].values)
            q_actual = input_df['q_actual'].values
            
            # Apply LOWESS smoothing for peak detection (always use q_actual for peak detection)
            q_smoothed_lowess = lowess(q_actual, t, frac=lowess_frac, return_sorted=False)
            
            # Find peaks and troughs
            peaks, troughs = find_peaks_and_troughs(q_smoothed_lowess, min_peak_rel_height)
            
            # Check for adjusted fitting period
            adjusted_period = get_adjusted_fitting_period(selected_well)
            if adjusted_period:
                best_start, best_end = adjusted_period
                # Validate that adjusted period is within data range
                if best_start < t.min() or best_end > t.max() or best_start >= best_end:
                    st.warning("Adjusted period is invalid. Using auto-detected period.")
                    adjusted_period = None
            
            if not adjusted_period:
                # Validate fitting period
                best_start, best_end = validate_fitting_period(
                    t, q_smoothed_lowess, q_actual, peaks, troughs, min_months
                )
            
            # Find indices for the fitting period
            start_idx = t[t >= best_start].index[0] if (t >= best_start).any() else t.index[0]
            end_idx = t[t <= best_end].index[-1] if (t <= best_end).any() else t.index[-1]
            
            # Determine which smoothed columns to use
            # If step3_result exists, use all smoothing methods; otherwise use raw q_actual
            smoothing_methods_to_fit = []
            if step3_result and smoothed_columns:
                smoothing_methods_to_fit = smoothed_columns
            else:
                smoothing_methods_to_fit = ['raw']  # Use raw q_actual
            
            # Model bounds (will be recalculated per smoothing method)
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
            forecast_curves = {}
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
            
            # Calculate percentiles
            percentiles_data = None
            if len(well_results) > 0 and forecast_method == 1:
                forecast_arrays = []
                common_start = start_forecast_time
                max_length = max(len(r['forecast_q']) for r in well_results)
                
                if max_length > 0:
                    common_t_forecast = np.arange(common_start, common_start + max_length)
                    
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
            
            # Store results
            result = {
                'well_id': selected_well,
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
            
            set_step_result('step4', selected_well, result)
            st.success(f"✅ DCA fitting completed! {len(well_results)} model(s) fitted successfully.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during DCA fitting: {str(e)}")
            st.exception(e)

# Display results
if has_results:
    result = get_step_result('step4', selected_well)
    if result:
        # Check if result has required fields
        if 't' not in result or 'fitting_period' not in result:
            st.warning("⚠️ Incomplete results found. Please re-run fitting for this well.")
        else:
            # Fitting period adjustment UI
            st.subheader("✏️ Fitting Period Editor")
            edit_period_mode = st.toggle("Enable Period Adjustment", key="edit_period_toggle",
                                        help="Enable to manually adjust the fitting period")
            
            auto_start = result['fitting_period'][0]
            auto_end = result['fitting_period'][1]
            adjusted_period = get_adjusted_fitting_period(selected_well)
            
            if edit_period_mode:
                col_period1, col_period2 = st.columns(2)
                
                with col_period1:
                    period_start = st.number_input(
                        "Start Time (months)",
                        min_value=float(result['t'].min()),
                        max_value=float(result['t'].max()),
                        value=float(adjusted_period[0]) if adjusted_period else float(auto_start),
                        step=1.0,
                        key="period_start_input",
                        help="Start time of fitting period"
                    )
                
                with col_period2:
                    period_end = st.number_input(
                        "End Time (months)",
                        min_value=float(result['t'].min()),
                        max_value=float(result['t'].max()),
                        value=float(adjusted_period[1]) if adjusted_period else float(auto_end),
                        step=1.0,
                        key="period_end_input",
                        help="End time of fitting period"
                    )
                
                if period_start >= period_end:
                    st.error("Start time must be less than end time")
                else:
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        if st.button("✅ Apply Adjusted Period", type="primary", key="apply_period_btn"):
                            set_adjusted_fitting_period(selected_well, (period_start, period_end))
                            st.success("Period adjusted! Click 'Re-run Fitting' to apply changes.")
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("🔄 Re-run Fitting with Adjusted Period", key="refit_btn"):
                            # Trigger refitting with adjusted period
                            st.info("Refitting with adjusted period...")
                            # The refitting will happen below
                            pass
                    
                    with col_btn3:
                        if st.button("↩️ Reset to Auto-Detected", key="reset_period_btn"):
                            clear_adjusted_fitting_period(selected_well)
                            st.success("Reset to auto-detected period")
                            st.rerun()
                
                st.markdown("---")
            
            # Fitting period visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎯 Fitting Period Detection")
                fig_fitting = create_fitting_period_plot(
                    pd.Series(result['t']),
                    pd.Series(result['q_actual']),
                    result['q_smoothed_lowess'],
                    result['peaks'],
                    result['troughs'],
                    auto_start,
                    auto_end,
                    adjusted_period=adjusted_period
                )
                st.plotly_chart(fig_fitting, use_container_width=True)
                
                current_period = adjusted_period if adjusted_period else (auto_start, auto_end)
                period_label = "Adjusted" if adjusted_period else "Auto-Detected"
                st.info(f"**{period_label} Fitting Period**: {current_period[0]:.1f} - {current_period[1]:.1f} months")
        
            with col2:
                st.subheader("📊 Fitting Period Info")
                current_period = adjusted_period if adjusted_period else (auto_start, auto_end)
                period_length = current_period[1] - current_period[0]
                st.metric("Period Length", f"{period_length:.1f} months")
                st.metric("Start Time", f"{current_period[0]:.1f} months")
                st.metric("End Time", f"{current_period[1]:.1f} months")
                st.metric("Models Fitted", len(result.get('well_results', [])))
                if adjusted_period:
                    st.info("⚠️ Using adjusted period")
            
            # Main forecast visualization
            st.subheader("📈 Decline Curve Forecasts")
            # Use adjusted period if available, otherwise use stored fitting period
            forecast_period = adjusted_period if adjusted_period else result['fitting_period']
            fig_forecast = create_plotly_dca_forecast(
                result['t'],
                result['q_actual'],
                result.get('forecast_curves', {}),
                forecast_period,
                result.get('percentiles'),
                well_type=well_type
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Summary table
            st.subheader("📊 Model Summary")
            
            if result.get('well_results'):
                summary_data = []
                eur_values = []
                
                param_names = {
                    'Arps_Hyperbolic': ['qi', 'Di', 'b'],
                    'Logistic_Growth': ['qi', 'aLGM', 'nLGM'],
                    'Stretched_Exponential': ['qi', 'tau', 'n'],
                    'Power_Law': ['qi', 'Di', 'beta'],
                    'Duong': ['qi', 'aD', 'mD'],
                    'Wang': ['qi', 'lambda_W'],
                    'VDMA': ['qi', 'Di', 'n_VDMA'],
                }
                
                for model_result in result['well_results']:
                    model_name = model_result['model']
                    smoothing_method = model_result.get('smoothing_method', 'raw')
                    params = model_result['params']
                    eur = model_result.get('eur')
                    if eur:
                        eur_values.append(eur)
                    
                    row = {
                        'Smoothing': smoothing_method,
                        'Model': model_name,
                        'EUR': f"{eur:.2f}" if eur else "N/A",
                        'R² (Adj)': f"{model_result['train_adj_r_squared']:.4f}",
                        'Prediction Acc (%)': f"{model_result.get('test_mean_abs_acc', 'N/A')}"
                    }
                    
                    # Add parameters
                    if model_name in param_names:
                        for i, param_name in enumerate(param_names[model_name]):
                            if i < len(params):
                                row[param_name] = f"{params[i]:.4f}"
                    
                    summary_data.append(row)
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # P10/P50/P90 statistics
                if eur_values:
                    st.subheader("📊 EUR Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    p10 = np.percentile(eur_values, 10)
                    p50 = np.percentile(eur_values, 50)
                    p90 = np.percentile(eur_values, 90)
                    mean_eur = np.mean(eur_values)
                    
                    col1.metric("P10 EUR", f"{p10:.2f}")
                    col2.metric("P50 EUR", f"{p50:.2f}")
                    col3.metric("P90 EUR", f"{p90:.2f}")
                    col4.metric("Mean EUR", f"{mean_eur:.2f}")
                
                # Export button
                csv_data = export_dataframe(summary_df)
                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=csv_data,
                    file_name=f"{selected_well}_dca_summary.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No models were successfully fitted. Try adjusting parameters.")
else:
    st.info("👆 Configure parameters and click 'Run DCA Fitting' to process the data")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    ### DCA Fitting Step
    
    This step fits decline curve models and forecasts production:
    
    1. **Fitting Period Detection**: Identifies optimal period for model fitting
    2. **Model Fitting**: Fits selected DCA models to historical data
    3. **Forecasting**: Projects future production
    4. **EUR Calculation**: Estimates Ultimate Recovery
    
    ### Available Models
    
    - **Arps Hyperbolic**: Most common, general decline
    - **Logistic Growth**: Unconventional wells, long tails
    - **Stretched Exponential**: Tight formations
    - **Power Law**: Unconventional reservoirs
    - **Duong**: Shale gas/oil, early-time behavior
    - **Wang**: Ultra-low permeability
    - **VDMA**: Variable decline rates
    
    ### Output
    
    - Fitted models with parameters
    - Production forecasts
    - EUR estimates (P10/P50/P90)
    - Goodness-of-fit metrics
    """)

