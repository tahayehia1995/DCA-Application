"""
Step 2: Interpolation
Fill gaps in time series and identify fitting periods
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import tempfile

# Add project root and src to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from streamlit_app.utils.session_state import (
    get_current_well_data, get_step_result, set_step_result, has_step_result, get_step_result as get_prev_step_result
)
from streamlit_app.components.well_selector import render_well_selector
from streamlit_app.utils.streamlit_helpers import create_plotly_interpolation, export_dataframe
from streamlit_app.config.default_config import get_default_config
from src.utils.file_handler import get_file_info
from step2_interpolation import interpolate_data, identify_fitting_periods

# Page config
st.set_page_config(page_title="Interpolation", layout="wide")

st.title("📈 Step 2: Interpolation")
st.markdown("Fill gaps in time series and identify fitting periods for decline curve analysis.")

# Check prerequisites
if not st.session_state.well_list:
    st.warning("⚠️ No wells loaded. Please upload CSV files from the main page.")
    st.stop()

# Well selector
selected_well = render_well_selector(st.session_state.well_list, key="step2_well_selector")

if not selected_well:
    st.stop()

# Check if Step 1 is completed (prefer Step 1, fallback to Step 0)
step1_result = get_prev_step_result('step1', selected_well)
step0_result = get_prev_step_result('step0', selected_well)

if step1_result:
    # Use cleaned data from Step 1
    input_df = None
    for algo_name, cleaned_df in step1_result['cleaned_data_dict'].items():
        input_df = cleaned_df  # Use first available algorithm
        break
    if input_df is None:
        st.error("No cleaned data available from Step 1")
        st.stop()
elif step0_result:
    input_df = step0_result['processed_df']
else:
    st.warning("⚠️ Please complete Step 0 (Raw Data) or Step 1 (Noise Removal) first.")
    st.stop()

# Parameter controls
st.sidebar.header("⚙️ Parameters")

default_config = get_default_config()
step2_config = default_config.get('step2_interpolation', {})

interpolation_method = st.sidebar.selectbox(
    "Interpolation Method",
    options=['linear', 'cubic', 'quadratic', 'nearest'],
    index=1 if step2_config.get('interpolation_method') == 'cubic' else (0 if step2_config.get('interpolation_method') == 'linear' else 1),
    help="Method for interpolating missing time points"
)

min_fitting_period = st.sidebar.number_input(
    "Minimum Fitting Period (months)",
    min_value=10,
    max_value=100,
    value=step2_config.get('min_fitting_period', 30),
    help="Minimum length for a valid decline fitting period"
)

# Check for existing results
has_results = has_step_result('step2', selected_well)
if has_results:
    st.sidebar.success("✅ Results available")
    if st.sidebar.button("🔄 Re-run Interpolation"):
        has_results = False

# Process button
process_btn = st.sidebar.button("▶️ Run Interpolation", type="primary", use_container_width=True)

# Main content
if process_btn:
    with st.spinner("Interpolating data and identifying fitting periods..."):
        try:
            # Run interpolation
            df_interpolated, interpolated_flags = interpolate_data(input_df.copy(), method=interpolation_method)
            
            # Identify fitting periods
            fitting_periods = identify_fitting_periods(df_interpolated.copy(), min_period_length=min_fitting_period)
            
            # Store results
            result = {
                'original_df': input_df,
                'interpolated_df': df_interpolated,
                'interpolated_flags': interpolated_flags,
                'fitting_periods': fitting_periods,
                'parameters': {
                    'interpolation_method': interpolation_method,
                    'min_fitting_period': min_fitting_period
                }
            }
            
            set_step_result('step2', selected_well, result)
            st.success("✅ Interpolation completed!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during interpolation: {str(e)}")
            st.exception(e)

# Display results
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Visualization")
    
    if has_results:
        result = get_step_result('step2', selected_well)
        if result:
            # Create visualization
            fig = create_plotly_interpolation(
                result['original_df'],
                result['interpolated_df'],
                result.get('interpolated_flags')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show fitting periods
            if result['fitting_periods']:
                st.subheader("🎯 Detected Fitting Periods")
                for idx, (start, end) in enumerate(result['fitting_periods']):
                    st.info(f"**Period {idx + 1}**: {start:.1f} - {end:.1f} months (Length: {end - start:.1f} months)")
    else:
        # Show original data
        fig = create_plotly_interpolation(input_df, input_df)
        st.plotly_chart(fig, use_container_width=True)
        st.info("👆 Click 'Run Interpolation' to process the data")

with col2:
    st.subheader("📊 Summary")
    
    if has_results:
        result = get_step_result('step2', selected_well)
        if result:
            original_points = len(result['original_df'])
            interpolated_points = len(result['interpolated_df'])
            gaps_filled = interpolated_points - original_points
            
            # Count interpolated points
            num_interpolated = 0
            if result.get('interpolated_flags') is not None:
                num_interpolated = np.sum(result['interpolated_flags'])
            
            stats_df = pd.DataFrame({
                'Metric': [
                    'Original Points',
                    'Interpolated Points',
                    'Gaps Filled',
                    'Points Interpolated',
                    'Fitting Periods Found'
                ],
                'Value': [
                    original_points,
                    interpolated_points,
                    gaps_filled,
                    num_interpolated,
                    len(result['fitting_periods'])
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Data preview
            st.subheader("📋 Interpolated Data Preview")
            st.dataframe(
                result['interpolated_df'].head(10),
                use_container_width=True,
                height=300
            )
            
            # Export button
            csv_data = export_dataframe(result['interpolated_df'])
            st.download_button(
                label="📥 Download Interpolated Data",
                data=csv_data,
                file_name=f"{selected_well}_interpolated.csv",
                mime="text/csv"
            )
    else:
        st.info("Run interpolation to see statistics")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    ### Interpolation Step
    
    This step performs:
    
    1. **Gap Filling**: Adds missing time points and interpolates values
    2. **Fitting Period Detection**: Identifies continuous periods suitable for DCA
    
    ### Interpolation Methods
    
    - **Linear**: Straight-line interpolation (most stable)
    - **Cubic**: Cubic spline interpolation (smooth curves)
    - **Quadratic**: Quadratic interpolation (moderate smoothing)
    - **Nearest**: Nearest neighbor interpolation (step-like)
    
    ### Fitting Periods
    
    The algorithm identifies periods where:
    - Production is relatively stable or declining
    - Minimum length requirement is met
    - Suitable for decline curve fitting
    
    ### Output
    
    - Complete time series with no gaps
    - Detected fitting periods (start/end times)
    - Visualization showing interpolated points
    """)

