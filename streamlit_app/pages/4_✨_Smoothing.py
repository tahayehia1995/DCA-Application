"""
Step 3: Smoothing
Apply various smoothing filters to production data
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add project root and src to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from streamlit_app.utils.session_state import (
    get_current_well_data, get_step_result, set_step_result, has_step_result, get_step_result as get_prev_step_result
)
from streamlit_app.components.well_selector import render_well_selector
from streamlit_app.utils.streamlit_helpers import create_plotly_smoothing, export_dataframe
from streamlit_app.config.default_config import get_default_config
from src.utils.file_handler import get_file_info
from step3_smoothing import apply_filters

# Page config
st.set_page_config(page_title="Smoothing", layout="wide")

st.title("✨ Step 3: Smoothing")
st.markdown("Apply various smoothing filters to reduce noise and prepare data for decline curve fitting.")

# Check prerequisites
if not st.session_state.well_list:
    st.warning("⚠️ No wells loaded. Please upload CSV files from the main page.")
    st.stop()

# Well selector
selected_well = render_well_selector(st.session_state.well_list, key="step3_well_selector")

if not selected_well:
    st.stop()

# Check if Step 2 is completed (prefer Step 2, fallback to earlier steps)
step2_result = get_prev_step_result('step2', selected_well)
step1_result = get_prev_step_result('step1', selected_well)
step0_result = get_prev_step_result('step0', selected_well)

if step2_result:
    input_df = step2_result['interpolated_df']
elif step1_result:
    # Use cleaned data from Step 1
    input_df = None
    for algo_name, cleaned_df in step1_result['cleaned_data_dict'].items():
        input_df = cleaned_df
        break
    if input_df is None:
        st.error("No cleaned data available from Step 1")
        st.stop()
elif step0_result:
    input_df = step0_result['processed_df']
else:
    st.warning("⚠️ Please complete Step 0 (Raw Data) first.")
    st.stop()

# Parameter controls
st.sidebar.header("⚙️ Parameters")

default_config = get_default_config()
step3_config = default_config.get('step3_smoothing', {})

available_filters = ['Gaussian', 'Savitzky-Golay', 'Spline', 'Lowess']
selected_filters = st.sidebar.multiselect(
    "Smoothing Filters",
    options=available_filters,
    default=step3_config.get('filters', available_filters),
    help="Select smoothing filters to apply"
)

if not selected_filters:
    st.sidebar.warning("Please select at least one filter")

# Filter-specific parameters (expandable)
with st.sidebar.expander("🔧 Filter Parameters"):
    filter_params = {}
    
    if 'Lowess' in selected_filters:
        st.markdown("**LOWESS**")
        lowess_frac = st.slider(
            "LOWESS Fraction",
            min_value=0.05,
            max_value=0.30,
            value=0.10,
            step=0.01,
            help="Fraction of data to use in LOWESS smoothing"
        )
        filter_params['Lowess'] = {'frac': lowess_frac}

# Check for existing results
has_results = has_step_result('step3', selected_well)
if has_results:
    st.sidebar.success("✅ Results available")
    if st.sidebar.button("🔄 Re-run Smoothing"):
        has_results = False

# Process button
process_btn = st.sidebar.button("▶️ Run Smoothing", type="primary", use_container_width=True)

# Main content
if process_btn and selected_filters:
    with st.spinner("Applying smoothing filters..."):
        try:
            # Prepare data
            t = input_df['t'].values
            q_actual = input_df['q_actual'].values
            
            # Apply filters
            filters_dict = apply_filters(t, q_actual)
            
            # Create output DataFrame
            output_df = input_df.copy()
            
            # Add filtered columns
            applied_filters = []
            for filter_name in selected_filters:
                if filter_name in filters_dict:
                    output_df[filter_name] = filters_dict[filter_name]
                    applied_filters.append(filter_name)
            
            # Store results
            result = {
                'input_df': input_df,
                'output_df': output_df,
                'filters_dict': {k: v for k, v in filters_dict.items() if k in applied_filters},
                'applied_filters': applied_filters,
                'parameters': {
                    'filters': selected_filters,
                    'filter_params': filter_params
                }
            }
            
            set_step_result('step3', selected_well, result)
            st.success("✅ Smoothing completed!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during smoothing: {str(e)}")
            st.exception(e)

# Display results
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Visualization")
    
    if has_results:
        result = get_step_result('step3', selected_well)
        if result:
            # Create visualization
            fig = create_plotly_smoothing(
                result['output_df'],
                result['applied_filters'],
                raw_column='q_actual'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Show input data
        fig = create_plotly_smoothing(input_df, [], raw_column='q_actual')
        st.plotly_chart(fig, use_container_width=True)
        st.info("👆 Select filters and click 'Run Smoothing' to process the data")

with col2:
    st.subheader("📊 Summary")
    
    if has_results:
        result = get_step_result('step3', selected_well)
        if result:
            # Filter comparison statistics
            comparison_data = []
            raw_data = result['input_df']['q_actual']
            
            comparison_data.append({
                'Filter': 'Raw Data',
                'Mean': raw_data.mean(),
                'Std': raw_data.std(),
                'Min': raw_data.min(),
                'Max': raw_data.max()
            })
            
            for filter_name in result['applied_filters']:
                if filter_name in result['output_df'].columns:
                    filter_data = result['output_df'][filter_name]
                    comparison_data.append({
                        'Filter': filter_name,
                        'Mean': filter_data.mean(),
                        'Std': filter_data.std(),
                        'Min': filter_data.min(),
                        'Max': filter_data.max()
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Select filter for export
            selected_filter = st.selectbox(
                "Select Filter for Export",
                options=['Raw'] + result['applied_filters'],
                key="export_filter_selector"
            )
            
            if selected_filter == 'Raw':
                export_df = result['input_df'][['t', 'q_actual']]
            elif selected_filter in result['output_df'].columns:
                export_df = result['output_df'][['t', 'q_actual', selected_filter]]
            else:
                export_df = result['output_df']
            
            csv_data = export_dataframe(export_df)
            st.download_button(
                label=f"📥 Download ({selected_filter})",
                data=csv_data,
                file_name=f"{selected_well}_smoothed_{selected_filter.lower()}.csv",
                mime="text/csv"
            )
    else:
        st.info("Run smoothing to see statistics")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    ### Smoothing Step
    
    This step applies various smoothing filters to reduce noise:
    
    1. **Gaussian Filter**: Fast, simple noise reduction
    2. **Savitzky-Golay Filter**: Preserves peaks and features
    3. **Spline Smoothing**: Very smooth curves
    4. **LOWESS**: Robust, adapts to local patterns
    
    ### Filter Characteristics
    
    - **Gaussian**: Best for general noise reduction
    - **Savitzky-Golay**: Best for preserving local maxima/minima
    - **Spline**: Best for very smooth curves (may over-smooth)
    - **LOWESS**: Best for non-parametric smoothing, robust to outliers
    
    ### Output
    
    - Smoothed data for each selected filter
    - Comparison visualization
    - Statistics for each filter
    - Export option for selected filter
    """)

