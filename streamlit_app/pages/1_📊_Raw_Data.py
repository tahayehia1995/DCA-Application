"""
Step 0: Raw Data Preprocessing
Handle NaN values, zeros, and generate cumulative production
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
    get_current_well_data, get_step_result, set_step_result, has_step_result
)
from streamlit_app.components.well_selector import render_well_selector
from streamlit_app.utils.streamlit_helpers import create_plotly_preprocessing, export_dataframe
from streamlit_app.config.default_config import get_default_config
from src.utils.file_handler import get_file_info

# Page config
st.set_page_config(page_title="Raw Data - Preprocessing", layout="wide")

st.title("📊 Step 0: Raw Data Preprocessing")
st.markdown("Clean production data by handling NaN values, zeros, and generating cumulative production.")

# Check if wells are loaded
if not st.session_state.well_list:
    st.warning("⚠️ No wells loaded. Please upload CSV files from the main page.")
    st.stop()

# Well selector
selected_well = render_well_selector(st.session_state.well_list, key="step0_well_selector")

if not selected_well:
    st.stop()

# Get well data
well_data = get_current_well_data()
if well_data is None:
    st.error(f"No data found for well: {selected_well}")
    st.stop()

# Determine well type
file_info = get_file_info(selected_well)
well_type = file_info.get('well_type') or 'oil'  # Handle None values

# Parameter controls
st.sidebar.header("⚙️ Parameters")

default_config = get_default_config()
step0_config = default_config.get('step0_preprocessing', {})

drop_zeros = st.sidebar.toggle(
    "Drop Zero Values",
    value=step0_config.get('drop_zeros', True),
    help="Remove rows with zero production values"
)

# Check if we already have results for this well
has_results = has_step_result('step0', selected_well)
if has_results:
    st.sidebar.success("✅ Results available")
    if st.sidebar.button("🔄 Re-run Processing"):
        has_results = False

# Process button
process_btn = st.sidebar.button("▶️ Run Preprocessing", type="primary", use_container_width=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Data Visualization")
    
    if process_btn or has_results:
        if process_btn:
            with st.spinner("Processing data..."):
                try:
                    # Prepare data for processing
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
                    
                    # Apply zero removal if requested
                    zeros_removed = None
                    if drop_zeros:
                        zeros_mask = df['q_actual'] == 0
                        zeros_removed = df[zeros_mask].copy()
                        df = df[~zeros_mask]
                    
                    final_length = len(df)
                    zeros_dropped = zero_count_before if drop_zeros else 0
                    
                    # Calculate cumulative production
                    df['Gp_actual'] = df['q_actual'].cumsum()
                    
                    # Store results
                    result = {
                        'processed_df': df,
                        'original_df': original_df,
                        'zeros_removed': zeros_removed,
                        'initial_length': initial_length,
                        'final_length': final_length,
                        'zeros_dropped': zeros_dropped,
                        'drop_zeros': drop_zeros
                    }
                    
                    set_step_result('step0', selected_well, result)
                    st.success("✅ Preprocessing completed!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
        
        # Display results
        result = get_step_result('step0', selected_well)
        if result:
            processed_df = result['processed_df']
            original_df = result['original_df']
            zeros_removed = result.get('zeros_removed')
            
            # Create visualization
            fig = create_plotly_preprocessing(
                original_df,
                processed_df,
                zeros_removed,
                well_type=well_type
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Show original data
        fig = create_plotly_preprocessing(well_data, well_type=well_type)
        st.plotly_chart(fig, use_container_width=True)
        st.info("👆 Click 'Run Preprocessing' to process the data")

with col2:
    st.subheader("📊 Summary Statistics")
    
    if has_results:
        result = get_step_result('step0', selected_well)
        if result:
            stats_df = pd.DataFrame({
                'Metric': [
                    'Initial Points',
                    'Final Points',
                    'Zeros Dropped',
                    'Percentage Removed'
                ],
                'Value': [
                    result['initial_length'],
                    result['final_length'],
                    result['zeros_dropped'],
                    f"{(result['zeros_dropped'] / result['initial_length'] * 100):.2f}%" if result['initial_length'] > 0 else "0%"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Data preview
            st.subheader("📋 Processed Data Preview")
            st.dataframe(
                result['processed_df'].head(10),
                use_container_width=True,
                height=300
            )
            
            # Export button
            csv_data = export_dataframe(result['processed_df'])
            st.download_button(
                label="📥 Download Processed Data (CSV)",
                data=csv_data,
                file_name=f"{selected_well}_preprocessed.csv",
                mime="text/csv"
            )
    else:
        st.info("Run preprocessing to see statistics")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    ### Preprocessing Step
    
    This step performs the following operations:
    
    1. **Data Validation**: Checks for required columns (t, q_actual)
    2. **NaN Handling**: Removes rows with invalid time values, fills missing production with zeros
    3. **Zero Removal**: Optionally removes rows with zero production values
    4. **Cumulative Production**: Calculates cumulative production (Gp) from rate data
    
    ### Parameters
    
    - **Drop Zero Values**: When enabled, removes all rows where q_actual = 0
    
    ### Output
    
    - Cleaned DataFrame with columns: t, q_actual, Gp_actual
    - Summary statistics showing data reduction
    - Interactive visualization comparing original and processed data
    """)

