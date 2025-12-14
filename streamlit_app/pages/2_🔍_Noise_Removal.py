"""
Step 1: Noise Removal
Remove outliers using window-based and/or machine learning algorithms
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
    get_current_well_data, get_step_result, set_step_result, has_step_result, get_step_result as get_prev_step_result,
    get_manual_outlier_selections, set_manual_outlier_selections, clear_manual_outlier_selections
)
from streamlit_app.components.well_selector import render_well_selector
from streamlit_app.utils.streamlit_helpers import create_plotly_noise_removal, export_dataframe
from streamlit_app.config.default_config import get_default_config
from src.utils.file_handler import get_file_info

# Page config
st.set_page_config(page_title="Noise Removal", layout="wide")

st.title("🔍 Step 1: Noise Removal")
st.markdown("Remove outliers using window-based methods and/or machine learning algorithms.")

# Check prerequisites
if not st.session_state.well_list:
    st.warning("⚠️ No wells loaded. Please upload CSV files from the main page.")
    st.stop()

# Well selector
selected_well = render_well_selector(st.session_state.well_list, key="step1_well_selector")

if not selected_well:
    st.stop()

# Check if Step 0 is completed
step0_result = get_prev_step_result('step0', selected_well)
if step0_result is None:
    st.warning("⚠️ Please complete Step 0 (Raw Data Preprocessing) first.")
    if st.button("Go to Step 0"):
        st.switch_page("pages/1_📊_Raw_Data.py")
    st.stop()

# Get input data (use Step 0 output if available, otherwise raw data)
input_df = step0_result['processed_df'] if step0_result else get_current_well_data()

if input_df is None:
    st.error(f"No data found for well: {selected_well}")
    st.stop()

# Parameter controls
st.sidebar.header("⚙️ Parameters")

default_config = get_default_config()
step1_config = default_config.get('step1_noise_removal', {})

use_window = st.sidebar.toggle(
    "Use Window Method",
    value=step1_config.get('use_window', True),
    help="Apply window-based outlier detection before ML algorithms"
)

if use_window:
    window_size = st.sidebar.number_input(
        "Window Size",
        min_value=5,
        max_value=50,
        value=step1_config.get('window_size', 15),
        help="Size of rolling window for outlier detection"
    )
    
    step_size = st.sidebar.number_input(
        "Step Size",
        min_value=1,
        max_value=20,
        value=step1_config.get('step_size', 5),
        help="Step size for moving window"
    )
    
    window_method = st.sidebar.selectbox(
        "Window Method",
        options=['lowest_quantile', 'both_quantiles', 'lowest_points'],
        index=2 if step1_config.get('window_method') == 'lowest_points' else 0,
        help="Method for detecting outliers in window"
    )
    
    num_lowest_points = None
    if window_method == 'lowest_points':
        num_lowest_points = st.sidebar.number_input(
            "Number of Lowest Points",
            min_value=1,
            max_value=10,
            value=step1_config.get('num_lowest_points', 3)
        )

# Algorithm selection
algorithm_options = ['knn', 'lof', 'abod', 'cof', 'cluster', 'iforest', 'all', 'window']
selected_algorithms = st.sidebar.multiselect(
    "Algorithms",
    options=algorithm_options,
    default=['all'] if 'all' in step1_config.get('algorithms', ['all']) else step1_config.get('algorithms', ['all']),
    help="Select anomaly detection algorithms to use"
)

if not selected_algorithms:
    st.sidebar.warning("Please select at least one algorithm")

# Hyperparameters (expandable)
with st.sidebar.expander("🔧 Algorithm Hyperparameters"):
    hyperparams = {}
    hyperparams_config = step1_config.get('hyperparameters', {})
    
    for algo in ['knn', 'lof', 'abod', 'cof', 'cluster', 'iforest']:
        if algo in selected_algorithms or 'all' in selected_algorithms:
            st.markdown(f"**{algo.upper()}**")
            algo_config = hyperparams_config.get(algo, {})
            
            fraction = st.slider(
                f"{algo} - Fraction",
                min_value=0.05,
                max_value=0.30,
                value=algo_config.get('fraction', 0.20),
                step=0.01,
                key=f"frac_{algo}"
            )
            
            if algo in ['knn', 'lof', 'abod', 'cof']:
                n_neighbors = st.number_input(
                    f"{algo} - N Neighbors",
                    min_value=3,
                    max_value=50,
                    value=algo_config.get('n_neighbors', 20 if algo in ['knn', 'lof'] else 3),
                    key=f"n_neigh_{algo}"
                )
                hyperparams[algo] = {'fraction': fraction, 'n_neighbors': n_neighbors}
            elif algo == 'cluster':
                n_clusters = st.number_input(
                    f"{algo} - N Clusters",
                    min_value=2,
                    max_value=10,
                    value=algo_config.get('n_clusters', 3),
                    key=f"n_clust_{algo}"
                )
                hyperparams[algo] = {'fraction': fraction, 'n_clusters': n_clusters}
            elif algo == 'iforest':
                n_estimators = st.number_input(
                    f"{algo} - N Estimators",
                    min_value=100,
                    max_value=300,
                    value=algo_config.get('n_estimators', 200),
                    key=f"n_est_{algo}"
                )
                hyperparams[algo] = {'fraction': fraction, 'n_estimators': n_estimators}

# Check for existing results
has_results = has_step_result('step1', selected_well)
if has_results:
    st.sidebar.success("✅ Results available")
    if st.sidebar.button("🔄 Re-run Noise Removal"):
        has_results = False

# Process button
process_btn = st.sidebar.button("▶️ Run Noise Removal", type="primary", use_container_width=True)

# Main content
if process_btn and selected_algorithms:
    with st.spinner("Running noise removal algorithms... This may take a while."):
        try:
            # Import noise removal function
            from step1_noise_removal import detect_and_remove_outliers
            
            # Prepare temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save input data to temp file
                temp_input_file = os.path.join(temp_dir, f"{selected_well}_input.csv")
                input_df.to_csv(temp_input_file, index=False)
                
                # Determine algorithm choice
                if 'all' in selected_algorithms:
                    algo_choice = 'all'
                elif len(selected_algorithms) == 1:
                    algo_choice = selected_algorithms[0]
                else:
                    algo_choice = selected_algorithms[0]  # Process first, can extend later
                
                # Run noise removal
                results = detect_and_remove_outliers(
                    directory=temp_dir,
                    algorithm_choice=algo_choice,
                    hyperparameters=hyperparams if hyperparams else None,
                    use_window=use_window,
                    window_size=window_size if use_window else 15,
                    step_size=step_size if use_window else 5,
                    drop_repeated=False,
                    window_method=window_method if use_window else 'lowest_points',
                    num_lowest_points=num_lowest_points if window_method == 'lowest_points' else 3,
                    wells_to_process=None
                )
                
                # Process results
                cleaned_data_dict = {}
                window_outliers_data = None
                algorithm_names = []
                
                for file, original_data, cleaned_data, window_outliers, x_col, y_col, algo, outlier_percent in results:
                    cleaned_data_dict[algo] = cleaned_data
                    algorithm_names.append(algo)
                    if window_outliers is not None and len(window_outliers) > 0:
                        window_outliers_data = window_outliers
                
                # Store results
                result = {
                    'original_df': input_df,
                    'cleaned_data_dict': cleaned_data_dict,
                    'window_outliers': window_outliers_data,
                    'algorithm_names': algorithm_names,
                    'parameters': {
                        'use_window': use_window,
                        'window_size': window_size if use_window else None,
                        'step_size': step_size if use_window else None,
                        'window_method': window_method if use_window else None,
                        'algorithms': selected_algorithms,
                        'hyperparameters': hyperparams
                    }
                }
                
                set_step_result('step1', selected_well, result)
                st.success("✅ Noise removal completed!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error during noise removal: {str(e)}")
            st.exception(e)
            st.info("💡 Tip: Make sure PyCaret is installed (pip install pycaret)")

# Display results
if has_results:
    result = get_step_result('step1', selected_well)
    if result:
        # Check if result has required fields
        if 'original_df' not in result or 'cleaned_data_dict' not in result:
            st.warning("⚠️ Incomplete results found. Please re-run noise removal for this well.")
        else:
            algorithm_names = result.get('algorithm_names', list(result.get('cleaned_data_dict', {}).keys()))
            
            # Point selection UI
            st.subheader("✏️ Point Selection Editor")
            edit_mode = st.toggle("Enable Edit Mode", key="edit_mode_toggle", 
                                 help="Enable to manually select/deselect outlier points")
            
            selected_algo_edit = None
            selected_outliers = set()
            deselected_outliers = set()
            
            if edit_mode:
                selected_algo_edit = st.selectbox(
                    "Select Algorithm to Edit",
                    options=algorithm_names,
                    key="edit_algo_selector",
                    help="Choose which algorithm's results to modify"
                )
                
                # Get current manual selections
                manual_selections = get_manual_outlier_selections(selected_well, selected_algo_edit)
                selected_outliers = manual_selections.get('selected_outliers', set())
                deselected_outliers = manual_selections.get('deselected_outliers', set())
                
                col_edit1, col_edit2 = st.columns(2)
                
                with col_edit1:
                    st.markdown("**Select Points as Outliers**")
                    st.caption("Enter point indices (comma-separated) to mark as outliers")
                    outlier_indices_input = st.text_input(
                        "Point Indices",
                        value=", ".join(map(str, sorted(selected_outliers))) if selected_outliers else "",
                        key="outlier_indices_input",
                        help="Enter comma-separated indices from the original data"
                    )
                    
                    if st.button("Add Selected Points", key="add_outliers_btn"):
                        try:
                            indices = [int(x.strip()) for x in outlier_indices_input.split(",") if x.strip()]
                            selected_outliers.update(indices)
                            # Remove from deselected if present
                            deselected_outliers.difference_update(indices)
                            set_manual_outlier_selections(selected_well, selected_algo_edit, {
                                'selected_outliers': selected_outliers,
                                'deselected_outliers': deselected_outliers
                            })
                            st.success(f"Added {len(indices)} point(s) as outliers")
                            st.rerun()
                        except ValueError:
                            st.error("Please enter valid comma-separated integers")
                
                with col_edit2:
                    st.markdown("**Deselect Points (Mark as Inliers)**")
                    st.caption("Enter point indices to mark as inliers (remove from outliers)")
                    inlier_indices_input = st.text_input(
                        "Point Indices",
                        value=", ".join(map(str, sorted(deselected_outliers))) if deselected_outliers else "",
                        key="inlier_indices_input",
                        help="Enter comma-separated indices to mark as inliers"
                    )
                    
                    if st.button("Add Deselected Points", key="add_inliers_btn"):
                        try:
                            indices = [int(x.strip()) for x in inlier_indices_input.split(",") if x.strip()]
                            deselected_outliers.update(indices)
                            # Remove from selected if present
                            selected_outliers.difference_update(indices)
                            set_manual_outlier_selections(selected_well, selected_algo_edit, {
                                'selected_outliers': selected_outliers,
                                'deselected_outliers': deselected_outliers
                            })
                            st.success(f"Added {len(indices)} point(s) as inliers")
                            st.rerun()
                        except ValueError:
                            st.error("Please enter valid comma-separated integers")
                
                # Show current selections
                if selected_outliers or deselected_outliers:
                    st.info(f"**Current Selections for {selected_algo_edit}:** "
                           f"{len(selected_outliers)} manually selected outliers, "
                           f"{len(deselected_outliers)} manually deselected outliers")
                    
                    if st.button("Clear All Selections", key="clear_selections_btn"):
                        clear_manual_outlier_selections(selected_well, selected_algo_edit)
                        st.success("Selections cleared")
                        st.rerun()
                
                st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Visualization")
                
                # Get manual selections for visualization
                selected_algo_viz = selected_algo_edit if (edit_mode and selected_algo_edit) else (algorithm_names[0] if algorithm_names else None)
                manual_selections_viz = None
                if selected_algo_viz:
                    manual_selections_viz = get_manual_outlier_selections(selected_well, selected_algo_viz)
                
                # Create visualization
                fig = create_plotly_noise_removal(
                    result['original_df'],
                    result['cleaned_data_dict'],
                    result.get('window_outliers'),
                    result.get('algorithm_names', list(result.get('cleaned_data_dict', {}).keys())),
                    well_id=selected_well,
                    selected_algorithm=selected_algo_viz,
                    manual_selections=manual_selections_viz
                )
                st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="noise_removal_chart")
                
                # Apply manual selections to cleaned data
                if edit_mode and (selected_outliers or deselected_outliers):
                    if st.button("✅ Apply Manual Selections", type="primary", key="apply_selections_btn"):
                        # Apply selections to cleaned data
                        original_df = result['original_df']
                        cleaned_df_original = result['cleaned_data_dict'][selected_algo_edit].copy()
                        x_col = original_df.columns[0]
                        y_col = original_df.columns[1]
                        
                        # Remove manually selected outliers
                        if selected_outliers:
                            # Find rows to remove by matching indices
                            for idx in selected_outliers:
                                if idx in original_df.index:
                                    row = original_df.loc[idx]
                                    x_val = row[x_col]
                                    y_val = row[y_col]
                                    cleaned_df_original = cleaned_df_original[
                                        ~((cleaned_df_original.iloc[:, 0] == x_val) & 
                                          (cleaned_df_original.iloc[:, 1] == y_val))
                                    ]
                        
                        # Add manually deselected outliers back
                        if deselected_outliers:
                            for idx in deselected_outliers:
                                if idx in original_df.index:
                                    row = original_df.loc[idx]
                                    x_val = row[x_col]
                                    y_val = row[y_col]
                                    # Check if not already present
                                    if not any((cleaned_df_original.iloc[:, 0] == x_val) & 
                                              (cleaned_df_original.iloc[:, 1] == y_val)):
                                        new_row = pd.DataFrame({cleaned_df_original.columns[0]: [x_val],
                                                              cleaned_df_original.columns[1]: [y_val]})
                                        cleaned_df_original = pd.concat([cleaned_df_original, new_row], ignore_index=True)
                        
                        # Update cleaned data dict
                        result['cleaned_data_dict'][selected_algo_edit] = cleaned_df_original.sort_values(
                            by=cleaned_df_original.columns[0]
                        ).reset_index(drop=True)
                        
                        # Store updated result
                        result['manual_selections'] = {
                            selected_algo_edit: {
                                'selected_outliers': list(selected_outliers),
                                'deselected_outliers': list(deselected_outliers)
                            }
                        }
                        set_step_result('step1', selected_well, result)
                        st.success("✅ Manual selections applied! Updated data will be used in downstream steps.")
                        st.rerun()
            
            with col2:
                st.subheader("📊 Summary")
                
                # Algorithm comparison
                summary_data = []
                for algo_name in algorithm_names:
                    if algo_name in result['cleaned_data_dict']:
                        original_len = len(result['original_df'])
                        cleaned_len = len(result['cleaned_data_dict'][algo_name])
                        removed = original_len - cleaned_len
                        percent_removed = (removed / original_len * 100) if original_len > 0 else 0
                        
                        # Check for manual selections
                        manual_selections_check = get_manual_outlier_selections(selected_well, algo_name)
                        manual_selected_count = len(manual_selections_check.get('selected_outliers', set()))
                        manual_deselected_count = len(manual_selections_check.get('deselected_outliers', set()))
                        
                        summary_data.append({
                            'Algorithm': algo_name.upper(),
                            'Points Removed': removed,
                            'Percent Removed': f"{percent_removed:.2f}%",
                            'Final Points': cleaned_len,
                            'Manual Edits': f"{manual_selected_count}+/{manual_deselected_count}-" if (manual_selected_count or manual_deselected_count) else "None"
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Select algorithm for export
                    selected_algo = st.selectbox(
                        "Select Algorithm for Export",
                        options=algorithm_names,
                        key="export_algo_selector"
                    )
                    
                    if selected_algo in result['cleaned_data_dict']:
                        export_df = result['cleaned_data_dict'][selected_algo]
                        csv_data = export_dataframe(export_df)
                        st.download_button(
                            label=f"📥 Download ({selected_algo.upper()})",
                            data=csv_data,
                            file_name=f"{selected_well}_cleaned_{selected_algo}.csv",
                            mime="text/csv"
                        )
else:
    st.info("👆 Configure parameters and click 'Run Noise Removal' to process the data")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    ### Noise Removal Step
    
    This step removes outliers from production data using:
    
    1. **Window-Based Method**: Rolling window analysis to detect outliers
    2. **Machine Learning Algorithms**: Various anomaly detection algorithms
    
    ### Available Algorithms
    
    - **KNN**: K-Nearest Neighbors outlier detection
    - **LOF**: Local Outlier Factor
    - **ABOD**: Angle-Based Outlier Detection
    - **COF**: Connectivity-Based Outlier Factor
    - **Cluster**: Clustering-based detection
    - **IForest**: Isolation Forest
    - **All**: Ensemble of all ML algorithms
    
    ### Parameters
    
    - **Window Size**: Size of rolling window (larger = more smoothing)
    - **Step Size**: How much window moves each iteration
    - **Window Method**: How outliers are detected in each window
    - **Fraction**: Percentage of data to flag as anomalies (ML algorithms)
    
    ### Output
    
    - Cleaned data for each selected algorithm
    - Comparison visualization
    - Summary statistics
    """)

