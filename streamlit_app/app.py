"""
Streamlit DCA Application - Main Entry Point
Decline Curve Analysis workflow with interactive UI
"""

import streamlit as st
import sys
import os

# Add project root and src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Initialize session state
from streamlit_app.utils.session_state import initialize_session_state
from streamlit_app.components.data_loader import load_uploaded_files
from streamlit_app.components.well_selector import render_well_selector, render_well_info
from streamlit_app.config.default_config import get_default_config

# Page configuration
st.set_page_config(
    page_title="DCA Workflow",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Load default configuration
if 'config' not in st.session_state or not st.session_state.config:
    st.session_state.config = get_default_config()

# Sidebar
with st.sidebar:
    st.markdown('<h1 class="main-header">📊 DCA Workflow</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # File upload section
    st.subheader("📁 Data Upload")
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files with columns: t, q_actual"
    )
    
    if uploaded_files:
        if len(uploaded_files) != len(st.session_state.uploaded_files) or \
           any(uf.name != sf.name for uf, sf in zip(uploaded_files, st.session_state.uploaded_files)):
            # New files uploaded, load them
            with st.spinner("Loading files..."):
                wells_data = load_uploaded_files(uploaded_files)
                
                # Update session state
                for well_id, df in wells_data.items():
                    from streamlit_app.utils.session_state import set_well_data
                    set_well_data(well_id, df)
                
                st.session_state.uploaded_files = uploaded_files
                st.success(f"Loaded {len(wells_data)} well(s)")
                st.rerun()
    
    st.markdown("---")
    
    # Well selection
    if st.session_state.well_list:
        st.subheader("🔍 Well Selection")
        selected_well = render_well_selector(st.session_state.well_list)
        
        if selected_well:
            render_well_info(selected_well)
    else:
        st.info("Upload CSV files to begin")
    
    st.markdown("---")
    
    # Navigation info
    st.subheader("📋 Workflow Steps")
    st.markdown("""
    1. **📊 Raw Data** - Preprocessing and zero removal
    2. **🔍 Noise Removal** - Outlier detection and removal
    3. **📈 Interpolation** - Fill gaps and identify fitting periods
    4. **✨ Smoothing** - Apply smoothing filters
    5. **📉 Fitting** - DCA model fitting and forecasting
    """)
    
    st.markdown("---")
    
    # Batch processing section
    if st.session_state.well_list:
        st.subheader("⚡ Batch Processing")
        
        # Well selection for batch processing
        selected_wells_batch = st.multiselect(
            "Select Wells to Process",
            options=st.session_state.well_list,
            default=st.session_state.well_list,
            help="Select which wells to process. Leave all selected to process all wells."
        )
        
        col_batch1, col_batch2 = st.columns(2)
        
        with col_batch1:
            if st.button("🚀 Run Selected Wells", type="primary", use_container_width=True, 
                        disabled=len(selected_wells_batch) == 0):
                from streamlit_app.utils.batch_processor import run_batch_processing_selected_wells
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    results = run_batch_processing_selected_wells(
                        selected_wells_batch, 
                        progress_bar=progress_bar, 
                        status_text=status_text
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if results.get('error'):
                        st.error(f"❌ {results['error']}")
                    else:
                        wells_processed = len(results['wells_processed'])
                        wells_failed = len(results['wells_failed'])
                        
                        st.success(f"✅ Batch processing completed!")
                        st.info(f"**Processed:** {wells_processed} well(s)")
                        
                        if wells_failed > 0:
                            st.warning(f"**Failed:** {wells_failed} well(s)")
                            with st.expander("View failed wells"):
                                for failed in results['wells_failed']:
                                    st.error(f"{failed['well_id']}: {failed['error']}")
                        
                        st.info("💡 You can now navigate to individual steps to review results and make adjustments.")
                        st.rerun()
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error during batch processing: {str(e)}")
                    st.exception(e)
        
        with col_batch2:
            if st.button("🚀 Run All Wells", use_container_width=True):
                from streamlit_app.utils.batch_processor import run_batch_processing_all_wells
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    results = run_batch_processing_all_wells(progress_bar=progress_bar, status_text=status_text)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if results.get('error'):
                        st.error(f"❌ {results['error']}")
                    else:
                        wells_processed = len(results['wells_processed'])
                        wells_failed = len(results['wells_failed'])
                        
                        st.success(f"✅ Batch processing completed!")
                        st.info(f"**Processed:** {wells_processed} well(s)")
                        
                        if wells_failed > 0:
                            st.warning(f"**Failed:** {wells_failed} well(s)")
                            with st.expander("View failed wells"):
                                for failed in results['wells_failed']:
                                    st.error(f"{failed['well_id']}: {failed['error']}")
                        
                        st.info("💡 You can now navigate to individual steps to review results and make adjustments.")
                        st.rerun()
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error during batch processing: {str(e)}")
                    st.exception(e)
        
        if len(selected_wells_batch) > 0:
            st.caption(f"📊 {len(selected_wells_batch)} well(s) selected for processing")
        
        st.markdown("---")
    
    # Clear data button
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.wells_data = {}
        st.session_state.well_list = []
        st.session_state.current_well = None
        st.session_state.uploaded_files = []
        from streamlit_app.utils.session_state import clear_all_results
        clear_all_results()
        st.success("All data cleared")
        st.rerun()

# Main content area
st.title("Decline Curve Analysis Workflow")
st.markdown("---")

# Check if data is loaded
if not st.session_state.well_list:
    st.info("👆 Please upload CSV files using the sidebar to begin.")
    st.markdown("""
    ### Getting Started
    
    1. **Upload Data**: Use the file uploader in the sidebar to upload one or more CSV files
    2. **Select Well**: Choose a well from the dropdown to analyze
    3. **Navigate Steps**: Use the pages menu (top left) to navigate through the workflow steps
    4. **Process Data**: Each step has controls to adjust parameters and process your data
    
    ### Data Format
    
    Your CSV files should have at least two columns:
    - `t`: Time in months (numeric)
    - `q_actual`: Production rate (numeric)
    
    ### Workflow Overview
    
    The DCA workflow consists of 5 sequential steps:
    
    1. **Raw Data (Preprocessing)**: Clean data, handle zeros and NaN values
    2. **Noise Removal**: Remove outliers using window-based and ML algorithms
    3. **Interpolation**: Fill gaps in time series and detect fitting periods
    4. **Smoothing**: Apply smoothing filters (Gaussian, Savitzky-Golay, Spline, LOWESS)
    5. **Fitting**: Fit DCA models and generate production forecasts
    
    Navigate to each step using the pages menu in the sidebar.
    """)
else:
    if st.session_state.current_well:
        st.success(f"✅ Ready to analyze: **{st.session_state.current_well}**")
        st.info("👈 Use the pages menu to navigate through the workflow steps")
    else:
        st.warning("⚠️ Please select a well from the sidebar")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Decline Curve Analysis Workflow v1.0 | "
    "Navigate through steps using the pages menu"
    "</div>",
    unsafe_allow_html=True
)

