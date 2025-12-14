"""
Well selector component for Streamlit app
Provides dropdown and well information display
"""

import streamlit as st
from typing import List, Optional, Dict
import pandas as pd
from streamlit_app.utils.session_state import get_well_data, get_current_well_data
from streamlit_app.components.data_loader import get_well_summary


def render_well_selector(well_list: List[str], key: str = "well_selector") -> Optional[str]:
    """
    Render well selector dropdown component.
    
    Args:
        well_list: List of available well IDs
        key: Unique key for the widget
    
    Returns:
        Selected well ID or None
    """
    if not well_list:
        st.info("No wells available. Please upload CSV files first.")
        return None
    
    # Create display names with metadata
    display_options = []
    for well_id in well_list:
        df = get_well_data(well_id)
        if df is not None:
            num_points = len(df)
            well_type = "Oil" if "oil" in well_id.lower() else "Gas" if "gas" in well_id.lower() else "Unknown"
            display_name = f"{well_id} ({well_type}, {num_points} pts)"
            display_options.append((display_name, well_id))
        else:
            display_options.append((well_id, well_id))
    
    # Create mapping for display
    display_to_id = {display: well_id for display, well_id in display_options}
    display_names = [display for display, _ in display_options]
    
    # Get current selection index
    current_idx = 0
    if st.session_state.current_well and st.session_state.current_well in well_list:
        current_idx = well_list.index(st.session_state.current_well)
    
    # Render selectbox
    selected_display = st.selectbox(
        "Select Well:",
        options=display_names,
        index=current_idx,
        key=key
    )
    
    # Get corresponding well ID
    selected_well_id = display_to_id.get(selected_display)
    
    # Update session state
    if selected_well_id:
        st.session_state.current_well = selected_well_id
    
    return selected_well_id


def render_well_info(well_id: Optional[str]):
    """
    Render well information panel.
    
    Args:
        well_id: Well identifier to display info for
    """
    if not well_id:
        st.info("No well selected")
        return
    
    df = get_well_data(well_id)
    if df is None:
        st.warning(f"No data found for well: {well_id}")
        return
    
    summary = get_well_summary(well_id, df)
    
    # Display summary in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", summary['num_points'])
        if summary['time_range']:
            st.caption(f"Time: {summary['time_range'][0]:.1f} - {summary['time_range'][1]:.1f} months")
    
    with col2:
        if summary['q_mean'] is not None:
            st.metric("Mean Rate", f"{summary['q_mean']:.2f}")
            st.caption(f"Std: {summary['q_std']:.2f}")
    
    with col3:
        if summary['q_min'] is not None:
            st.metric("Rate Range", f"{summary['q_min']:.2f} - {summary['q_max']:.2f}")
            if summary['zero_count'] > 0:
                st.caption(f"Zeros: {summary['zero_count']}")


def filter_wells_by_type(well_list: List[str], well_type: Optional[str] = None) -> List[str]:
    """
    Filter wells by type (oil/gas).
    
    Args:
        well_list: List of well IDs
        well_type: 'oil', 'gas', or None for all
    
    Returns:
        Filtered list of well IDs
    """
    if well_type is None:
        return well_list
    
    filtered = []
    for well_id in well_list:
        well_id_lower = well_id.lower()
        if well_type.lower() == 'oil' and 'oil' in well_id_lower:
            filtered.append(well_id)
        elif well_type.lower() == 'gas' and 'gas' in well_id_lower:
            filtered.append(well_id)
    
    return filtered

