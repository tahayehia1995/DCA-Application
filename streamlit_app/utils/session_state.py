"""
Session state management for Streamlit app
Initializes and manages all session state variables
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd


def initialize_session_state():
    """
    Initialize all session state variables if they don't exist.
    This should be called at the start of the main app.
    """
    # File upload and data storage
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'wells_data' not in st.session_state:
        st.session_state.wells_data = {}  # Dict[well_id, DataFrame]
    
    if 'well_list' not in st.session_state:
        st.session_state.well_list = []
    
    if 'current_well' not in st.session_state:
        st.session_state.current_well = None
    
    # Step results storage
    # Each step stores results as: Dict[well_id, result_data]
    if 'step0_results' not in st.session_state:
        st.session_state.step0_results = {}
    
    if 'step1_results' not in st.session_state:
        st.session_state.step1_results = {}
    
    if 'step2_results' not in st.session_state:
        st.session_state.step2_results = {}
    
    if 'step3_results' not in st.session_state:
        st.session_state.step3_results = {}
    
    if 'step4_results' not in st.session_state:
        st.session_state.step4_results = {}
    
    # Configuration
    if 'config' not in st.session_state:
        st.session_state.config = {}
    
    # Processing flags
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Error messages
    if 'error_messages' not in st.session_state:
        st.session_state.error_messages = []
    
    # Manual point selections for outlier/inlier toggling
    # Dict[well_id, Dict[algorithm, Dict['selected_outliers': Set[int], 'deselected_outliers': Set[int]]]]
    if 'manual_outlier_selections' not in st.session_state:
        st.session_state.manual_outlier_selections = {}
    
    # Adjusted fitting periods
    # Dict[well_id, Tuple[start_time, end_time]]
    if 'adjusted_fitting_periods' not in st.session_state:
        st.session_state.adjusted_fitting_periods = {}


def get_well_data(well_id: str) -> Optional[pd.DataFrame]:
    """
    Get data for a specific well.
    
    Args:
        well_id: Well identifier
    
    Returns:
        DataFrame for the well, or None if not found
    """
    return st.session_state.wells_data.get(well_id)


def set_well_data(well_id: str, data: pd.DataFrame):
    """
    Store data for a specific well.
    
    Args:
        well_id: Well identifier
        data: DataFrame to store
    """
    st.session_state.wells_data[well_id] = data
    if well_id not in st.session_state.well_list:
        st.session_state.well_list.append(well_id)


def get_step_result(step: str, well_id: str) -> Optional[Any]:
    """
    Get result for a specific step and well.
    
    Args:
        step: Step name (e.g., 'step0', 'step1', etc.)
        well_id: Well identifier
    
    Returns:
        Result data or None if not found
    """
    step_key = f'{step}_results'
    if step_key in st.session_state:
        return st.session_state[step_key].get(well_id)
    return None


def set_step_result(step: str, well_id: str, result: Any):
    """
    Store result for a specific step and well.
    
    Args:
        step: Step name (e.g., 'step0', 'step1', etc.)
        well_id: Well identifier
        result: Result data to store
    """
    step_key = f'{step}_results'
    if step_key not in st.session_state:
        st.session_state[step_key] = {}
    st.session_state[step_key][well_id] = result


def clear_step_results(step: str):
    """
    Clear all results for a specific step.
    
    Args:
        step: Step name (e.g., 'step0', 'step1', etc.)
    """
    step_key = f'{step}_results'
    if step_key in st.session_state:
        st.session_state[step_key] = {}


def clear_all_results():
    """Clear all step results."""
    for step in ['step0', 'step1', 'step2', 'step3', 'step4']:
        clear_step_results(step)


def has_step_result(step: str, well_id: str) -> bool:
    """
    Check if a step result exists for a well.
    
    Args:
        step: Step name
        well_id: Well identifier
    
    Returns:
        True if result exists, False otherwise
    """
    step_key = f'{step}_results'
    if step_key not in st.session_state:
        return False
    return well_id in st.session_state[step_key]


def get_current_well_data() -> Optional[pd.DataFrame]:
    """
    Get data for the currently selected well.
    
    Returns:
        DataFrame for current well, or None if no well selected
    """
    if st.session_state.current_well:
        return get_well_data(st.session_state.current_well)
    return None


def get_manual_outlier_selections(well_id: str, algorithm: str) -> Dict[str, set]:
    """
    Get manual outlier selections for a well and algorithm.
    
    Args:
        well_id: Well identifier
        algorithm: Algorithm name
    
    Returns:
        Dictionary with 'selected_outliers' and 'deselected_outliers' sets
    """
    if well_id not in st.session_state.manual_outlier_selections:
        return {'selected_outliers': set(), 'deselected_outliers': set()}
    if algorithm not in st.session_state.manual_outlier_selections[well_id]:
        return {'selected_outliers': set(), 'deselected_outliers': set()}
    return st.session_state.manual_outlier_selections[well_id][algorithm]


def set_manual_outlier_selections(well_id: str, algorithm: str, selections: Dict[str, set]):
    """
    Store manual outlier selections for a well and algorithm.
    
    Args:
        well_id: Well identifier
        algorithm: Algorithm name
        selections: Dictionary with 'selected_outliers' and 'deselected_outliers' sets
    """
    if well_id not in st.session_state.manual_outlier_selections:
        st.session_state.manual_outlier_selections[well_id] = {}
    st.session_state.manual_outlier_selections[well_id][algorithm] = selections


def clear_manual_outlier_selections(well_id: str, algorithm: Optional[str] = None):
    """
    Clear manual outlier selections for a well.
    
    Args:
        well_id: Well identifier
        algorithm: Optional algorithm name. If None, clears all algorithms for the well
    """
    if well_id in st.session_state.manual_outlier_selections:
        if algorithm is None:
            del st.session_state.manual_outlier_selections[well_id]
        elif algorithm in st.session_state.manual_outlier_selections[well_id]:
            del st.session_state.manual_outlier_selections[well_id][algorithm]


def get_adjusted_fitting_period(well_id: str) -> Optional[Tuple[float, float]]:
    """
    Get adjusted fitting period for a well.
    
    Args:
        well_id: Well identifier
    
    Returns:
        Tuple of (start_time, end_time) or None if not adjusted
    """
    return st.session_state.adjusted_fitting_periods.get(well_id)


def set_adjusted_fitting_period(well_id: str, period: Tuple[float, float]):
    """
    Store adjusted fitting period for a well.
    
    Args:
        well_id: Well identifier
        period: Tuple of (start_time, end_time)
    """
    st.session_state.adjusted_fitting_periods[well_id] = period


def clear_adjusted_fitting_period(well_id: str):
    """
    Clear adjusted fitting period for a well.
    
    Args:
        well_id: Well identifier
    """
    if well_id in st.session_state.adjusted_fitting_periods:
        del st.session_state.adjusted_fitting_periods[well_id]

