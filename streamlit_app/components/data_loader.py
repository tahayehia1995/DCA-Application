"""
Data loading and parsing utilities for CSV files
Handles file upload, well ID extraction, and data organization
"""

import pandas as pd
import io
import re
import os
from typing import List, Dict, Tuple, Optional
import streamlit as st
from src.utils.file_handler import get_well_id, get_file_info


def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that CSV has required columns.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "CSV file is empty"
    
    # Check for required columns
    required_cols = ['t', 'q_actual']
    
    # Try to identify columns if they don't have standard names
    if len(df.columns) < 2:
        return False, f"CSV must have at least 2 columns. Found {len(df.columns)}"
    
    # If columns don't have standard names, try to rename
    if 't' not in df.columns or 'q_actual' not in df.columns:
        # Try to rename first two columns
        df.columns = ['t', 'q_actual'] + list(df.columns[2:])
    
    # Check data types
    try:
        df['t'] = pd.to_numeric(df['t'], errors='coerce')
        df['q_actual'] = pd.to_numeric(df['q_actual'], errors='coerce')
    except Exception as e:
        return False, f"Error converting data types: {str(e)}"
    
    # Check for too many NaN values
    if df['t'].isna().sum() > len(df) * 0.5:
        return False, "Too many missing values in 't' column"
    
    if df['q_actual'].isna().sum() > len(df) * 0.5:
        return False, "Too many missing values in 'q_actual' column"
    
    return True, ""


def parse_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
    """
    Parse an uploaded CSV file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Tuple of (DataFrame, well_id, error_message)
    """
    try:
        # Read CSV file
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
        
        # Drop second row if it exists (often contains units)
        if len(df) > 1:
            df = df.drop(df.index[1])
        
        # Validate format
        is_valid, error_msg = validate_csv_format(df)
        if not is_valid:
            return None, None, error_msg
        
        # Extract well ID from filename
        well_id = get_well_id(uploaded_file.name)
        
        # If no well ID found, use filename without extension
        if not well_id:
            well_id = os.path.splitext(uploaded_file.name)[0]
        
        return df, well_id, ""
    
    except Exception as e:
        return None, None, f"Error parsing file {uploaded_file.name}: {str(e)}"


def load_uploaded_files(uploaded_files: List) -> Dict[str, pd.DataFrame]:
    """
    Load and parse multiple uploaded CSV files.
    
    Args:
        uploaded_files: List of Streamlit uploaded file objects
    
    Returns:
        Dictionary mapping well_id to DataFrame
    """
    wells_data = {}
    errors = []
    
    for uploaded_file in uploaded_files:
        df, well_id, error_msg = parse_uploaded_file(uploaded_file)
        
        if df is not None and well_id:
            # Handle duplicate well IDs by appending suffix
            original_well_id = well_id
            counter = 1
            while well_id in wells_data:
                well_id = f"{original_well_id}_{counter}"
                counter += 1
            
            wells_data[well_id] = df
        else:
            errors.append(f"{uploaded_file.name}: {error_msg}")
    
    if errors:
        st.warning(f"Some files failed to load:\n" + "\n".join(errors))
    
    return wells_data


def extract_well_ids(filenames: List[str]) -> List[str]:
    """
    Extract well IDs from a list of filenames.
    
    Args:
        filenames: List of filename strings
    
    Returns:
        List of well IDs
    """
    well_ids = []
    for filename in filenames:
        well_id = get_well_id(filename)
        if well_id:
            well_ids.append(well_id)
    return well_ids


def organize_by_well(wells_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Organize well data with metadata.
    
    Args:
        wells_data: Dictionary mapping well_id to DataFrame
    
    Returns:
        Dictionary with well metadata
    """
    organized = {}
    
    for well_id, df in wells_data.items():
        file_info = get_file_info(well_id)
        
        organized[well_id] = {
            'data': df,
            'well_id': well_id,
            'well_type': file_info.get('well_type', 'unknown'),
            'interval': file_info.get('interval'),
            'num_points': len(df),
            'time_range': (df['t'].min(), df['t'].max()) if 't' in df.columns else None,
            'q_range': (df['q_actual'].min(), df['q_actual'].max()) if 'q_actual' in df.columns else None,
        }
    
    return organized


def get_well_summary(well_id: str, df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for a well.
    
    Args:
        well_id: Well identifier
        df: Well DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'well_id': well_id,
        'num_points': len(df),
        'time_range': (df['t'].min(), df['t'].max()) if 't' in df.columns else None,
        'q_mean': df['q_actual'].mean() if 'q_actual' in df.columns else None,
        'q_std': df['q_actual'].std() if 'q_actual' in df.columns else None,
        'q_min': df['q_actual'].min() if 'q_actual' in df.columns else None,
        'q_max': df['q_actual'].max() if 'q_actual' in df.columns else None,
        'zero_count': (df['q_actual'] == 0).sum() if 'q_actual' in df.columns else 0,
    }
    
    return summary

