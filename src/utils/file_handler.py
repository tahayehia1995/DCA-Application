"""
File handling utilities for loading, saving, and managing CSV files
"""

import os
import pandas as pd
import glob
import re
from typing import List, Dict, Optional, Tuple


def get_well_id(filename: str) -> str:
    """
    Extract well ID from filename or return base filename.
    
    Args:
        filename: Filename to parse
    
    Returns:
        Well ID string (API format if found, otherwise base filename without extension)
    
    Examples:
        >>> get_well_id("interval_475_rates_oil_42-495-33759_clear.csv")
        '42-495-33759'
        >>> get_well_id("my_well_data.csv")
        'my_well_data'
    """
    # Try to match API/UWI format (XX-XXX-XXXXX)
    pattern = r'(\d{2}-\d{3}-\d{5})'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)
    
    # If no API format found, return base filename without extension
    base_name = os.path.splitext(filename)[0]
    # Remove any leading numbers and dots (like "1. " from ranked files)
    base_name = re.sub(r'^\d+\.\s*', '', base_name)
    return base_name


def load_csv_files(directory: str, pattern: str = '*.csv', 
                   well_ids: List[str] = None,
                   exclude_patterns: List[str] = None) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load CSV files from a directory with optional filtering.
    
    Args:
        directory: Directory containing CSV files
        pattern: Glob pattern for file matching
        well_ids: List of well IDs to include (None or ['all'] for all files)
        exclude_patterns: List of patterns to exclude from filenames
    
    Returns:
        List of tuples (filename, dataframe)
    """
    # Find all matching files
    file_pattern = os.path.join(directory, pattern)
    all_files = glob.glob(file_pattern)
    
    # Apply exclusion patterns
    if exclude_patterns:
        filtered_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if not any(excl in filename for excl in exclude_patterns):
                filtered_files.append(file_path)
        all_files = filtered_files
    
    # Filter by well IDs if specified
    if well_ids and well_ids != ['all']:
        filtered_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            well_id = get_well_id(filename)
            if well_id and well_id in well_ids:
                filtered_files.append(file_path)
        all_files = filtered_files
    
    # Load files
    loaded_files = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            loaded_files.append((filename, df))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return loaded_files


def save_results(data: pd.DataFrame, output_path: str, 
                create_dir: bool = True) -> bool:
    """
    Save DataFrame to CSV file.
    
    Args:
        data: DataFrame to save
        output_path: Output file path
        create_dir: Whether to create output directory if it doesn't exist
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if needed
        if create_dir:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        
        # Save file
        data.to_csv(output_path, index=False)
        return True
    
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")
        return False


def get_file_info(filename: str) -> Dict[str, str]:
    """
    Extract information from filename.
    
    Args:
        filename: Filename to parse
    
    Returns:
        Dictionary with extracted information
    """
    info = {
        'well_id': get_well_id(filename),
        'interval': None,
        'well_type': None,
        'processing_stage': []
    }
    
    # Extract interval number
    interval_match = re.search(r'interval_(\d+)', filename)
    if interval_match:
        info['interval'] = interval_match.group(1)
    
    # Extract well type (oil or gas)
    if '_oil_' in filename:
        info['well_type'] = 'oil'
    elif '_gas_' in filename:
        info['well_type'] = 'gas'
    
    # Identify processing stages
    if 'clear' in filename:
        info['processing_stage'].append('preprocessed')
    if any(algo in filename for algo in ['lof', 'knn', 'abod', 'cof', 'cluster', 'iforest']):
        info['processing_stage'].append('noise_removed')
    if 'interpolated' in filename:
        info['processing_stage'].append('interpolated')
    if 'smoothed' in filename:
        info['processing_stage'].append('smoothed')
    
    return info


def organize_files_by_well(directory: str, pattern: str = '*.csv') -> Dict[str, List[str]]:
    """
    Organize files by well ID.
    
    Args:
        directory: Directory containing files
        pattern: Glob pattern for file matching
    
    Returns:
        Dictionary mapping well IDs to lists of filenames
    """
    files = glob.glob(os.path.join(directory, pattern))
    organized = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        well_id = get_well_id(filename)
        
        if well_id:
            if well_id not in organized:
                organized[well_id] = []
            organized[well_id].append(filename)
    
    return organized


def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """
    Create standard directory structure for the workflow.
    
    Args:
        base_dir: Base output directory
    
    Returns:
        Dictionary mapping step names to directory paths
    """
    directories = {
        'step0': os.path.join(base_dir, 'step0'),
        'step1': os.path.join(base_dir, 'step1'),
        'step2': os.path.join(base_dir, 'step2'),
        'step3': os.path.join(base_dir, 'step3'),
        'step4': os.path.join(base_dir, 'step4'),
        'figures': os.path.join(base_dir, 'figures')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories
