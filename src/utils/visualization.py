"""
Visualization utilities for production data analysis
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def setup_plot_style():
    """Set up consistent plot styling"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_preprocessing(data: pd.DataFrame, title: str = "Preprocessing Results",
                       save_path: Optional[str] = None, show: bool = True,
                       well_type: str = 'oil'):
    """
    Plot preprocessing results with dual-axis: log scale flow rate and cumulative production.
    Matches original notebook style.
    
    Args:
        data: DataFrame with 't' and 'q_actual' columns
        title: Plot title
        save_path: Path to save figure (None to skip saving)
        show: Whether to display the plot
        well_type: 'oil' or 'gas' for axis labeling
    """
    setup_plot_style()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative production
    if 'Gp_actual' not in data.columns:
        data['Gp_actual'] = data['q_actual'].cumsum()
    
    # Determine labels based on well type
    flow_rate_label = "Flow Rate (bbl/d)" if well_type == 'OIL' else "Flow Rate (Mscf/d)"
    cumulative_label = "Cumulative Production (bbl)" if well_type == 'OIL' else "Cumulative Production (Mscf)"
    
    # Plot flow rate on left y-axis (log scale)
    ax1.set_xlabel('Time (Months)')
    ax1.set_ylabel(flow_rate_label, color='blue')
    ax1.plot(data['t'], data['q_actual'], 'b-', label='Flow Rate')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary y-axis for cumulative production
    ax2 = ax1.twinx()
    ax2.set_ylabel(cumulative_label, color='red')
    ax2.plot(data['t'], data['Gp_actual'], 'r-', label='Cumulative Production')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(title)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_noise_removal(original_data: pd.DataFrame, cleaned_data: pd.DataFrame,
                      outliers_data: Optional[pd.DataFrame] = None,
                      title: str = "Noise Removal Results",
                      save_path: Optional[str] = None, show: bool = True,
                      algorithm: str = 'all'):
    """
    Plot noise removal results showing original, outliers, and cleaned data.
    Matches original notebook style: original (blue), window outliers (green), cleaned (red).
    
    Args:
        original_data: Original DataFrame
        cleaned_data: Cleaned DataFrame
        outliers_data: DataFrame with window outlier points (green)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
        algorithm: Algorithm name for title
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x_col = original_data.columns[0]
    y_col = original_data.columns[1]
    
    # Plot original data (blue) - matches notebook style
    ax.scatter(original_data[x_col], original_data[y_col], 
              color='blue', s=30, alpha=0.6, 
              label='Commonly Detected Outliers by All Algorithms')
    
    # Plot window outliers (green) - matches notebook style
    if outliers_data is not None and len(outliers_data) > 0:
        ax.scatter(outliers_data[x_col], outliers_data[y_col],
                  color='green', s=50, marker='o', label='Window Outliers', zorder=5)
    
    # Plot cleaned data (red) - matches notebook style
    ax.scatter(cleaned_data[x_col], cleaned_data[y_col],
              color='red', s=20, alpha=0.8, label='Cleaned Data')
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_yscale('log')
    ax.set_title(f'Outlier Removal using {algorithm}')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_interpolation(original_data: pd.DataFrame, interpolated_data: pd.DataFrame,
                      interpolated_flags: Optional[np.ndarray] = None,
                      title: str = "Interpolation Results",
                      save_path: Optional[str] = None, show: bool = True):
    """
    Plot interpolation results. Matches original notebook style exactly:
    - Plots interpolated data as "Original" (all points in blue)
    - Highlights interpolated points in red
    
    Args:
        original_data: Can be same as interpolated_data (original notebook passes same data twice)
        interpolated_data: Interpolated DataFrame with continuous time
        interpolated_flags: Boolean array indicating which points were interpolated
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_col = interpolated_data.columns[0] if 't' not in interpolated_data.columns else 't'
    y_col = interpolated_data.columns[1] if 'q_actual' not in interpolated_data.columns else 'q_actual'
    
    t = interpolated_data[x_col].values
    q_interpolated = interpolated_data[y_col].values
    
    # Plot all interpolated data points as "Original" (blue dots) - matches notebook
    # Note: Original notebook passes same data twice, so "Original" label shows all data
    ax.semilogy(t, q_interpolated, 'bo', label='Original', markersize=4)
    
    # Plot interpolated points (red dots) - matches notebook
    if interpolated_flags is not None and len(interpolated_flags) > 0:
        # Ensure flags are boolean array and match data length
        if len(interpolated_flags) == len(t):
            interpolated_mask = interpolated_flags.astype(bool)
        elif len(interpolated_flags) < len(t):
            # Pad with False if needed
            interpolated_mask = np.pad(interpolated_flags.astype(bool), 
                                      (0, len(t) - len(interpolated_flags)), 
                                      constant_values=False)
        else:
            # Trim if needed
            interpolated_mask = interpolated_flags[:len(t)].astype(bool)
        
        # Only plot if there are actually interpolated points
        if np.any(interpolated_mask):
            ax.semilogy(t[interpolated_mask], q_interpolated[interpolated_mask], 
                       'ro', label='Interpolated', markersize=4)
    # Note: If no flags or no interpolated points, we don't plot red dots (matches original behavior)
    
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Flow rate (q_actual)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_fitting_period(t: pd.Series, q_actual: pd.Series, q_smoothed_lowess: np.ndarray,
                       peaks: np.ndarray, troughs: np.ndarray,
                       best_start: float, best_end: float,
                       title: str = "Best Fitting Period",
                       save_path: Optional[str] = None, show: bool = True):
    """
    Plot fitting period identification. Matches original notebook style:
    q_actual, smoothed LOWESS curve, peaks (x), troughs (o), highlighted fitting period.
    
    Args:
        t: Time series
        q_actual: Actual production rate
        q_smoothed_lowess: LOWESS smoothed values
        peaks: Array of peak indices
        troughs: Array of trough indices
        best_start: Start time of best fitting period
        best_end: End time of best fitting period
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot q_actual
    ax.semilogy(t, q_actual, label='q_actual')
    
    # Plot smoothed LOWESS curve
    ax.semilogy(t, q_smoothed_lowess, label='Smoothed curve (LOWESS)', linestyle='-.')
    
    # Plot peaks (x markers, red) - matches notebook
    if len(peaks) > 0:
        ax.scatter(t.iloc[peaks], q_smoothed_lowess[peaks], 
                  marker='x', color='red', label='Peaks', s=100)
    
    # Plot troughs (o markers, blue) - matches notebook
    if len(troughs) > 0:
        ax.scatter(t.iloc[troughs], q_smoothed_lowess[troughs], 
                  marker='o', color='blue', label='Troughs', s=100)
    
    # Highlight best fitting period (yellow) - matches notebook
    ax.axvspan(best_start, best_end, color='yellow', alpha=0.3, label='Best fitting period')
    
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Flow rate (q)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_smoothing(data: pd.DataFrame, filter_columns: List[str],
                  title: str = "Smoothing Results",
                  save_path: Optional[str] = None, show: bool = True):
    """
    Plot smoothing results comparing different filters.
    Matches original notebook style: all filters plotted with original data.
    
    Args:
        data: DataFrame with 't', 'q_actual', and filter columns
        filter_columns: List of column names containing filtered data
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    t = data['t']
    
    # Plot each filter column - matches notebook style
    for column in filter_columns:
        if column in data.columns:
            ax.plot(t, data[column].iloc[:len(t)], label=column, linewidth=2)
    
    # Extract well name from title if available
    well_name = title.split('Well: ')[-1] if 'Well: ' in title else title.split(' for ')[-1] if ' for ' in title else ""
    if well_name:
        ax.set_title(f"Well: {well_name}")
    else:
        ax.set_title(title)
    
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Flow Rate (q_actual)')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_decline_curves(t: np.ndarray, q_actual: np.ndarray,
                       fitted_models: Optional[Dict[str, Dict]] = None,
                       fitting_period: Optional[Tuple[int, int]] = None,
                       forecast_curves: Optional[Dict[str, Dict]] = None,
                       percentiles: Optional[Tuple] = None,
                       title: str = "Decline Curve Forecasts",
                       well_type: str = 'oil',
                       save_path: Optional[str] = None, show: bool = True):
    """
    Plot decline curve analysis results with fitted models and forecasts.
    Matches original notebook style exactly.
    
    Args:
        t: Time array
        q_actual: Actual production rate
        fitted_models: Dictionary of fitted model data with keys like '{filter} {model_name}'
        fitting_period: Tuple of (start_time, end_time) for fitting
        forecast_curves: Dictionary of forecasted curves with 'forecasted_t' and 'forecasted_q'
        percentiles: Tuple of (p10, p50, p90, average) arrays and common_t
        title: Plot title
        well_type: 'oil' or 'gas' for axis labeling
        save_path: Path to save figure
        show: Whether to display the plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot q_actual (blue dots) - matches notebook
    ax.semilogy(t, q_actual, 'o', label='q_actual', markersize=3)
    
    # Highlight fitting period (yellow) - matches notebook
    if fitting_period:
        start, end = fitting_period
        ax.axvspan(start, end, color='yellow', alpha=0.3)
    
    # Plot fitted and forecasted curves for each model - matches notebook
    if forecast_curves:
        for key, model_data in forecast_curves.items():
            if 'combined_t' in model_data and 'combined_q' in model_data:
                combined_t = model_data['combined_t']
                combined_q = model_data['combined_q']
                ax.semilogy(combined_t, combined_q, label=key, linewidth=1)
    
    # Plot percentiles - matches notebook style
    if percentiles:
        p10, p50, p90, average, common_t = percentiles
        ax.semilogy(common_t, p10, label='P90', linewidth=3, color='red')
        ax.semilogy(common_t, p50, label='P50', linewidth=3, color='green')
        ax.semilogy(common_t, p90, label='P10', linewidth=3, color='blue')
        ax.semilogy(common_t, average, label='Average', linewidth=3, 
                   linestyle='--', color='black')
    
    # Labels - matches notebook
    if 'gas' in title.lower() or well_type == 'gas':
        ax.set_ylabel('Flow Rate (MSCF/D)')
    elif 'oil' in title.lower() or well_type == 'oil':
        ax.set_ylabel('Flow Rate (STB/D)')
    else:
        ax.set_ylabel('Flow Rate')
    
    ax.set_xlabel('Time (months)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--")
    # Note: Legend commented out in original notebook, but we'll include it
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_figures(figures_dict: Dict[str, plt.Figure], output_dir: str,
                format: str = 'png', dpi: int = 300):
    """
    Save multiple figures to a directory.
    
    Args:
        figures_dict: Dictionary mapping filenames to Figure objects
        output_dir: Output directory path
        format: Image format (png, pdf, svg, etc.)
        dpi: Image resolution
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, fig in figures_dict.items():
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
        print(f"Saved: {output_path}")
        plt.close(fig)


def create_summary_plot(results_data: List[Dict], output_path: Optional[str] = None,
                       show: bool = True):
    """
    Create a summary plot showing key statistics across all wells.
    
    Args:
        results_data: List of dictionaries with well results
        output_path: Path to save figure
        show: Whether to display the plot
    """
    if not results_data:
        print("No data to plot")
        return
    
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Workflow Summary Statistics', fontsize=16)
    
    # Extract data for plotting
    well_names = [r.get('well_name', f"Well {i}") for i, r in enumerate(results_data)]
    
    # Plot 1: Number of data points per step
    ax = axes[0, 0]
    if all('data_points' in r for r in results_data):
        data_points = [r['data_points'] for r in results_data]
        ax.bar(range(len(well_names)), data_points)
        ax.set_xlabel('Wells')
        ax.set_ylabel('Number of Data Points')
        ax.set_title('Data Points per Well')
        ax.set_xticks(range(len(well_names)))
        ax.set_xticklabels(well_names, rotation=45, ha='right')
    
    # Plot 2: Model fit quality
    ax = axes[0, 1]
    if all('r_squared' in r for r in results_data):
        r_squared = [r['r_squared'] for r in results_data]
        ax.bar(range(len(well_names)), r_squared, color='green')
        ax.set_xlabel('Wells')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Fit Quality')
        ax.set_xticks(range(len(well_names)))
        ax.set_xticklabels(well_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
    
    # Plot 3: EUR comparison
    ax = axes[1, 0]
    if all('eur' in r for r in results_data):
        eur = [r['eur'] for r in results_data]
        ax.bar(range(len(well_names)), eur, color='orange')
        ax.set_xlabel('Wells')
        ax.set_ylabel('EUR')
        ax.set_title('Estimated Ultimate Recovery')
        ax.set_xticks(range(len(well_names)))
        ax.set_xticklabels(well_names, rotation=45, ha='right')
    
    # Plot 4: Processing summary
    ax = axes[1, 1]
    ax.text(0.1, 0.9, f"Total Wells Processed: {len(results_data)}", 
           transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.7, f"Successful Fits: {sum(1 for r in results_data if r.get('success', False))}", 
           transform=ax.transAxes, fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
