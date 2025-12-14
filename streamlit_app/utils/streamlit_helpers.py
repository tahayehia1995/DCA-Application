"""
Streamlit-specific visualization and utility functions
Provides Plotly-based interactive visualizations for the DCA workflow
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any


def create_plotly_preprocessing(
    original_df: pd.DataFrame,
    processed_df: Optional[pd.DataFrame] = None,
    zeros_removed: Optional[pd.DataFrame] = None,
    well_type: str = 'oil'
) -> go.Figure:
    """
    Create Plotly figure for preprocessing step.
    
    Args:
        original_df: Original DataFrame with 't' and 'q_actual'
        processed_df: Processed DataFrame (if None, uses original_df)
        zeros_removed: DataFrame with removed zero points
        well_type: 'oil' or 'gas' for axis labeling
    
    Returns:
        Plotly figure object
    """
    if processed_df is None:
        processed_df = original_df.copy()
    
    # Create subplots with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Determine labels (handle None values)
    well_type = well_type or 'oil'  # Default to 'oil' if None
    flow_rate_label = "Flow Rate (STB/D)" if well_type.lower() == 'oil' else "Flow Rate (MSCF/D)"
    cumulative_label = "Cumulative Production (bbl)" if well_type.lower() == 'oil' else "Cumulative Production (Mscf)"
    
    # Plot original data
    fig.add_trace(
        go.Scatter(
            x=original_df['t'],
            y=original_df['q_actual'],
            mode='lines+markers',
            name='Original Data',
            marker=dict(color='blue', size=4),
            line=dict(color='blue', width=1)
        ),
        secondary_y=False
    )
    
    # Plot zeros removed if provided
    if zeros_removed is not None and len(zeros_removed) > 0:
        fig.add_trace(
            go.Scatter(
                x=zeros_removed['t'],
                y=zeros_removed['q_actual'],
                mode='markers',
                name='Zeros Removed',
                marker=dict(color='red', size=6, symbol='x')
            ),
            secondary_y=False
        )
    
    # Calculate and plot cumulative production
    if 'Gp_actual' not in processed_df.columns:
        processed_df['Gp_actual'] = processed_df['q_actual'].cumsum()
    
    fig.add_trace(
        go.Scatter(
            x=processed_df['t'],
            y=processed_df['Gp_actual'],
            mode='lines',
            name='Cumulative Production',
            line=dict(color='red', width=2)
        ),
        secondary_y=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (Months)")
    fig.update_yaxes(title_text=flow_rate_label, type="log", secondary_y=False)
    fig.update_yaxes(title_text=cumulative_label, secondary_y=True)
    
    fig.update_layout(
        title="Preprocessing Results",
        hovermode='closest',
        height=500,
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=1.01, orientation="v")
    )
    
    return fig


def create_plotly_noise_removal(
    original_df: pd.DataFrame,
    cleaned_data_dict: Dict[str, pd.DataFrame],
    window_outliers: Optional[pd.DataFrame] = None,
    algorithm_names: Optional[List[str]] = None,
    well_id: Optional[str] = None,
    selected_algorithm: Optional[str] = None,
    manual_selections: Optional[Dict[str, set]] = None
) -> go.Figure:
    """
    Create Plotly figure for noise removal step with multiple algorithm results.
    
    Args:
        original_df: Original DataFrame
        cleaned_data_dict: Dictionary mapping algorithm names to cleaned DataFrames
        window_outliers: DataFrame with window-based outliers
        algorithm_names: List of algorithm names to display
        well_id: Well identifier for storing selections
        selected_algorithm: Currently selected algorithm for point editing
        manual_selections: Dictionary with 'selected_outliers' and 'deselected_outliers' sets
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Color palette for algorithms
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Create index mapping for original data
    x_col = original_df.columns[0]
    y_col = original_df.columns[1]
    
    # Use the DataFrame index as the point identifier
    original_indices = original_df.index.values
    
    # Plot original data with indices in customdata for selection
    fig.add_trace(
        go.Scatter(
            x=original_df.iloc[:, 0],
            y=original_df.iloc[:, 1],
            mode='markers',
            name='Original Data',
            marker=dict(color='blue', size=4, opacity=0.6),
            customdata=original_indices,
            hovertemplate=f'<b>%{{fullData.name}}</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>Index: %{{customdata}}<extra></extra>'
        )
    )
    
    # Plot window outliers
    if window_outliers is not None and len(window_outliers) > 0:
        fig.add_trace(
            go.Scatter(
                x=window_outliers.iloc[:, 0],
                y=window_outliers.iloc[:, 1],
                mode='markers',
                name='Window Outliers',
                marker=dict(color='green', size=8, symbol='x')
            )
        )
    
    # Plot cleaned data for each algorithm
    if algorithm_names is None:
        algorithm_names = list(cleaned_data_dict.keys())
    
    for idx, algo_name in enumerate(algorithm_names):
        if algo_name in cleaned_data_dict:
            cleaned_df = cleaned_data_dict[algo_name]
            color = colors[idx % len(colors)]
            
            # Get manual selections for this algorithm
            selected_outliers = set()
            deselected_outliers = set()
            if manual_selections:
                selected_outliers = manual_selections.get('selected_outliers', set())
                deselected_outliers = manual_selections.get('deselected_outliers', set())
            
            # Find which points are in cleaned data vs original
            # Match points by their (x, y) coordinates and track original indices
            cleaned_indices = []
            cleaned_x = []
            cleaned_y = []
            
            for orig_idx in original_df.index:
                row = original_df.loc[orig_idx]
                x_val = row[x_col]
                y_val = row[y_col]
                # Check if this point exists in cleaned data
                matching = cleaned_df[(cleaned_df.iloc[:, 0] == x_val) & (cleaned_df.iloc[:, 1] == y_val)]
                if len(matching) > 0:
                    cleaned_indices.append(orig_idx)
                    cleaned_x.append(x_val)
                    cleaned_y.append(y_val)
            
            # Separate points based on manual selections
            normal_points_x = []
            normal_points_y = []
            selected_points_x = []
            selected_points_y = []
            deselected_points_x = []
            deselected_points_y = []
            
            for i, orig_idx in enumerate(cleaned_indices):
                if orig_idx in selected_outliers:
                    selected_points_x.append(cleaned_x[i])
                    selected_points_y.append(cleaned_y[i])
                elif orig_idx in deselected_outliers:
                    deselected_points_x.append(cleaned_x[i])
                    deselected_points_y.append(cleaned_y[i])
                else:
                    normal_points_x.append(cleaned_x[i])
                    normal_points_y.append(cleaned_y[i])
            
            # Plot normal cleaned points
            if len(normal_points_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=normal_points_x,
                        y=normal_points_y,
                        mode='markers',
                        name=f'Cleaned ({algo_name})',
                        marker=dict(color=color, size=3, opacity=0.7),
                        visible='legendonly' if idx > 0 else True
                    )
                )
            
            # Plot manually selected outliers (marked as outliers)
            if len(selected_points_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=selected_points_x,
                        y=selected_points_y,
                        mode='markers',
                        name=f'Manually Selected Outliers ({algo_name})',
                        marker=dict(color='red', size=10, symbol='star', opacity=0.9),
                        visible='legendonly' if idx > 0 else True
                    )
                )
            
            # Plot manually deselected outliers (marked as inliers)
            if len(deselected_points_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=deselected_points_x,
                        y=deselected_points_y,
                        mode='markers',
                        name=f'Manually Deselected ({algo_name})',
                        marker=dict(color='lime', size=8, symbol='circle', opacity=0.9),
                        visible='legendonly' if idx > 0 else True
                    )
                )
    
    fig.update_layout(
        title="Noise Removal Results (Click points to toggle outlier/inlier status)",
        xaxis_title=original_df.columns[0],
        yaxis_title=original_df.columns[1],
        yaxis_type="log",
        hovermode='closest',
        height=500,
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=1.01, orientation="v"),
        clickmode='event+select'
    )
    
    return fig


def create_plotly_interpolation(
    original_df: pd.DataFrame,
    interpolated_df: pd.DataFrame,
    interpolated_flags: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Create Plotly figure for interpolation step.
    
    Args:
        original_df: Original DataFrame with gaps
        interpolated_df: Interpolated DataFrame with continuous time
        interpolated_flags: Boolean array indicating interpolated points
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot all interpolated data as "Original"
    fig.add_trace(
        go.Scatter(
            x=interpolated_df['t'],
            y=interpolated_df['q_actual'],
            mode='markers',
            name='Original Data',
            marker=dict(color='blue', size=4)
        )
    )
    
    # Highlight interpolated points
    if interpolated_flags is not None and len(interpolated_flags) > 0:
        if len(interpolated_flags) == len(interpolated_df):
            interpolated_mask = interpolated_flags.astype(bool)
            if np.any(interpolated_mask):
                fig.add_trace(
                    go.Scatter(
                        x=interpolated_df.loc[interpolated_mask, 't'],
                        y=interpolated_df.loc[interpolated_mask, 'q_actual'],
                        mode='markers',
                        name='Interpolated Points',
                        marker=dict(color='red', size=6)
                    )
                )
    
    # Plot continuous line
    fig.add_trace(
        go.Scatter(
            x=interpolated_df['t'],
            y=interpolated_df['q_actual'],
            mode='lines',
            name='Continuous Series',
            line=dict(color='green', width=2),
            opacity=0.5
        )
    )
    
    fig.update_layout(
        title="Interpolation Results",
        xaxis_title="Time (months)",
        yaxis_title="Flow Rate (q_actual)",
        yaxis_type="log",
        hovermode='closest',
        height=500,
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=1.01, orientation="v")
    )
    
    return fig


def create_plotly_smoothing(
    data: pd.DataFrame,
    filter_columns: List[str],
    raw_column: str = 'q_actual'
) -> go.Figure:
    """
    Create Plotly figure comparing smoothing filters.
    
    Args:
        data: DataFrame with 't' and filter columns
        filter_columns: List of filter column names
        raw_column: Name of raw data column
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot raw data
    if raw_column in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['t'],
                y=data[raw_column],
                mode='lines+markers',
                name='Raw Data',
                marker=dict(size=3, opacity=0.5),
                line=dict(color='blue', width=1)
            )
        )
    
    # Color palette for filters
    colors = ['red', 'green', 'orange', 'purple']
    
    # Plot each filter
    for idx, filter_col in enumerate(filter_columns):
        if filter_col in data.columns:
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=data['t'],
                    y=data[filter_col],
                    mode='lines',
                    name=filter_col,
                    line=dict(color=color, width=2),
                    visible='legendonly' if idx > 0 else True
                )
            )
    
    fig.update_layout(
        title="Smoothing Comparison",
        xaxis_title="Time (t)",
        yaxis_title="Flow Rate",
        hovermode='closest',
        height=500,
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=1.01, orientation="v")
    )
    
    return fig


def create_plotly_dca_forecast(
    t: np.ndarray,
    q_actual: np.ndarray,
    forecast_curves: Dict[str, Dict],
    fitting_period: Optional[Tuple[float, float]] = None,
    percentiles: Optional[Tuple] = None,
    well_type: str = 'oil'
) -> go.Figure:
    """
    Create Plotly figure for DCA fitting and forecasting.
    
    Args:
        t: Time array for historical data
        q_actual: Actual production rate
        forecast_curves: Dictionary of forecast curves with keys like '{filter} {model_name}'
        fitting_period: Tuple of (start_time, end_time) for fitting period
        percentiles: Tuple of (p10, p50, p90, average, common_t) arrays
        well_type: 'oil' or 'gas' for axis labeling
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Determine y-axis label (handle None values)
    well_type = well_type or 'oil'  # Default to 'oil' if None
    ylabel = "Flow Rate (STB/D)" if well_type.lower() == 'oil' else "Flow Rate (MSCF/D)"
    
    # Plot historical data
    fig.add_trace(
        go.Scatter(
            x=t,
            y=q_actual,
            mode='markers',
            name='Historical Data',
            marker=dict(color='blue', size=4)
        )
    )
    
    # Highlight fitting period
    if fitting_period:
        start, end = fitting_period
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="yellow",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Fitting Period"
        )
    
    # Color palette for models
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    # Plot forecast curves
    for idx, (key, model_data) in enumerate(forecast_curves.items()):
        if 'combined_t' in model_data and 'combined_q' in model_data:
            color = colors[idx % len(colors)]
            combined_t = model_data['combined_t']
            combined_q = model_data['combined_q']
            
            # Split into fitted and forecasted portions
            if 'forecast_t' in model_data:
                forecast_start_idx = len(combined_t) - len(model_data['forecast_t'])
                fitted_t = combined_t[:forecast_start_idx]
                fitted_q = combined_q[:forecast_start_idx]
                forecast_t = combined_t[forecast_start_idx:]
                forecast_q = combined_q[forecast_start_idx:]
                
                # Plot fitted portion
                fig.add_trace(
                    go.Scatter(
                        x=fitted_t,
                        y=fitted_q,
                        mode='lines',
                        name=f'{key} (Fitted)',
                        line=dict(color=color, width=2),
                        visible='legendonly' if idx > 0 else True
                    )
                )
                
                # Plot forecasted portion (dashed)
                fig.add_trace(
                    go.Scatter(
                        x=forecast_t,
                        y=forecast_q,
                        mode='lines',
                        name=f'{key} (Forecast)',
                        line=dict(color=color, width=2, dash='dash'),
                        visible='legendonly' if idx > 0 else True
                    )
                )
            else:
                # Plot combined curve
                fig.add_trace(
                    go.Scatter(
                        x=combined_t,
                        y=combined_q,
                        mode='lines',
                        name=key,
                        line=dict(color=color, width=2),
                        visible='legendonly' if idx > 0 else True
                    )
                )
    
    # Plot percentiles if provided
    if percentiles:
        p10, p50, p90, average, common_t = percentiles
        
        fig.add_trace(
            go.Scatter(
                x=common_t,
                y=p10,
                mode='lines',
                name='P10',
                line=dict(color='red', width=3),
                visible='legendonly'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=common_t,
                y=p50,
                mode='lines',
                name='P50',
                line=dict(color='green', width=3),
                visible='legendonly'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=common_t,
                y=p90,
                mode='lines',
                name='P90',
                line=dict(color='blue', width=3),
                visible='legendonly'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=common_t,
                y=average,
                mode='lines',
                name='Average',
                line=dict(color='black', width=3, dash='dot'),
                visible='legendonly'
            )
        )
    
    fig.update_layout(
        title="Decline Curve Analysis - Forecasts",
        xaxis_title="Time (months)",
        yaxis_title=ylabel,
        yaxis_type="log",
        hovermode='closest',
        height=600,
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=1.01, orientation="v")
    )
    
    return fig


def create_summary_table(df: pd.DataFrame, highlight_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create styled summary table for display.
    
    Args:
        df: DataFrame to display
        highlight_col: Column name to highlight (e.g., best model)
    
    Returns:
        Styled DataFrame
    """
    # Return unstyled for now - Streamlit handles styling
    return df


def export_dataframe(df: pd.DataFrame, filename: str = "data.csv") -> bytes:
    """
    Convert DataFrame to CSV bytes for download.
    
    Args:
        df: DataFrame to export
        filename: Filename for download
    
    Returns:
        CSV bytes
    """
    return df.to_csv(index=False).encode('utf-8')


def create_fitting_period_plot(
    t: pd.Series,
    q_actual: pd.Series,
    q_smoothed: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    best_start: float,
    best_end: float,
    adjusted_period: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Create Plotly figure for fitting period identification.
    
    Args:
        t: Time series
        q_actual: Actual production rate
        q_smoothed: Smoothed LOWESS values
        peaks: Array of peak indices
        troughs: Array of trough indices
        best_start: Start time of best fitting period
        best_end: End time of best fitting period
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot q_actual
    fig.add_trace(
        go.Scatter(
            x=t,
            y=q_actual,
            mode='lines',
            name='q_actual',
            line=dict(color='blue', width=1)
        )
    )
    
    # Plot smoothed curve
    fig.add_trace(
        go.Scatter(
            x=t,
            y=q_smoothed,
            mode='lines',
            name='Smoothed (LOWESS)',
            line=dict(color='orange', width=2, dash='dash')
        )
    )
    
    # Plot peaks
    if len(peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=t.iloc[peaks],
                y=q_smoothed[peaks],
                mode='markers',
                name='Peaks',
                marker=dict(color='red', size=10, symbol='x')
            )
        )
    
    # Plot troughs
    if len(troughs) > 0:
        fig.add_trace(
            go.Scatter(
                x=t.iloc[troughs],
                y=q_smoothed[troughs],
                mode='markers',
                name='Troughs',
                marker=dict(color='blue', size=10, symbol='circle')
            )
        )
    
    # Use adjusted period if provided, otherwise use auto-detected
    display_start = adjusted_period[0] if adjusted_period else best_start
    display_end = adjusted_period[1] if adjusted_period else best_end
    period_label = "Adjusted Fitting Period" if adjusted_period else "Auto-Detected Fitting Period"
    
    # Highlight fitting period
    fig.add_vrect(
        x0=display_start,
        x1=display_end,
        fillcolor="yellow",
        opacity=0.3,
        layer="below",
        line_width=2,
        line_color="orange" if adjusted_period else "yellow",
        annotation_text=f"{period_label}: {display_start:.1f} - {display_end:.1f}"
    )
    
    # Show auto-detected period if different from adjusted
    if adjusted_period and (best_start != adjusted_period[0] or best_end != adjusted_period[1]):
        fig.add_vrect(
            x0=best_start,
            x1=best_end,
            fillcolor="lightblue",
            opacity=0.2,
            layer="below",
            line_width=1,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Auto-Detected: {best_start:.1f} - {best_end:.1f}"
        )
    
    fig.update_layout(
        title="Fitting Period Identification",
        xaxis_title="Time (t)",
        yaxis_title="Flow Rate (q)",
        yaxis_type="log",
        hovermode='closest',
        height=500,
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=1.01, orientation="v")
    )
    
    return fig

