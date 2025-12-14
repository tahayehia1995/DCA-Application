"""
Default configuration for Streamlit DCA app
Provides default parameter values matching config/config.json structure
"""


def get_default_config():
    """
    Get default configuration dictionary.
    
    Returns:
        Dictionary with default configuration values
    """
    return {
        "project_name": "Wolfcamp Production Analysis",
        
        "global_settings": {
            "base_output_directory": "output/",
            "save_mode": "per_step",
            "figure_format": "png",
            "figure_dpi": 300,
            "verbose": True
        },
        
        "step0_preprocessing": {
            "enabled": True,
            "input_directory": "Original/",
            "output_directory": "output/step0/",
            "drop_zeros": True,
            "save_intermediate": True,
            "wells_to_process": ["all"],
            "visualize_wells": []
        },
        
        "step1_noise_removal": {
            "enabled": True,
            "input_directory": "output/step0/",
            "output_directory": "output/step1/",
            "use_window": True,
            "window_size": 15,
            "step_size": 5,
            "window_method": "lowest_points",
            "num_lowest_points": 3,
            "algorithms": ["all"],
            "hyperparameters": {
                "knn": {
                    "fraction": 0.20,
                    "n_neighbors": 20
                },
                "lof": {
                    "fraction": 0.20,
                    "n_neighbors": 20
                },
                "abod": {
                    "fraction": 0.20,
                    "n_neighbors": 3
                },
                "cof": {
                    "fraction": 0.20,
                    "n_neighbors": 3
                },
                "cluster": {
                    "fraction": 0.20,
                    "n_clusters": 3
                },
                "iforest": {
                    "fraction": 0.20,
                    "n_estimators": 200
                }
            },
            "drop_repeated_files": True,
            "save_intermediate": True,
            "wells_to_process": ["all"],
            "visualize_wells": []
        },
        
        "step2_interpolation": {
            "enabled": True,
            "input_directory": "output/step1/",
            "output_directory": "output/step2/",
            "interpolation_method": "cubic",
            "min_fitting_period": 30,
            "save_intermediate": True,
            "wells_to_process": ["all"],
            "visualize_wells": []
        },
        
        "step3_smoothing": {
            "enabled": True,
            "input_directory": "output/step2/",
            "output_directory": "output/step3/",
            "filters": ["Gaussian", "Savitzky-Golay", "Spline", "Lowess"],
            "save_intermediate": True,
            "wells_to_process": ["all"],
            "visualize_wells": []
        },
        
        "step4_q_fitting": {
            "enabled": True,
            "input_directory": "output/step3/",
            "q_actual_directory": "output/step0/",
            "output_directory": "output/step4/",
            "forecast_method": 1,
            "forecast_end_time": 400,
            "forecast_end_flow_rate_gas": 100.0,
            "forecast_end_flow_rate_oil": 3.0,
            "fitting_method": 1,  # curve_fit + differential_evolution
            "lmfit_method": "leastsq",
            "models": [
                "Arps_Hyperbolic",
                "Logistic_Growth",
                "Stretched_Exponential",
                "Power_Law",
                "Duong",
                "Wang",
                "VDMA"
            ],
            "fitting_accuracy_threshold": 90,
            "prediction_accuracy_threshold": 80,
            "minimum_production_history": 3,
            "min_peak_rel_height": 0.2,
            "lowess_frac": 0.30,
            "min_months": 30,
            "save_intermediate": True,
            "save_summary": True,
            "summary_metrics": ["r_squared"],
            "wells_to_process": ["all"],
            "visualize_wells": []
        }
    }


def merge_config(user_config: dict, default_config: dict = None) -> dict:
    """
    Merge user configuration with defaults.
    
    Args:
        user_config: User-provided configuration
        default_config: Default configuration (uses get_default_config() if None)
    
    Returns:
        Merged configuration dictionary
    """
    if default_config is None:
        default_config = get_default_config()
    
    merged = default_config.copy()
    
    # Deep merge user config
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(value, merged[key])
        else:
            merged[key] = value
    
    return merged


def validate_config(config: dict) -> tuple[bool, str]:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Add validation logic here if needed
    return True, ""

