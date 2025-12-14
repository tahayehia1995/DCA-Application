"""
Flexible Workflow Runner

This module provides intelligent workflow execution that:
1. Checks which steps are enabled in config
2. Automatically determines input directories for each step based on previous step
3. Skips disabled steps
4. Handles the complete pipeline flexibly
"""

import json
import os
from typing import Dict, List, Optional


def get_latest_output_directory(base_dir: str = 'output') -> str:
    """
    Find the most recent step's output directory that contains files.
    
    Args:
        base_dir: Base output directory
    
    Returns:
        Path to the latest output directory with files
    """
    # Check steps in reverse order
    for step_num in range(4, -1, -1):
        step_dir = os.path.join(base_dir, f'step{step_num}')
        if os.path.exists(step_dir) and os.listdir(step_dir):
            csv_files = [f for f in os.listdir(step_dir) if f.endswith('.csv')]
            if csv_files:
                return step_dir
    
    # If no output found, return Original directory
    return 'Original/'


def determine_input_directory(step_name: str, config: Dict, 
                              previous_step_output: Optional[str] = None) -> str:
    """
    Determine the correct input directory for a step.
    
    Args:
        step_name: Name of the current step (e.g., 'step1_noise_removal')
        config: Full configuration dictionary
        previous_step_output: Output directory of the previous enabled step
    
    Returns:
        Input directory path for this step
    """
    # If previous step provided output, use it
    if previous_step_output:
        return previous_step_output
    
    # Otherwise, use configured input or find latest
    step_config = config.get(step_name, {})
    configured_input = step_config.get('input_directory')
    
    # If input is configured and exists, use it
    if configured_input and os.path.exists(configured_input):
        return configured_input
    
    # Otherwise, find latest output
    return get_latest_output_directory(config['global_settings']['base_output_directory'])


def run_workflow(config: Dict) -> Dict[str, any]:
    """
    Run the complete workflow with flexible step execution.
    
    Args:
        config: Complete configuration dictionary
    
    Returns:
        Dictionary with results from all executed steps
    """
    print("="*60)
    print("FLEXIBLE WORKFLOW EXECUTION")
    print("="*60)
    
    results = {
        'executed_steps': [],
        'skipped_steps': [],
        'step_results': {}
    }
    
    # Define workflow steps
    steps = [
        ('step0_preprocessing', 'run_preprocessing', 'Step 0: Preprocessing'),
        ('step1_noise_removal', 'run_noise_removal', 'Step 1: Noise Removal'),
        ('step2_interpolation', 'run_interpolation', 'Step 2: Interpolation'),
        ('step3_smoothing', 'run_smoothing', 'Step 3: Smoothing'),
        ('step4_q_fitting', 'run_q_fitting', 'Step 4: DCA Fitting')
    ]
    
    previous_output = None
    
    for step_name, func_name, display_name in steps:
        step_config = config.get(step_name, {})
        
        # Check if step is enabled
        if not step_config.get('enabled', True):
            print(f"\n[SKIP] {display_name} - disabled in config")
            results['skipped_steps'].append(step_name)
            continue
        
        # Determine input directory
        input_dir = determine_input_directory(step_name, config, previous_output)
        
        # Update step config with determined input
        step_config['input_directory'] = input_dir
        
        # Add global settings to step config
        global_settings = config.get('global_settings', {})
        step_config['figure_format'] = global_settings.get('figure_format', 'png')
        step_config['figure_dpi'] = global_settings.get('figure_dpi', 300)
        
        print(f"\n[RUN] {display_name}")
        print(f"  Input: {input_dir}")
        print(f"  Output: {step_config.get('output_directory', 'N/A')}")
        
        try:
            # Import and run the step function
            if step_name == 'step0_preprocessing':
                from step0_preprocessing import run_preprocessing
                step_result = run_preprocessing(step_config)
            elif step_name == 'step1_noise_removal':
                from step1_noise_removal import run_noise_removal
                step_result = run_noise_removal(step_config)
            elif step_name == 'step2_interpolation':
                from step2_interpolation import run_interpolation
                step_result = run_interpolation(step_config)
            elif step_name == 'step3_smoothing':
                from step3_smoothing import run_smoothing
                step_result = run_smoothing(step_config)
            elif step_name == 'step4_q_fitting':
                from step4_q_fitting import run_q_fitting
                step_result = run_q_fitting(step_config)
            else:
                print(f"  [ERROR] Unknown step: {step_name}")
                continue
            
            # Store results
            results['executed_steps'].append(step_name)
            results['step_results'][step_name] = step_result
            
            # Update previous output for next step
            if step_config.get('save_intermediate', True):
                previous_output = step_config.get('output_directory')
            
            print(f"  [OK] {display_name} completed")
            
        except Exception as e:
            print(f"  [ERROR] Error in {display_name}: {str(e)}")
            results['step_results'][step_name] = {'error': str(e)}
    
    print("\n" + "="*60)
    print("WORKFLOW SUMMARY")
    print("="*60)
    print(f"Executed: {len(results['executed_steps'])} steps")
    print(f"Skipped: {len(results['skipped_steps'])} steps")
    
    if results['executed_steps']:
        print("\nExecuted steps:")
        for step in results['executed_steps']:
            print(f"  [OK] {step}")
    
    if results['skipped_steps']:
        print("\nSkipped steps:")
        for step in results['skipped_steps']:
            print(f"  [SKIP] {step}")
    
    print("="*60)
    
    return results


def run_from_config_file(config_path: str = 'config/config.json') -> Dict[str, any]:
    """
    Load config and run workflow.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Workflow execution results
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return run_workflow(config)


if __name__ == "__main__":
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config/config.json'
    
    print(f"Loading configuration from: {config_file}")
    results = run_from_config_file(config_file)
    
    # Exit with error code if any step failed
    errors = sum(1 for r in results['step_results'].values() if 'error' in r)
    sys.exit(1 if errors > 0 else 0)

