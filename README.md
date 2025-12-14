# Wolfcamp Production Analysis (DCA Application)

This README consolidates the previously separate documentation files into one place.

## Table of contents

- [Streamlit DCA Application - User Guide](#streamlit-dca-application---user-guide)
- [Sharing Guide - DCA Application](#sharing-guide---dca-application)
- [Wolfcamp Production Analysis - Project Documentation](#wolfcamp-production-analysis---project-documentation)

---

## Streamlit DCA Application - User Guide

*(Content merged from `README_STREAMLIT.md`)*

# Streamlit DCA Application - User Guide

## Overview

The Streamlit DCA (Decline Curve Analysis) Application provides an interactive web-based interface for analyzing oil and gas well production data. The application wraps the existing DCA workflow with a user-friendly interface, allowing you to process data step-by-step with full control over parameters and real-time visualization.

## Quick Start

### Launching the Application

1. **Double-click `launch_app.bat`** in the project root directory
   - The script will automatically:
     - Check Python installation
     - Create/use a per-user virtual environment
     - Install required dependencies
     - Launch the Streamlit app in your browser

2. **Manual Launch** (if .bat file doesn't work):
   ```bash
   # Windows (PowerShell or CMD): use the same short per-user venv path as launch_app.bat
   # Create venv once:
   python -m venv "%LOCALAPPDATA%\DCA_App\venv"

   # Install dependencies:
   "%LOCALAPPDATA%\DCA_App\venv\Scripts\python.exe" -m pip install -r requirements.txt

   # Launch app:
   "%LOCALAPPDATA%\DCA_App\venv\Scripts\python.exe" -m streamlit run streamlit_app/app.py
   ```

### First Steps

1. **Upload Data**: Use the file uploader in the sidebar to upload CSV files
2. **Select Well**: Choose a well from the dropdown
3. **Navigate Steps**: Use the pages menu to go through each workflow step
4. **Process Data**: Adjust parameters and run each step

## Application Structure

### Main Page (`app.py`)

The main entry point provides:
- **File Upload**: Upload one or more CSV files
- **Well Selection**: Dropdown to select which well to analyze
- **Navigation**: Links to all workflow steps
- **Data Management**: Clear all data button

### Workflow Steps

The application consists of 5 sequential steps:

#### Step 0: 📊 Raw Data (Preprocessing)
- **Purpose**: Clean raw production data
- **Features**:
  - Handle NaN values and zeros
  - Generate cumulative production
  - Zero removal option
- **Output**: Cleaned DataFrame with Gp (cumulative production)

#### Step 1: 🔍 Noise Removal
- **Purpose**: Remove outliers from production data
- **Features**:
  - Window-based outlier detection
  - Multiple ML algorithms (KNN, LOF, ABOD, COF, Cluster, IForest)
  - Algorithm comparison
  - Adjustable hyperparameters
- **Output**: Cleaned data for each selected algorithm

#### Step 2: 📈 Interpolation
- **Purpose**: Fill gaps in time series
- **Features**:
  - Multiple interpolation methods (linear, cubic, quadratic, nearest)
  - Fitting period detection
  - Gap visualization
- **Output**: Complete time series with detected fitting periods

#### Step 3: ✨ Smoothing
- **Purpose**: Apply smoothing filters
- **Features**:
  - Multiple filters (Gaussian, Savitzky-Golay, Spline, LOWESS)
  - Filter comparison
  - Adjustable parameters
- **Output**: Smoothed data for each selected filter

#### Step 4: 📉 DCA Fitting
- **Purpose**: Fit decline curve models and forecast
- **Features**:
  - 7 DCA models (Arps, Logistic, Stretched Exponential, Power Law, Duong, Wang, VDMA)
  - Fitting period visualization
  - Production forecasting
  - EUR calculation (P10/P50/P90)
  - Model comparison
- **Output**: Fitted models, forecasts, and EUR statistics

## Data Format

### Required CSV Format

Your CSV files must have at least two columns:

| Column | Description | Type |
|--------|-------------|------|
| `t` | Time in months | Numeric |
| `q_actual` | Production rate | Numeric |

**Example:**
```csv
t,q_actual
1,3246.0
2,2807.0
3,2912.0
```

### File Naming

- Well IDs are automatically extracted from filenames
- Supports API/UWI format: `XX-XXX-XXXXX`
- Example: `interval_475_rates_oil_42-495-33759.csv` → Well ID: `42-495-33759`

## Features

### Interactive Visualizations

All plots use Plotly for interactivity:
- **Zoom**: Click and drag to zoom
- **Pan**: Click and drag to pan
- **Hover**: Hover over points to see exact values
- **Toggle Traces**: Click legend items to show/hide data series
- **Reset**: Double-click to reset zoom

### Parameter Controls

Each step provides:
- **Sidebar Controls**: All parameters in easy-to-use widgets
- **Real-time Updates**: Changes apply when you run the step
- **Sensible Defaults**: Pre-configured with recommended values
- **Tooltips**: Help text for each parameter

### Data Export

Export processed data at any step:
- **CSV Format**: Download processed data as CSV
- **Summary Tables**: Export model summaries and statistics
- **Multiple Formats**: Export different algorithm/filter results

### Session Management

- **Session State**: All data persists during your session
- **Well Selection**: Switch between wells without losing progress
- **Step Results**: Results from each step are stored and accessible
- **Clear Data**: Reset all data when needed

## Usage Guide

### Step-by-Step Workflow

1. **Upload Data**
   - Click "Browse files" in sidebar
   - Select one or more CSV files
   - Wait for files to load

2. **Select Well**
   - Choose a well from dropdown
   - View well information (points, type, range)

3. **Step 0: Preprocessing**
   - Navigate to "📊 Raw Data" page
   - Toggle "Drop Zero Values" if needed
   - Click "Run Preprocessing"
   - Review visualization and statistics
   - Export if needed

4. **Step 1: Noise Removal**
   - Navigate to "🔍 Noise Removal" page
   - Select algorithms to use
   - Adjust hyperparameters
   - Click "Run Noise Removal"
   - Compare algorithm results
   - Select best algorithm and export

5. **Step 2: Interpolation**
   - Navigate to "📈 Interpolation" page
   - Select interpolation method
   - Set minimum fitting period
   - Click "Run Interpolation"
   - Review detected fitting periods

6. **Step 3: Smoothing**
   - Navigate to "✨ Smoothing" page
   - Select smoothing filters
   - Adjust filter parameters
   - Click "Run Smoothing"
   - Compare filter results
   - Select best filter and export

7. **Step 4: DCA Fitting**
   - Navigate to "📉 Fitting" page
   - Review detected fitting period
   - Select DCA models to fit
   - Configure fitting parameters
   - Set forecast parameters
   - Click "Run DCA Fitting"
   - Review model results and EUR statistics
   - Export summary table

### Tips and Best Practices

1. **Start with Defaults**: Use default parameters first, then adjust as needed
2. **Check Visualizations**: Always review plots to ensure processing looks correct
3. **Compare Algorithms**: In Step 1, try multiple algorithms to find the best fit
4. **Validate Fitting Period**: In Step 4, verify the detected fitting period makes sense
5. **Export Results**: Save important results using the download buttons
6. **One Well at a Time**: Process one well completely before moving to the next

## Troubleshooting

### Common Issues

**Problem**: "No wells loaded"
- **Solution**: Upload CSV files using the sidebar file uploader

**Problem**: "Please complete Step X first"
- **Solution**: Complete steps in order (0 → 1 → 2 → 3 → 4)

**Problem**: "PyCaret not available" (Step 1)
- **Solution**: Install PyCaret: `pip install pycaret`
- **Note**: PyCaret requires Python 3.8-3.11 (not 3.12+)

**Problem**: "Insufficient data for fitting" (Step 4)
- **Solution**: Reduce `minimum_production_history` or use more data

**Problem**: "No models were successfully fitted" (Step 4)
- **Solution**: 
  - Lower `fitting_accuracy_threshold`
  - Lower `prediction_accuracy_threshold`
  - Try different models
  - Check if fitting period is appropriate

**Problem**: App crashes or freezes
- **Solution**: 
  - Check Python version (3.8-3.11 recommended)
  - Ensure all dependencies are installed
  - Try clearing browser cache
  - Restart the application

### Performance Tips

- **Large Datasets**: Process wells one at a time
- **Multiple Algorithms**: Step 1 can be slow with many algorithms - start with fewer
- **Complex Models**: Step 4 fitting can take time - be patient
- **Browser**: Use Chrome or Firefox for best performance

## Technical Details

### Dependencies

All dependencies are listed in `requirements.txt`:
- Core: pandas, numpy, scipy
- ML: pycaret (for anomaly detection)
- Optimization: lmfit, scikit-optimize, pyswarm
- Visualization: plotly, matplotlib
- Web: streamlit

### Architecture

- **Modular Design**: Each step is a separate page
- **Session State**: Data persists across page navigation
- **Wrapper Functions**: Existing workflow functions are wrapped for Streamlit
- **No Code Changes**: Original `src/` modules remain unchanged

### File Structure

```
streamlit_app/
├── app.py                 # Main entry point
├── pages/                 # Step pages
│   ├── 1_📊_Raw_Data.py
│   ├── 2_🔍_Noise_Removal.py
│   ├── 3_📈_Interpolation.py
│   ├── 4_✨_Smoothing.py
│   └── 5_📉_Fitting.py
├── components/            # Reusable components
│   ├── data_loader.py
│   └── well_selector.py
├── utils/                 # Utilities
│   ├── session_state.py
│   └── streamlit_helpers.py
└── config/               # Configuration
    └── default_config.py
```

## Support

### Getting Help

1. **Check Instructions**: Each page has an "Instructions" expander
2. **Review Parameters**: Hover over parameter controls for tooltips
3. **Check Console**: Look for error messages in the Streamlit console
4. **Validate Data**: Ensure CSV files have correct format

### Known Limitations

- PyCaret requires Python 3.8-3.11 (not 3.12+)
- Large datasets may be slow to process
- Some optimization methods require additional packages
- Browser compatibility: Best with Chrome/Firefox

## Version History

- **v1.0.0** (Current): Initial release
  - All 5 workflow steps implemented
  - Interactive Plotly visualizations
  - Full parameter control
  - Data export functionality

## License

Same as the main DCA project.

---

For issues or questions, refer to the sections below or contact the development team.

---

## Sharing Guide - DCA Application

*(Content merged from `SHARING_GUIDE.md`)*

# Sharing Guide - DCA Application

This guide explains how to share the application folder with others or move it to a different location.

## ✅ What Works Automatically

The application is designed to be **fully portable**:

- ✅ **All paths are relative** - The code uses relative paths, so it works regardless of where the folder is located
- ✅ **Automatic setup** - The `launch_app.bat` script automatically:
  - Checks Python installation
  - Creates/uses a per-user virtual environment (short path to avoid Windows long-path issues)
  - Validates requirements file (basic checks)
  - Installs dependencies
  - Launches the application

## 📦 What to Share

When sharing the application folder, include:

### Required Files:
- ✅ `launch_app.bat` - Main launcher script
- ✅ `requirements.txt` - Python dependencies
- ✅ `src/dca_tools.py` - Single validation/integrity tool (requirements, checksums, python checks)
- ✅ `config/config.json` - Configuration file
- ✅ `streamlit_app/` - Application code
- ✅ `src/` - Source code modules
- ✅ `Data/` or `Original/` - Your data files (if sharing data)

### Optional Files:
- `README_STREAMLIT.md` - Documentation
- `PROJECT_DOCUMENTATION.md` - Full documentation
- `src/run_workflow.py` - Command-line workflow runner
- `file_checksums.txt` - Checksum manifest (generated by validation script)

### ❌ Do NOT Share:
- ❌ `venv/` folder - Virtual environment (contains machine-specific paths)
- ❌ `output/` folder - Generated outputs (can be large)
- ❌ `__pycache__/` folders - Python cache files
- ❌ `logs.log` - Log files
- ❌ `.git/` folder - Git repository (if using version control)

## 🚀 How to Share

### Option 1: Zip the Folder (Recommended)

1. **Before zipping**, make sure to:
   - **IMPORTANT: Validate files before sharing** (prevents corruption issues):
     ```bash
     python src/dca_tools.py write-checksums
     ```
     This will:
     - Generate `file_checksums.txt` for integrity verification
   - Delete the `venv/` folder (if it exists)
   - Optionally delete `output/` folder (if you don't want to share outputs)
   - Optionally delete `logs.log` (if it exists)

2. **Zip the folder** with all required files
   - Include `file_checksums.txt` if it was generated
   - Use a reliable compression method (ZIP format recommended)

3. **Share the zip file**
   - Use a reliable transfer method (avoid OneDrive sync if possible - use direct download)
   - For large files, consider using cloud storage with direct download links

4. **Recipient should**:
   - Extract the zip to any location
   - **Optional: Verify file integrity** (if `file_checksums.txt` exists):
     ```bash
     python src/dca_tools.py verify-checksums
     ```
   - Double-click `launch_app.bat`
   - The script will automatically set everything up

### Option 2: Use Git (For Developers)

If using Git, the `.gitignore` file will automatically exclude:
- `venv/` folder
- `__pycache__/` folders
- Output files
- Log files

Recipients can clone and run `launch_app.bat`.

## 🔧 First-Time Setup on New Machine

When someone receives the application folder:

1. **Extract/Place** the folder anywhere (e.g., `C:\Users\Friend\DCA-Application\`)

2. **Double-click** `launch_app.bat`

3. **The script will automatically**:
   - Check Python installation (needs Python 3.8-3.11)
   - Create a new virtual environment
   - Validate requirements file (catches corruption/typos)
   - Install all dependencies
   - Launch the application

4. **If Python is not installed**:
   - Download from https://www.python.org/downloads/
   - Install Python 3.8, 3.9, 3.10, or 3.11
   - Make sure to check "Add Python to PATH" during installation

## 🛡️ Preventing File Corruption

File corruption can occur during transfer, especially with cloud sync services (OneDrive, Dropbox). This can cause errors like:
- `ERROR: Could not find a version that satisfies the requirement puccinialin`
- Package names getting corrupted (e.g., "plotly" → "puccinialin")

### Prevention Steps (Before Sharing):

1. **Run pre-sharing validation**:
   ```bash
   python src/dca_tools.py write-checksums
   ```
   This validates all files and generates checksums.

2. **Use reliable transfer methods**:
   - ✅ ZIP file download (recommended)
   - ✅ Git repository
   - ⚠️ OneDrive/Dropbox sync (can cause corruption - use direct download instead)
   - ⚠️ USB drives (verify after copying)

3. **Include checksums file**:
   - The `file_checksums.txt` file allows recipients to verify file integrity
   - Always include it when sharing

### Verification Steps (After Receiving):

1. **Verify file integrity** (if `file_checksums.txt` exists):
   ```bash
   python src/dca_tools.py verify-checksums
   ```
   This will detect any corruption before installation.

2. **Manual validation**:
   ```bash
   python src/dca_tools.py validate-requirements requirements.txt
   ```
   This checks the requirements file for corruption.

## ⚠️ Common Issues

### Issue: "Python is not installed or not in PATH"
**Solution**: Install Python 3.8-3.11 and ensure it's added to PATH

### Issue: "Requirements file validation failed" or "File corruption detected"
**Solution**: The requirements file may be corrupted. This is a common issue when files are transferred via cloud sync.

**Immediate steps**:
1. Run: `python src/dca_tools.py verify-checksums` to identify corrupted files (if `file_checksums.txt` exists)
2. Ask the sender to:
   - Run `python src/dca_tools.py write-checksums` before sharing
   - Re-share the project folder using a ZIP file (not sync)
   - Provide a fresh copy of `requirements.txt`
3. If you have the original, compare files to find differences
4. Common corruption: "plotly" becomes "puccinialin" - manually fix if needed

**Prevention**: Always use ZIP files for sharing, not cloud sync.

### Issue: "Failed to install dependencies" with package name errors
**Solution**: 
- Check if the error mentions a suspicious package name (like "puccinialin")
- This indicates file corruption - see above solution
- Check internet connection
- Verify Python version is 3.8-3.11
- Try running: `pip install --upgrade pip`
- Check error message for specific package that failed

### Issue: Virtual environment activation fails
**Solution**: The script will automatically recreate the virtual environment if activation fails

## 📝 Notes

- **Python Version**: The application requires Python 3.8, 3.9, 3.10, or 3.11 (not 3.12+)
- **Platform**: Currently optimized for Windows. For Mac/Linux, use the manual commands in `README_STREAMLIT.md`
- **Data Files**: Make sure to include your CSV data files in the `Data/` or `Original/` folder
- **Configuration**: The `config/config.json` file can be modified to change settings

## ✅ Verification Checklist

### Before Sharing:
- [ ] Run `python src/dca_tools.py write-checksums` to generate `file_checksums.txt`
- [ ] `venv/` folder is deleted (or not included)
- [ ] `requirements.txt` exists and is valid
- [ ] `launch_app.bat` exists
- [ ] `src/dca_tools.py` exists
- [ ] All source code folders (`src/`, `streamlit_app/`) are included
- [ ] `config/config.json` exists
- [ ] `file_checksums.txt` is generated and included (if using validation)

### After Receiving:
- [ ] Python 3.8-3.11 is installed
- [ ] All files are extracted correctly
- [ ] Run `python src/dca_tools.py verify-checksums` to check for corruption (if checksums file exists)
- [ ] Run `python src/dca_tools.py validate-requirements requirements.txt` to validate requirements file
- [ ] No files appear corrupted
- [ ] Internet connection is available (for dependency installation)

---

**The application is designed to work anywhere - just extract and run `launch_app.bat`!**

---

## Wolfcamp Production Analysis - Project Documentation

*(Content merged from `PROJECT_DOCUMENTATION.md`)*

# Wolfcamp Production Analysis - Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Project Structure](#project-structure-1)
5. [Quick Start Guide](#quick-start-guide)
6. [Configuration File Guide](#configuration-file-guide)
7. [Workflow Steps](#workflow-steps)
8. [Advanced Configuration](#advanced-configuration)
9. [Troubleshooting](#troubleshooting-1)
10. [Output Files](#output-files)

---

## Project Overview

The **Wolfcamp Production Analysis** project is a comprehensive Python-based workflow for analyzing oil and gas well production data. The system processes raw production data through multiple stages—from preprocessing and noise removal to decline curve analysis (DCA) and forecasting—to estimate ultimate recovery (EUR) and generate production forecasts.

### Key Features

- **Automated Multi-Step Workflow**: Five sequential processing steps from raw data to forecasts
- **Flexible Configuration**: JSON-based configuration system with extensive customization options
- **Multiple DCA Models**: Support for 7 different decline curve models
- **Advanced Noise Removal**: Ensemble machine learning algorithms for outlier detection
- **Multiple Smoothing Filters**: Gaussian, Savitzky-Golay, Spline, and LOWESS filters
- **Comprehensive Forecasting**: P10/P50/P90 production scenarios and EUR calculations
- **Batch Processing**: Process multiple wells simultaneously
- **Visualization Support**: Generate plots for specific wells at each processing step

### Typical Use Cases

- Unconventional reservoir production forecasting
- Decline curve analysis for multiple wells
- EUR (Estimated Ultimate Recovery) estimation
- Production data quality control and cleaning
- Comparative model performance analysis

---

## System Requirements

### Software Requirements

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | 3.8 - 3.11 | Main programming language |
| pandas | ≥ 1.5.0 | Data manipulation |
| numpy | ≥ 1.23.0 | Numerical computing |
| scipy | ≥ 1.10.0 | Scientific computing and optimization |
| pycaret | ≥ 3.0.0 | Machine learning anomaly detection |
| lmfit | ≥ 1.2.0 | Non-linear optimization |
| statsmodels | ≥ 0.14.0 | Statistical models |
| matplotlib | ≥ 3.6.0 | Visualization |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| scikit-optimize | Bayesian optimization (fitting_method = 3) |
| pyswarm | Particle Swarm Optimization (fitting_method = 4) |

### Hardware Recommendations

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: Minimum 8 GB, 16 GB recommended for large datasets
- **Storage**: 1 GB free space for outputs

---

## Installation

### Step 1: Clone or Download the Project

Ensure you have the project directory with all necessary files.

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

Check that all key packages are installed:

```bash
python -c "import pandas, numpy, scipy, pycaret, lmfit; print('All packages installed successfully!')"
```

---

## Project Structure

```
final_code/
├── config/config.json           # Main configuration file
├── src/run_workflow.py          # Workflow execution script
├── requirements.txt             # Python dependencies
├── PROJECT_DOCUMENTATION.md     # This documentation file
│
├── Original/                    # Input data directory
│   ├── example_well.csv
│   └── interval_*.csv          # Raw production data files
│
├── output/                      # Output directory (auto-generated)
│   ├── step0/                  # Preprocessed data
│   ├── step1/                  # Noise-removed data
│   ├── step2/                  # Interpolated data
│   ├── step3/                  # Smoothed data
│   └── step4/                  # DCA fitting results and forecasts
│
└── src/                         # Source code directory
    ├── workflow_runner.py       # Workflow orchestration
    ├── step0_preprocessing.py   # Data preprocessing module
    ├── step1_noise_removal.py   # Outlier detection module
    ├── step2_interpolation.py   # Data interpolation module
    ├── step3_smoothing.py       # Data smoothing module
    ├── step4_q_fitting.py       # DCA fitting module
    └── utils/                   # Utility functions
```

---

## Quick Start Guide

### Basic Workflow Execution

**1. Prepare Your Data**

Place your CSV files in the `Original/` directory. Each file should contain:
- Column 1: `t` (time in months)
- Column 2: `q_actual` (production rate)

**2. Configure the Workflow**

Review and modify `config/config.json` as needed (see Configuration File Guide below).

**3. Run the Workflow**

```bash
python src/run_workflow.py
```

**4. Check Results**

Results are saved in the `output/` directory, organized by step:
- `output/step0/` - Cleaned data
- `output/step1/` - Outlier-free data
- `output/step2/` - Interpolated data
- `output/step3/` - Smoothed data
- `output/step4/` - DCA models, forecasts, and EUR summary

### Using a Custom Configuration File

```bash
python src/run_workflow.py my_custom_config.json
```

### Processing Specific Wells Only

Edit `config/config.json` and modify the `wells_to_process` parameter in any step:

```json
"wells_to_process": ["42-495-33759", "42-301-33452"]
```

Or process all wells:

```json
"wells_to_process": ["all"]
```

---

## Configuration File Guide

The `config/config.json` file controls all aspects of the workflow. It is organized into sections for global settings and individual steps.

### Global Settings

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `project_name` | string | Any text | Name of your project |
| `base_output_directory` | string | Path | Base directory for all outputs |
| `save_mode` | string | `per_step`, `final_only` | Save outputs at each step or only final results |
| `figure_format` | string | `png`, `pdf`, `svg` | Format for saved figures |
| `figure_dpi` | integer | 100-600 | Resolution of saved figures |
| `verbose` | boolean | `true`, `false` | Enable detailed logging |

### Common Step Parameters

These parameters are available in all step configurations:

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | boolean | Enable/disable this step in the workflow |
| `input_directory` | string | Directory containing input files for this step |
| `output_directory` | string | Directory to save output files |
| `save_intermediate` | boolean | Save CSV outputs after this step |
| `wells_to_process` | array | List of well IDs or `["all"]` to process all |
| `visualize_wells` | array | List of well IDs to generate plots for, or `[]` for none |

---

## Workflow Steps

*(For full step-by-step explanations and configuration tables, see the previous sections above.)*

---

## Advanced Configuration

*(For advanced configuration examples, presets, and performance guidance, see the previous sections above.)*

---

## Troubleshooting

*(For troubleshooting tables and validation checks, see the earlier troubleshooting sections above.)*

---

## Output Files

*(For output directory structure and file details, see the earlier “Output Files” section above.)*


