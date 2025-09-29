# Environment Setup

To create the environment:
```bash
conda create -n phd-pdm python=3.10
conda activate phd-pdm
pip install -r requirements.txt
```
# Data Structure and Setup Guide

⚠️ **Data files are not included in this repository due to size constraints.**

## Overview

This project uses 5 different industrial datasets for comprehensive predictive maintenance evaluation:

1. **C-MAPSS (NASA Turbofan Engine)** - Aircraft engine degradation simulation
2. **PRONOSTIA Bearing Dataset** - Ball bearing accelerated life testing  
3. **IMS Bearing Dataset** - Industrial bearing run-to-failure data
4. **NASA Milling Dataset** - Tool wear monitoring during milling operations
5. **NASA Battery Dataset** - Battery degradation and capacity fade

## Required Data Structure

After downloading and extracting all datasets, your `data/` folder should look like this:

```
data/
├── processed_data/                    # Generated during preprocessing
└── raw_data/
    ├── C_MAPSS/                      # NASA Turbofan Engine Dataset
    │   ├── train_FD001.txt           # Training data - Dataset FD001 (~3.5MB)
    │   ├── test_FD001.txt            # Test data - Dataset FD001 (~2.2MB)
    │   ├── RUL_FD001.txt             # True RUL values - Dataset FD001 (429 bytes)
    │   ├── train_FD002.txt           # Training data - Dataset FD002 (~9.1MB)
    │   ├── test_FD002.txt            # Test data - Dataset FD002 (~5.7MB)
    │   ├── RUL_FD002.txt             # True RUL values - Dataset FD002 (1,110 bytes)
    │   ├── train_FD003.txt           # Training data - Dataset FD003 (~4.2MB)
    │   ├── test_FD003.txt            # Test data - Dataset FD003 (~2.8MB)
    │   ├── RUL_FD003.txt             # True RUL values - Dataset FD003 (428 bytes)
    │   ├── train_FD004.txt           # Training data - Dataset FD004 (~10.4MB)
    │   ├── test_FD004.txt            # Test data - Dataset FD004 (~7.0MB)
    │   ├── RUL_FD004.txt             # True RUL values - Dataset FD004 (1,084 bytes)
    │   ├── readme.txt                # Dataset documentation
    │   └── Damage Propagation Modeling.pdf  # Technical documentation (~434KB)
    ├── IMS/                          # IMS Bearing Dataset
    │   ├── 1st_test/                 # First bearing test run
    │   │   ├── 2003.10.22.12.06.24   # Vibration data files
    │   │   ├── 2003.10.22.12.36.24
    │   │   └── ...
    │   ├── 2nd_test/                 # Second bearing test run
    │   │   ├── 2004.02.12.10.32.39
    │   │   └── ...
    │   └── 3rd_test/                 # Third bearing test run
    │       ├── 2004.02.12.10.32.39
    │       └── ...
    ├── Nasa_Battery/                 # NASA Battery Dataset (CSV format)
    │   ├── data/                     # Battery cycling data (CSV files)
    │   │   ├── 00001.csv             # Battery cycling data (~37KB)
    │   │   ├── 00002.csv             # Battery cycling data (~9.6KB)
    │   │   ├── 00003.csv             # Battery cycling data (~125KB)
    │   │   ├── 00004.csv             # Battery cycling data (~9.6KB)
    │   │   ├── 00005.csv             # Battery cycling data (~32KB)
    │   │   ├── 00006.csv             # Battery cycling data (~131KB)
    │   │   ├── 00007.csv             # Battery cycling data (~32KB)
    │   │   ├── 00008.csv             # Battery cycling data (~124KB)
    │   │   ├── 00009.csv             # Battery cycling data (~31KB)
    │   │   └── ...                   # Additional battery CSV files
    │   └── extra_infos/              # Additional documentation
    │       ├── README_05_06_07_18.txt    # Documentation for batteries 5,6,7,18
    │       ├── README_25_26_27_28.txt    # Documentation for batteries 25,26,27,28
    │       ├── README_29_30_31_32.txt    # Documentation for batteries 29,30,31,32
    │       ├── README_33_34_36.txt       # Documentation for batteries 33,34,36
    │       ├── README_38_39_40.txt       # Documentation for batteries 38,39,40
    │       ├── README_41_42_43_44.txt    # Documentation for batteries 41,42,43,44
    │       ├── README_45_46_47_48.txt    # Documentation for batteries 45,46,47,48
    │       ├── README_49_50_51_52.txt    # Documentation for batteries 49,50,51,52
    │       └── README_53_54_55_56.txt    # Documentation for batteries 53,54,55,56
    ├── Nasa_Milling/                 # NASA Milling Dataset
    │   └── mill.csv                  # Milling data (~19KB)
    └── Pronostia/                    # PRONOSTIA Bearing Dataset
        ├── Full_Test_Set/            # Complete test set (11 bearings)
        │   ├── Bearing1_3/
        │   │   ├── acc_00001.csv     # Vibration acceleration data
        │   │   ├── acc_00002.csv
        │   │   └── ...
        │   ├── Bearing1_4/
        │   ├── Bearing1_5/
        │   ├── Bearing1_6/
        │   ├── Bearing1_7/
        │   ├── Bearing2_3/
        │   ├── Bearing2_4/
        │   ├── Bearing2_5/
        │   ├── Bearing2_6/
        │   ├── Bearing2_7/
        │   └── Bearing3_3/
        ├── Learning_set/             # Training set (6 bearings)
        │   ├── Bearing1_1/
        │   │   ├── acc_00001.csv     # Vibration acceleration data
        │   │   ├── acc_00002.csv
        │   │   └── ...
        │   ├── Bearing1_2/
        │   ├── Bearing2_1/
        │   ├── Bearing2_2/
        │   ├── Bearing3_1/
        │   └── Bearing3_2/
        └── Test_set/                 # Test set (same as Full_Test_Set)
            ├── Bearing1_3/
            ├── Bearing1_4/
            ├── ...
            └── Bearing3_3/
```

## Dataset Download Instructions

### 1. NASA C-MAPSS Turbofan Engine Dataset
- **Source**: [NASA Prognostics Center](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- **Direct Link**: [Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)
- **Files needed**: Download `CMAPSSData.zip`
- **Extract to**: `data/raw_data/C_MAPSS/`
- **Description**: 4 different datasets (FD001-FD004) with varying operating conditions and fault modes

### 2. PRONOSTIA Bearing Dataset
- **Source**: [IEEE PHM 2012 Challenge](https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge)
- **Alternative**: [Kaggle PRONOSTIA](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)
- **Files needed**: Complete PRONOSTIA bearing dataset
- **Extract to**: `data/raw_data/Pronostia/`
- **Description**: Accelerated life testing of ball bearings with vibration and temperature monitoring

### 3. IMS Bearing Dataset
- **Source**: [IMS, University of Cincinnati](https://www.dropbox.com/s/a3q8dl2gqk4g52v/IMS.rar?dl=0)
- **Alternative**: [NASA Prognostics Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing)
- **Files needed**: `IMS.rar` or individual test folders
- **Extract to**: `data/raw_data/IMS/`
- **Description**: Run-to-failure bearing data with 4-channel vibration measurements

### 4. NASA Battery Dataset
- **Source**: [NASA Prognostics Center](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#battery)
- **Files needed**: Battery dataset (CSV format)
- **Extract to**: `data/raw_data/Nasa_Battery/`
- **Description**: Li-ion battery aging data with cycling information in CSV format

### 5. NASA Milling Dataset
- **Source**: [NASA Prognostics Center](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#milling)
- **Files needed**: Milling dataset files
- **Extract to**: `data/raw_data/Nasa_Milling/`
- **Description**: Tool wear monitoring with sensor measurements in CSV format

## Dataset Characteristics

| Dataset | Equipment Type | Sensors | Training Units | Test Units | Target Variable |
|---------|---------------|---------|----------------|------------|-----------------|
| C-MAPSS FD001 | Turbofan Engine | 21 sensors | 100 engines | 100 engines | RUL (cycles) |
| C-MAPSS FD002 | Turbofan Engine | 21 sensors | 260 engines | 259 engines | RUL (cycles) |
| C-MAPSS FD003 | Turbofan Engine | 21 sensors | 100 engines | 100 engines | RUL (cycles) |
| C-MAPSS FD004 | Turbofan Engine | 21 sensors | 248 engines | 249 engines | RUL (cycles) |
| PRONOSTIA | Ball Bearings | 2 sensors (vibration, temp) | 6 bearings | 11 bearings | RUL (hours) |
| IMS | Roller Bearings | 4 sensors (vibration) | 3 test runs | N/A | Failure detection |
| NASA Battery | Li-ion Battery | Voltage, current, temp | Multiple batteries (CSV) | N/A | Capacity/cycle data |
| NASA Milling | Milling Tool | Various sensors | Single dataset | N/A | Tool wear/performance |

## Quick Setup Verification

After downloading all datasets, run this script to verify your setup:

```python
import os
from pathlib import Path

def verify_data_structure():
    """Verify all required datasets are properly placed"""
    
    base_path = Path("data/raw_data")
    
    required_datasets = {
        "C_MAPSS": ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"],
        "Pronostia": ["Learning_set", "Test_set", "Full_Test_Set"],
        "IMS": ["1st_test", "2nd_test", "3rd_test"],
        "Nasa_Battery": ["data", "extra_infos"],
        "Nasa_Milling": ["mill.csv"]
    }
    
    all_good = True
    
    for dataset, required_items in required_datasets.items():
        dataset_path = base_path / dataset
        
        if not dataset_path.exists():
            print(f"❌ Missing dataset folder: {dataset}")
            all_good = False
            continue
            
        print(f"✅ Found dataset: {dataset}")
        
        for item in required_items:
            item_path = dataset_path / item
            if not item_path.exists():
                print(f"   ⚠️  Missing: {item}")
                all_good = False
            else:
                print(f"   ✅ Found: {item}")
    
    # Check processed_data folder exists
    processed_path = Path("data/processed_data")
    if not processed_path.exists():
        print("📁 Creating processed_data folder...")
        processed_path.mkdir(parents=True, exist_ok=True)
        print("✅ processed_data folder created")
    
    if all_good:
        print("\n🎉 All datasets properly configured!")
    else:
        print("\n⚠️  Some datasets need attention. Check the missing items above.")
    
    return all_good

# Run verification
if __name__ == "__main__":
    verify_data_structure()
```

## Data Loader Usage

### Phase 1: C-MAPSS and NASA Milling (Current Implementation)

The multi-dataset loader currently supports C-MAPSS turbofan datasets and NASA Milling tool wear data, with automatic path detection from the configured directory structure.

**Quick Start:**
```python
from src.data_loader import MultiDatasetLoader

# Initialize loader (uses paths from src/config.py)
loader = MultiDatasetLoader()

# Load individual C-MAPSS datasets
fd001_data = loader.load_fd001()
fd002_data = loader.load_fd002()

# Or load all C-MAPSS datasets at once
all_cmapss = loader.load_all_cmapss()

# Load NASA Milling dataset
milling_data = loader.load_nasa_milling()

# Access train/test splits
train_df = fd001_data['train']
test_df = fd001_data['test']
config = fd001_data['config']
```

## Storage Requirements

- **C-MAPSS**: ~50 MB (4 datasets: FD001-FD004, text files)
- **PRONOSTIA**: ~2.5 GB (CSV vibration data across all bearings)
- **IMS**: ~800 MB (time-series vibration measurements)
- **NASA Battery**: ~5-10 MB (Multiple CSV files with cycling data)
- **NASA Milling**: ~19 KB (Single CSV file)
- **Total**: ~3.4 GB raw data + ~1-2 GB processed data

## Processing Notes

- **Recommended**: Use SSD storage for faster data loading
- **Memory**: Ensure at least 8GB RAM for full dataset processing  
- **Initial processing**: May take 5-15 minutes depending on hardware
- **Processed data**: Will be saved in `data/processed_data/` for faster subsequent loading
