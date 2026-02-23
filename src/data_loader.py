"""
Multi-Dataset Data Loader for Predictive Maintenance
Supports C-MAPSS and NASA Milling datasets

Author: Fatima Khadija Benzine
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add project root
current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent

# Add both src and project root to path
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

# Import config (now that paths are set up)
from config import RAW_DATA_DIR, C_MAPSS_DIR, IMS_DIR, NASA_BATTERY_DIR, NASA_MILLING_DIR, PRONOSTIA_DIR

class DatasetConfig:
    """Configuration class for different datasets"""
    
    FD001 = {
        'name': 'FD001',
        'type': 'turbofan',
        'description': 'Single operating condition, single fault mode (HPC degradation)',
        'file_format': 'txt',
        'separator': ' ',
        'columns': ['unit', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + 
                  [f'sensor_{i}' for i in range(1, 22)],
        'target_col': 'rul',
        'unit_col': 'unit',
        'cycle_col': 'cycle',
        'operating_conditions': 1,
        'fault_modes': 1
    }
    
    FD002 = {
        'name': 'FD002', 
        'type': 'turbofan',
        'description': 'Multiple operating conditions, single fault mode (HPC degradation)',
        'file_format': 'txt',
        'separator': ' ',
        'columns': ['unit', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + 
                  [f'sensor_{i}' for i in range(1, 22)],
        'target_col': 'rul',
        'unit_col': 'unit', 
        'cycle_col': 'cycle',
        'operating_conditions': 6,
        'fault_modes': 1
    }

    FD003 = {
        'name': 'FD003',
        'type': 'turbofan',
        'description': 'Single operating condition, multiple fault modes (HPC & Fan degradation)',
        'file_format': 'txt',
        'separator': ' ',
        'columns': ['unit', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] +
                  [f'sensor_{i}' for i in range(1, 22)],
        'target_col': 'rul',
        'unit_col': 'unit',
        'cycle_col': 'cycle',
        'operating_conditions': 1,
        'fault_modes': 2
    }
    FD004 = {
        'name': 'FD004',
        'type': 'turbofan',
        'description': 'Multiple operating conditions, multiple fault modes (HPC & Fan degradation)', 
        'file_format': 'txt',
        'separator': ' ',
        'columns': ['unit', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + 
                  [f'sensor_{i}' for i in range(1, 22)],
        'target_col': 'rul',
        'unit_col': 'unit',
        'cycle_col': 'cycle',
        'operating_conditions': 6, 
        'fault_modes': 2
    }
    
    # NASA Milling Dataset configuration
    NASA_MILLING = {
        'name': 'NASA_Milling',
        'type': 'milling',
        'description': 'Tool wear prediction on milling machine with varying speeds, feeds, depth of cut',
        'file_format': 'csv',
        'separator': ',',
        'target_col': 'rul',  # VB - flank wear
        'unit_col': 'case',
        'cycle_col': 'run',
        'source': 'UC Berkeley via NASA Prognostics Center',
        'wear_measurement': 'VB',  # Flank wear (mm)
        'sampling_rate': 100,  # 100 ms sampling
        'n_experiments': 16
    }

class MultiDatasetLoader:
    """
    Focused data loader for C-MAPSS and NASA Milling datasets
    """
    
    def __init__(self, data_root: Path = RAW_DATA_DIR):
        self.data_root = Path(data_root)
        self.datasets = {}
        self.dataset_info = {}
        
    def load_cmapss_dataset(self, 
                           dataset_name: str,
                           train_path: str = None,
                           test_path: str = None, 
                           rul_path: str = None) -> Dict:
        """
        Load any C-MAPSS dataset (FD001, FD002, FD003, FD004)
        
        Args:
            dataset_name: Name of dataset ('FD001', 'FD002', 'FD003', 'FD004')
            train_path: Optional custom path for training file
            test_path: Optional custom path for test file  
            rul_path: Optional custom path for RUL file
            
        Returns:
            Dict with 'train', 'test', 'rul' dataframes
        """
        valid_datasets = ['FD001', 'FD002', 'FD003', 'FD004']
        if dataset_name not in valid_datasets:
            raise ValueError(f"Dataset must be one of {valid_datasets}")
        
        # Get configuration for the specific dataset
        config = getattr(DatasetConfig, dataset_name)
        
        # Set default file paths if not provided
        if train_path is None:
            train_path = f"train_{dataset_name}.txt"
        if test_path is None:
            test_path = f"test_{dataset_name}.txt"  
        if rul_path is None:
            rul_path = f"RUL_{dataset_name}.txt"
        
        # Construct full paths
        train_file = self.data_root / "C_MAPSS" / train_path
        test_file = self.data_root / "C_MAPSS" / test_path
        rul_file = self.data_root / "C_MAPSS" / rul_path
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        if not rul_file.exists():
            raise FileNotFoundError(f"RUL file not found: {rul_file}")
        
        print(f"Loading {dataset_name} dataset...")
        print(f"  Files: train={train_file.exists()}, test={test_file.exists()}, rul={rul_file.exists()}")
        
        # Load training data
        try:
            train_df = pd.read_csv(train_file, sep=r'\s+', 
                                  header=None, names=config['columns'], engine='python')
            print(f"  - Training data shape: {train_df.shape}")
            
            # Check for empty or malformed data
            if train_df.empty:
                raise ValueError("Training data is empty")
            if train_df.isnull().all().any():
                print(f"    Warning: Some columns are entirely null")
                
        except Exception as e:
            raise Exception(f"Failed to load training file {train_file}: {e}")
        
        # Calculate RUL for training data
        try:
            train_df['rul'] = train_df.groupby('unit')['cycle'].transform(lambda x: x.max() - x)
            print(f"  - Training units: {train_df['unit'].nunique()}")
            print(f"  - Training RUL range: [{train_df['rul'].min()}, {train_df['rul'].max()}]")
        except Exception as e:
            raise Exception(f"Failed to calculate training RUL: {e}")
        
        # Load test data
        try:
            test_df = pd.read_csv(test_file, sep=r'\s+', 
                                 header=None, names=config['columns'], engine='python')
            print(f"  - Test data shape: {test_df.shape}")
            
            if test_df.empty:
                raise ValueError("Test data is empty")
                
        except Exception as e:
            raise Exception(f"Failed to load test file {test_file}: {e}")
        
        # Load RUL for test data
        try:
            rul_values = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['rul'], engine='python')
            print(f"  - RUL values shape: {rul_values.shape}")
            
            if rul_values.empty:
                raise ValueError("RUL file is empty")
                
        except Exception as e:
            raise Exception(f"Failed to load RUL file {rul_file}: {e}")
        
        # Add RUL to test data
        test_units = sorted(test_df['unit'].unique())
        test_df['rul'] = 0
        
        # Debug information
        print(f"  - Test units found: {len(test_units)} (units: {test_units[:5]}{'...' if len(test_units) > 5 else ''})")
        print(f"  - RUL values provided: {len(rul_values)}")
        
        if len(test_units) != len(rul_values):
            print(f"  - Warning: Unit count mismatch! {len(test_units)} units vs {len(rul_values)} RUL values")
            print(f"  - Test units range: {min(test_units)} to {max(test_units)}")
            print(f"  - Sample RUL values: {rul_values.head().values.flatten()}")
            raise ValueError(f"Mismatch: {len(test_units)} test units but {len(rul_values)} RUL values")

        # Assign RUL values to test units
        for idx, unit in enumerate(test_units):
            if idx >= len(rul_values):
                print(f"  - Error: Trying to access RUL index {idx} but only have {len(rul_values)} values")
                break
                
            mask = test_df['unit'] == unit
            max_cycle = test_df[mask]['cycle'].max()
            rul_value = rul_values.iloc[idx, 0]
            test_df.loc[mask, 'rul'] = rul_value + (max_cycle - test_df.loc[mask, 'cycle'])
            
            if idx < 3:  # Show first few assignments for debugging
                print(f"    Unit {unit}: max_cycle={max_cycle}, base_RUL={rul_value}")
        
        # Calculate operating condition clusters for multi-condition datasets
        operating_conditions = None
        if config['operating_conditions'] > 1:
            operating_conditions = self._identify_operating_conditions(train_df, test_df, config)
        
        dataset = {
            'train': train_df,
            'test': test_df,
            'config': config,
            'operating_conditions': operating_conditions,
            'info': {
                'n_units_train': train_df['unit'].nunique(),
                'n_units_test': test_df['unit'].nunique(),
                'n_features': len(config['columns']) - 2,  # excluding unit and cycle
                'n_sensors': 21,  # C-MAPSS has 21 sensors
                'n_settings': 3,  # C-MAPSS has 3 operational settings
                'max_cycles_train': train_df.groupby('unit')['cycle'].max().max(),
                'max_cycles_test': test_df.groupby('unit')['cycle'].max().max(),
                'operating_conditions': config['operating_conditions'],
                'fault_modes': config['fault_modes'],
                'description': config['description'],
                'rul_range_train': [train_df['rul'].min(), train_df['rul'].max()],
                'rul_range_test': [test_df['rul'].min(), test_df['rul'].max()]
            }
        }
        
        self.datasets[dataset_name] = dataset
        print(f"✓ {dataset_name} loaded: {len(train_df)} train, {len(test_df)} test samples")
        return dataset
    
    def _identify_operating_conditions(self, train_df: pd.DataFrame, 
                                     test_df: pd.DataFrame, 
                                     config: Dict) -> Dict:
        """
        Identify operating condition clusters for multi-condition datasets
        Uses the three operational settings (setting_1, setting_2, setting_3)
        """
        from sklearn.cluster import KMeans
        
        # Combine train and test for clustering
        all_data = pd.concat([train_df, test_df])
        
        # Use operational settings for clustering  
        settings_cols = ['setting_1', 'setting_2', 'setting_3']
        settings_data = all_data[settings_cols].values
        
        # Perform clustering
        n_clusters = config['operating_conditions']
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(settings_data)
        
        # Add cluster labels back to original data
        all_data['operating_condition'] = clusters
        
        # Split back into train and test
        train_size = len(train_df)
        train_df['operating_condition'] = all_data['operating_condition'].iloc[:train_size].values
        test_df['operating_condition'] = all_data['operating_condition'].iloc[train_size:].values
        
        # Create operating condition summary
        op_conditions = {
            'n_conditions': n_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'settings_columns': settings_cols,
            'train_distribution': train_df['operating_condition'].value_counts().to_dict(),
            'test_distribution': test_df['operating_condition'].value_counts().to_dict()
        }
        
        return op_conditions

    def load_nasa_milling(self, dataset_path: Path = None) -> Dict:
        """
        Load NASA Milling dataset for tool wear prediction

        Args:
            dataset_path: Path to the NASA Milling dataset folder (defaults to NASA_MILLING_DIR from config)

        Returns:
            Dict with processed milling data
        """
        if dataset_path is None:
            dataset_path = NASA_MILLING_DIR

        config = DatasetConfig.NASA_MILLING

        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"NASA Milling dataset directory not found: {dataset_dir}")
        
        print("Loading NASA Milling dataset...")
        
        # Look for the main data file (usually train.csv or mill.csv)
        main_files = list(dataset_dir.glob("*.csv"))
        if not main_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
        
        # Try common file names
        main_file = None
        for filename in ["mill.csv", "train.csv", "data.csv", "milling.csv"]:
            candidate = dataset_dir / filename
            if candidate.exists():
                main_file = candidate
                break
        
        if main_file is None:
            main_file = main_files[0]  # Use first CSV file found
            print(f"Using file: {main_file.name}")
        
        # Load main data file
        df = pd.read_csv(main_file)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Process the data based on expected structure
        processed_df = self._process_milling_data(df, config)
        
        # Split into train/test (temporal split for each experiment)
        train_data = []
        test_data = []
        
        for exp_id in processed_df['case'].unique():
            exp_data = processed_df[processed_df['case'] == exp_id].sort_values('run')
            
            # Use 80% for training, 20% for testing
            split_idx = int(0.8 * len(exp_data))
            train_data.append(exp_data.iloc[:split_idx])
            test_data.append(exp_data.iloc[split_idx:])
        
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        # Calculate RUL based on tool wear
        if 'VB' in train_df.columns:
            # RUL = max_wear - current_wear for each experiment
            for exp_id in train_df['case'].unique():
                mask = train_df['case'] == exp_id
                max_wear = train_df[mask]['VB'].max()
                train_df.loc[mask, 'rul'] = max_wear - train_df.loc[mask, 'VB']
            
            for exp_id in test_df['case'].unique():
                mask = test_df['case'] == exp_id
                max_wear = test_df[mask]['VB'].max()
                test_df.loc[mask, 'rul'] = max_wear - test_df.loc[mask, 'VB']
        
        dataset = {
            'train': train_df,
            'test': test_df,
            'combined': processed_df,
            'config': config,
            'info': {
                'n_experiments': processed_df['case'].nunique(),
                'n_features': len([col for col in processed_df.columns 
                                 if col not in ['case', 'run', 'VB', 'rul']]),
                'total_measurements': len(processed_df),
                'source': config['source'],
                'wear_range': [processed_df['VB'].min(), processed_df['VB'].max()] 
                             if 'VB' in processed_df.columns else None,
                'sampling_rate_ms': config['sampling_rate']
            }
        }
        
        self.datasets['NASA_Milling'] = dataset
        print(f"✓ NASA Milling loaded: {len(train_df)} train, {len(test_df)} test samples")
        return dataset

    def _process_milling_data(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Process raw milling data to extract relevant features
        Expected columns after index removal: ['case', 'run', 'VB', 'time', 'DOC', 'feed', 'material', 'smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
        """
        processed_df = df.copy()
        
        # If we loaded with index_col=0, columns should already be correct
        expected_cols = ['case', 'run', 'VB', 'time', 'DOC', 'feed', 'material', 'smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
        
        # Basic column validation
        if len(processed_df.columns) == len(expected_cols):
            processed_df.columns = expected_cols
            print(f"  Applied expected column names: {expected_cols}")
        else:
            print(f"  Warning: Expected {len(expected_cols)} columns, got {len(processed_df.columns)}")
            print(f"  Using existing column names: {list(processed_df.columns)}")
        
        # Ensure required columns exist and have correct types
        required_cols = ['case', 'run', 'VB']
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to appropriate data types
        processed_df['case'] = processed_df['case'].astype(int)
        processed_df['run'] = processed_df['run'].astype(int) 
        processed_df['VB'] = pd.to_numeric(processed_df['VB'], errors='coerce')
        
        # Remove any rows with invalid VB values
        initial_len = len(processed_df)
        processed_df = processed_df.dropna(subset=['VB'])
        if len(processed_df) < initial_len:
            print(f"  Removed {initial_len - len(processed_df)} rows with invalid VB values")
        
        return processed_df
        
    # Convenience methods for individual datasets
    def load_fd001(self, **kwargs) -> Dict:
        """Load FD001 C-MAPSS dataset"""
        return self.load_cmapss_dataset('FD001', **kwargs)
    
    def load_fd002(self, **kwargs) -> Dict:
        """Load FD002 C-MAPSS dataset"""
        return self.load_cmapss_dataset('FD002', **kwargs)
        
    def load_fd003(self, **kwargs) -> Dict:
        """Load FD003 C-MAPSS dataset"""
        return self.load_cmapss_dataset('FD003', **kwargs)
        
    def load_fd004(self, **kwargs) -> Dict:
        """Load FD004 C-MAPSS dataset"""
        return self.load_cmapss_dataset('FD004', **kwargs)
    
    def load_all_cmapss(self) -> Dict:
        """
        Load all four C-MAPSS datasets at once
        
        Returns:
            Dict with all four datasets
        """
        datasets = {}
        
        for dataset_name in ['FD001', 'FD002', 'FD003', 'FD004']:
            try:
                print(f"\n--- Attempting to load {dataset_name} ---")
                datasets[dataset_name] = self.load_cmapss_dataset(dataset_name)
                print(f"✓ {dataset_name} loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {dataset_name}")
                print(f"  Error: {e}")
                
                # Try to provide more debugging info
                try:
                    debug_info = self.debug_dataset_files(dataset_name)
                    print(f"  Debug info:")
                    print(f"    Files exist: {debug_info['files_exist']}")
                    if debug_info['files_exist']['test'] and debug_info['files_exist']['rul']:
                        print(f"    Estimated test units: {debug_info.get('estimated_test_units', 'N/A')}")
                        print(f"    RUL values count: {debug_info.get('rul_values', 'N/A')}")
                except Exception as debug_e:
                    print(f"  Debug failed: {debug_e}")
                
                datasets[dataset_name] = None
        
        return datasets
        
    def validate_dataset(self, dataset_name: str) -> Dict:
        """
        Validate dataset integrity and return validation report
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Dict with validation results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset = self.datasets[dataset_name]
        validation_report = {
            'dataset_name': dataset_name,
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for required components
        required_keys = ['train', 'test', 'config']
        for key in required_keys:
            if key not in dataset:
                validation_report['issues'].append(f"Missing required key: {key}")
                validation_report['is_valid'] = False
        
        if not validation_report['is_valid']:
            return validation_report
        
        train_df = dataset['train']
        test_df = dataset['test']
        config = dataset['config']
        
        # Check data shapes and types
        validation_report['statistics']['train_shape'] = train_df.shape
        validation_report['statistics']['test_shape'] = test_df.shape
        
        # Check for missing values
        train_missing = train_df.isnull().sum().sum()
        test_missing = test_df.isnull().sum().sum()
        
        if train_missing > 0:
            validation_report['warnings'].append(f"Training data has {train_missing} missing values")
        if test_missing > 0:
            validation_report['warnings'].append(f"Test data has {test_missing} missing values")
        
        # Check for required columns
        required_cols = ['rul']
        for col in required_cols:
            if col not in train_df.columns:
                validation_report['issues'].append(f"Missing required column in train: {col}")
                validation_report['is_valid'] = False
            if col not in test_df.columns:
                validation_report['issues'].append(f"Missing required column in test: {col}")
                validation_report['is_valid'] = False
        
        # Check RUL distribution
        if validation_report['is_valid']:
            validation_report['statistics']['train_rul_range'] = [
                train_df['rul'].min(), train_df['rul'].max()
            ]
            validation_report['statistics']['test_rul_range'] = [
                test_df['rul'].min(), test_df['rul'].max()
            ]
            
            # Check for negative RUL values
            if (train_df['rul'] < 0).any():
                validation_report['issues'].append("Training data contains negative RUL values")
                validation_report['is_valid'] = False
            if (test_df['rul'] < 0).any():
                validation_report['issues'].append("Test data contains negative RUL values")
                validation_report['is_valid'] = False
        
        return validation_report

    def debug_dataset_files(self, dataset_name: str) -> Dict:
        """
        Debug helper to check file existence and basic structure for a dataset

        Args:
            dataset_name: Name of the dataset to debug

        Returns:
            Dict with file existence and basic stats
        """
        config = getattr(DatasetConfig, dataset_name)
        files_exist = {'train': False, 'test': False, 'rul': False}
        debug_info = {'files_exist': files_exist}

        # Check file existence
        train_file = self.data_root / "C_MAPSS" / f"train_{dataset_name}.txt"
        test_file = self.data_root / "C_MAPSS" / f"test_{dataset_name}.txt"
        rul_file = self.data_root / "C_MAPSS" / f"RUL_{dataset_name}.txt"

        files_exist['train'] = train_file.exists()
        files_exist['test'] = test_file.exists()
        files_exist['rul'] = rul_file.exists()

        # Try to get basic info if files exist
        if files_exist['test']:
            try:
                test_df = pd.read_csv(test_file, sep=r'\s+', header=None, engine='python')
                debug_info['estimated_test_units'] = test_df[0].nunique()  # unit column
            except Exception:
                debug_info['estimated_test_units'] = 'error reading file'

        if files_exist['rul']:
            try:
                rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, engine='python')
                debug_info['rul_values'] = len(rul_df)
            except Exception:
                debug_info['rul_values'] = 'error reading file'

        return debug_info

    def get_dataset_summary(self) -> Dict:
        """
        Get summary of all loaded datasets
        """
        summary = {
            'loaded_datasets': list(self.datasets.keys()),
            'total_datasets': len(self.datasets),
            'details': {}
        }
        
        for name, dataset in self.datasets.items():
            if 'info' in dataset:
                summary['details'][name] = dataset['info']
            
            # Add validation status
            try:
                validation = self.validate_dataset(name)
                summary['details'][name]['is_valid'] = validation['is_valid']
                summary['details'][name]['n_issues'] = len(validation['issues'])
                summary['details'][name]['n_warnings'] = len(validation['warnings'])
            except Exception as e:
                summary['details'][name]['validation_error'] = str(e)
        
        return summary
    
    def get_dataset(self, dataset_name: str) -> Dict:
        """
        Get a specific dataset
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(self.datasets.keys())}")
        return self.datasets[dataset_name]
