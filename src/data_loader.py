
class FD001DataLoader:
    """
    Data loader specifically for NASA C-MAPSS FD001 dataset
    
    Handles:
    - Loading training/test data
    - RUL calculation and loading
    - Data validation
    - Basic preprocessing options
    """
    
    def __init__(self, data_path="data/raw_data/C_MAPSS"):
        self.data_path = Path(data_path)
        self.columns = self._define_columns()
        
        # Cache for loaded data
        self._train_data = None
        self._test_data = None
        self._rul_data = None
        
        print(f"🏗️ FD001DataLoader initialized")
        print(f"   Data path: {self.data_path}")
        print(f"   Expected columns: {len(self.columns)}")

    def _define_columns(self):
        """Define column names for FD001 dataset"""
        columns = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
        columns += [f'sensor_{i}' for i in range(1, 22)]  # 21 sensors
        return columns
        
    def load_train_data(self, force_reload=False):
        """Load FD001 training data"""
        if self._train_data is not None and not force_reload:
            return self._train_data
        
        train_file = self.data_path / "train_FD001.txt"
        
        if not train_file.exists():
            print(f"❌ Training file not found: {train_file}")
            print("💡 Using sample data for development...")
            return train_data  # Use the sample data created above
        
        try:
            print(f"📂 Loading training data from {train_file}...")
            self._train_data = pd.read_csv(train_file, sep=r'\s+', header=None, names=self.columns)
            
            print(f"✅ Training data loaded successfully!")
            print(f"   Shape: {self._train_data.shape}")
            print(f"   Engines: {self._train_data['engine_id'].nunique()}")
            
            # Calculate RUL for training data
            self._train_data = self._calculate_rul_training(self._train_data)
            
            return self._train_data
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return None
            
    def load_test_data(self, force_reload=False):
        """Load FD001 test data"""
        if self._test_data is not None and not force_reload:
            return self._test_data
        
        test_file = self.data_path / "test_FD001.txt"
        
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            return None
        
        try:
            print(f"📂 Loading test data from {test_file}...")
            self._test_data = pd.read_csv(test_file, sep=r'\s+', header=None, names=self.columns)
            
            print(f"✅ Test data loaded successfully!")
            print(f"   Shape: {self._test_data.shape}")
            print(f"   Engines: {self._test_data['engine_id'].nunique()}")
            
            return self._test_data
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return None

    def load_rul_data(self, force_reload=False):
        """Load true RUL values for test data"""
        if self._rul_data is not None and not force_reload:
            return self._rul_data
        
        rul_file = self.data_path / "RUL_FD001.txt"
        
        if not rul_file.exists():
            print(f"❌ RUL file not found: {rul_file}")
            return None
        
        try:
            print(f"📂 Loading RUL data from {rul_file}...")
            self._rul_data = pd.read_csv(rul_file, header=None, names=['RUL'])
            self._rul_data['engine_id'] = range(1, len(self._rul_data) + 1)
            
            print(f"✅ RUL data loaded successfully!")
            print(f"   Shape: {self._rul_data.shape}")
            print(f"   RUL range: {self._rul_data['RUL'].min()}-{self._rul_data['RUL'].max()}")
            
            return self._rul_data
            
        except Exception as e:
            print(f"❌ Error loading RUL data: {e}")
            return None

    def _calculate_rul_training(self, data):
        """Calculate RUL for training data (reverse cycle count)"""
        print("🔢 Calculating RUL for training data...")
        
        data_with_rul = data.copy()
        
        # For each engine, calculate RUL as (max_cycle - current_cycle)
        for engine_id in data['engine_id'].unique():
            engine_mask = data_with_rul['engine_id'] == engine_id
            max_cycle = data_with_rul[engine_mask]['cycle'].max()
            data_with_rul.loc[engine_mask, 'RUL'] = max_cycle - data_with_rul.loc[engine_mask, 'cycle']
        
        print(f"✅ RUL calculated for training data")
        return data_with_rul

    def validate_data(self, data, data_type="unknown"):
        """Validate loaded data quality"""
        print(f"🔍 Validating {data_type} data...")
        
        issues = []
        
        # Check basic structure
        expected_cols = len(self.columns)
        if 'RUL' in data.columns:
            expected_cols += 1
            
        if data.shape[1] < expected_cols - 1:  # Allow some flexibility
            issues.append(f"Unexpected column count: {data.shape[1]} (expected ~{expected_cols})")
        
        # Check for missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            issues.append(f"Found {missing_count} missing values")
        
        # Check engine ID progression
        for engine_id in data['engine_id'].unique()[:5]:  # Check first 5 engines
            engine_data = data[data['engine_id'] == engine_id]
            cycles = sorted(engine_data['cycle'].values)
            expected_cycles = list(range(1, len(cycles) + 1))
            if cycles != expected_cycles:
                issues.append(f"Engine {engine_id} has irregular cycle progression")
                break
        
        # Check sensor value ranges (basic sanity check)
        sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
        for col in sensor_cols[:3]:  # Check first 3 sensors
            if data[col].std() == 0:
                issues.append(f"{col} has zero variance (constant values)")
        
        if len(issues) == 0:
            print(f"✅ {data_type} data validation passed!")
        else:
            print(f"⚠️ {data_type} data validation found {len(issues)} issues:")
            for issue in issues:
                print(f"   - {issue}")
        
        return len(issues) == 0

    def get_data_summary(self):
        """Get comprehensive data summary"""
        print("📊 FD001 Dataset Summary")
        print("=" * 40)
        
        train_data = self.load_train_data()
        test_data = self.load_test_data()
        rul_data = self.load_rul_data()
        
        if train_data is not None:
            print(f"📈 Training Data:")
            print(f"   Shape: {train_data.shape}")
            print(f"   Engines: {train_data['engine_id'].nunique()}")
            print(f"   Total cycles: {train_data.shape[0]:,}")
            if 'RUL' in train_data.columns:
                print(f"   RUL range: {train_data['RUL'].min()}-{train_data['RUL'].max()}")
        
        if test_data is not None:
            print(f"📊 Test Data:")
            print(f"   Shape: {test_data.shape}")
            print(f"   Engines: {test_data['engine_id'].nunique()}")
            print(f"   Total cycles: {test_data.shape[0]:,}")
        
        if rul_data is not None:
            print(f"🎯 RUL Data:")
            print(f"   Shape: {rul_data.shape}")
            print(f"   RUL range: {rul_data['RUL'].min()}-{rul_data['RUL'].max()}")
    
