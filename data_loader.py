"""
Data loading and preprocessing for CICIDS dataset
"""
import pandas as pd
import numpy as np
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CICIDSDataLoader:
    def __init__(self, config):
        self.config = config
        self.numeric_cols = None
        
    def load_cicids_data(self, file_paths=None):
        """
        Load CICIDS data from multiple files
        """
        if file_paths is None:
            # Try to load from common CICIDS filenames
            file_paths = [
                "monday.csv",
                "tuesday.csv",
                "wednesday.csv",
                "thursday.csv",
                "friday.csv"
            ]
        
        dfs = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(f"{self.config.data_path}/{file_path}")
                dfs.append(df)
                print(f"Loaded {file_path} with {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not dfs:
            print("No data loaded, creating synthetic data")
            return self.create_synthetic_data()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sample data
        if self.config.sample_fraction < 1.0:
            combined_df = combined_df.sample(
                frac=self.config.sample_fraction, 
                random_state=42
            )
        
        return self.preprocess_data(combined_df)
    
    def create_synthetic_data(self, n_samples=5000):
        """Create synthetic CICIDS-like data if files not available"""
        np.random.seed(42)
        
        # Generate synthetic data similar to your original code
        data = {
            'Flow Duration': np.random.exponential(1000000, n_samples),
            'Total Fwd Packets': np.random.poisson(10, n_samples),
            'Total Backward Packets': np.random.poisson(8, n_samples),
            'Total Length of Fwd Packets': np.random.exponential(5000, n_samples),
            'Total Length of Bwd Packets': np.random.exponential(4000, n_samples),
            'Fwd Packet Length Mean': np.random.normal(500, 100, n_samples),
            'Bwd Packet Length Mean': np.random.normal(450, 100, n_samples),
            'Flow Bytes/s': np.random.lognormal(10, 1, n_samples),
            'Flow Packets/s': np.random.lognormal(7, 1, n_samples),
            'Flow IAT Mean': np.random.exponential(10000, n_samples),
            'Fwd IAT Mean': np.random.exponential(8000, n_samples),
            'Bwd IAT Mean': np.random.exponential(9000, n_samples),
            'Fwd PSH Flags': np.random.binomial(1, 0.1, n_samples),
            'Bwd PSH Flags': np.random.binomial(1, 0.1, n_samples),
            'Fwd URG Flags': np.random.binomial(1, 0.01, n_samples),
            'Bwd URG Flags': np.random.binomial(1, 0.01, n_samples),
            'Fwd Header Length': np.random.poisson(40, n_samples),
            'Bwd Header Length': np.random.poisson(40, n_samples),
            'Fwd Packets/s': np.random.lognormal(6, 1, n_samples),
            'Bwd Packets/s': np.random.lognormal(5, 1, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create synthetic labels
        labels = ['BENIGN'] * int(n_samples * 0.8) + \
                ['DDoS'] * int(n_samples * 0.08) + \
                ['PortScan'] * int(n_samples * 0.06) + \
                ['BruteForce'] * int(n_samples * 0.04) + \
                ['Web Attack'] * int(n_samples * 0.02)
        random.shuffle(labels)
        df['Label'] = labels[:n_samples]
        
        return self.preprocess_data(df)
    
    def preprocess_data(self, df):
        """Preprocess the loaded data"""
        # Clean data
        df = df.dropna()
        
        # Convert to binary labels
        df['binary_label'] = df['Label'].apply(
            lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK'
        )
        
        # Identify numeric columns
        self.numeric_cols = [
            col for col in df.columns 
            if col not in ['Label', 'binary_label'] 
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        print(f"Loaded dataset size: {len(df)} rows")
        print(f"Class distribution:\n{df['binary_label'].value_counts()}")
        print(f"Numeric features: {len(self.numeric_cols)}")
        
        return df
    
    def prepare_datasets(self, df, tokenizer, adversarial_df=None):
        """Prepare datasets for training"""
        if adversarial_df is not None:
            df = pd.concat([df, adversarial_df], ignore_index=True)
        
        # Format for LLM
        formatted_data = []
        for _, row in df.iterrows():
            formatted_data.append(self.format_for_llm(row))
        
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(batch):
            encoding = tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            encoding["labels"] = encoding["input_ids"].clone()
            return encoding
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]
        )
        
        # Split
        split_dataset = tokenized_dataset.train_test_split(
            test_size=self.config.test_size, 
            seed=42
        )
        
        return split_dataset["train"], split_dataset["test"]
    
    def format_for_llm(self, row):
        """Format a single row for LLM training"""
        # Extract features with defaults
        features = {
            'src_port': row.get('Source Port', row.get('Src Port', '54321')),
            'dst_port': row.get('Destination Port', row.get('Dst Port', '80')),
            'protocol': row.get('Protocol', '6'),
            'duration': row.get('Flow Duration', '1000000'),
            'fwd_packets': row.get('Total Fwd Packets', '10'),
            'bwd_packets': row.get('Total Backward Packets', '8'),
            'fwd_bytes': row.get('Total Length of Fwd Packets', '5000'),
            'bwd_bytes': row.get('Total Length of Bwd Packets', '4000'),
            'flow_bytes_sec': row.get('Flow Bytes/s', '10000'),
        }
        
        # Create prompt
        prompt = f"""Analyze this network flow for security threats:

Flow Characteristics:
- Source Port: {features['src_port']}
- Destination Port: {features['dst_port']}
- Protocol: {features['protocol']} ({"TCP" if features['protocol'] == '6' else "UDP" if features['protocol'] == '17' else "Other"})
- Duration: {features['duration']} microseconds
- Forward Packets: {features['fwd_packets']}
- Backward Packets: {features['bwd_packets']}
- Forward Bytes: {features['fwd_bytes']}
- Backward Bytes: {features['bwd_bytes']}
- Bytes per Second: {features['flow_bytes_sec']}

Task: Classify this network flow as BENIGN, MALICIOUS, or ADVERSARIAL_ATTACK.
Provide your reasoning and key indicators."""

        # Create response
        if row.get('binary_label') == 'ADVERSARIAL_ATTACK':
            original_label = row.get('original_label', 'UNKNOWN')
            adv_type = row.get('adversarial_type', 'unknown')
            response = f"""ANALYSIS: This appears to be an ADVERSARIAL_ATTACK.

REASONING: The flow exhibits characteristics of a {original_label} that has been deliberately modified to evade detection.
Key indicators include feature values that are statistically inconsistent with normal traffic patterns.

INDICATORS:
- Unusual correlation between flow duration and packet counts
- Packet size distributions that deviate from expected protocols
- Timing patterns inconsistent with the observed service
- Feature values that fall between typical benign and malicious ranges

CONCLUSION: ADVERSARIAL_ATTACK ({adv_type}, originally {original_label})"""
        
        elif row['binary_label'] == 'ATTACK':
            attack_type = row.get('Label', 'MALICIOUS')
            response = f"""ANALYSIS: This appears to be MALICIOUS activity.

REASONING: The flow exhibits clear indicators of {attack_type} behavior.

INDICATORS:
- Suspicious port and protocol combinations
- Unusual packet size distributions
- Abnormal flow duration and timing patterns
- High volume of traffic indicative of scanning/exploitation

CONCLUSION: MALICIOUS ({attack_type})"""
        
        else:  # BENIGN
            response = f"""ANALYSIS: This appears to be BENIGN network traffic.

REASONING: The flow characteristics match expected patterns for normal network operations.

INDICATORS:
- Consistent protocol and port usage
- Normal flow duration and timing
- Expected packet size distributions
- Balanced forward/backward traffic ratios

CONCLUSION: BENIGN"""

        return {"text": f"### Instruction: {prompt}\n### Response: {response}"}