import numpy as np
import random
import pandas as pd

class AdversarialGenerator:
    def __init__(self, config):
        self.config = config
    
    def generate_adversarial_data(self, df, num_samples=None):
        """Generate all types of synthetic adversarial data"""
        if num_samples is None:
            num_samples = self.config.adversarial_samples
        
        synthetic_samples = []
        
        # Get numeric columns
        numeric_cols = [col for col in df.columns 
                       if col not in ['Label', 'binary_label']
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        # Generate different types
        types = ['evasion', 'poisoning', 'novel']
        samples_per_type = max(1, num_samples // len(types))
        
        for adv_type in types:
            if adv_type == 'evasion':
                samples = self.generate_evasion_samples(df, numeric_cols, samples_per_type)
            elif adv_type == 'poisoning':
                samples = self.generate_poisoning_samples(df, numeric_cols, samples_per_type)
            else:  # novel
                samples = self.generate_novel_attacks(df, numeric_cols, samples_per_type)
            
            synthetic_samples.extend(samples)
        
        return pd.DataFrame(synthetic_samples)
    
    def generate_evasion_samples(self, df, numeric_cols, num_samples):
        samples = []
        attack_data = df[df['binary_label'] == 'ATTACK']
        benign_stats = df[df['binary_label'] == 'BENIGN'][numeric_cols].mean()
        
        if len(attack_data) == 0:
            return samples
        
        for _ in range(num_samples):
            # Using random.choice or sample with replacement to handle large requests
            sample = attack_data.sample(1, replace=True).iloc[0].copy()
            
            for col in numeric_cols:
                if col in sample and col in benign_stats:
                    alpha = random.uniform(0.2, 0.5)
                    if pd.notna(sample[col]) and pd.notna(benign_stats[col]):
                        sample[col] = sample[col] * (1-alpha) + benign_stats[col] * alpha
            
            sample['binary_label'] = 'ADVERSARIAL_ATTACK'
            sample['original_label'] = sample.get('Label', 'ATTACK')
            sample['adversarial_type'] = 'evasion'
            samples.append(sample.to_dict())
        
        return samples

    def generate_poisoning_samples(self, df, numeric_cols, num_samples):
        samples = []
        benign_data = df[df['binary_label'] == 'BENIGN']
        
        if len(benign_data) == 0:
            return samples
        
        for _ in range(num_samples):
            sample = benign_data.sample(1, replace=True).iloc[0].copy()
            
            for col in numeric_cols:
                if col in sample:
                    std = df[col].std()
                    if pd.notna(std) and std > 0:
                        noise_level = random.uniform(0.1, 0.3)
                        sample[col] += np.random.normal(0, std * noise_level)
            
            sample['binary_label'] = 'ADVERSARIAL_ATTACK'
            sample['original_label'] = 'BENIGN'
            sample['adversarial_type'] = 'poisoning'
            samples.append(sample.to_dict())
        
        return samples

    def generate_novel_attacks(self, df, numeric_cols, num_samples):
        samples = []
        benign_stats = df[df['binary_label'] == 'BENIGN'][numeric_cols].mean()
        attack_stats = df[df['binary_label'] == 'ATTACK'][numeric_cols].mean()
        
        for i in range(num_samples):
            sample = {}
            for col in numeric_cols:
                if col in benign_stats and col in attack_stats:
                    rand_val = random.random()
                    if rand_val < 0.3:
                        sample[col] = attack_stats[col] * random.uniform(2, 5)
                    elif rand_val < 0.5:
                        alpha = random.uniform(0.3, 0.7)
                        sample[col] = benign_stats[col] * alpha + attack_stats[col] * (1-alpha)
                    else:
                        min_val = min(benign_stats[col], attack_stats[col])
                        max_val = max(benign_stats[col], attack_stats[col])
                        sample[col] = random.uniform(min_val * 0.5, max_val * 1.5)
            
            sample['binary_label'] = 'ADVERSARIAL_ATTACK'
            sample['Label'] = 'SYNTHETIC_ATTACK'
            sample['original_label'] = 'SYNTHETIC'
            sample['adversarial_type'] = 'novel'
            samples.append(sample)
        
        return samples

if __name__ == "__main__":
    # 1. Config set to 10,000 samples
    class MockConfig:
        adversarial_samples = 10000 
        
    config = MockConfig()
    
    # 2. Increased initial data size to 5,000 for better statistical variety
    print("Generating synthetic input data...")
    synthetic_data = {
        'Flow Duration': np.random.exponential(1000000, 5000),
        'Total Fwd Packets': np.random.poisson(10, 5000),
        'Total Backward Packets': np.random.poisson(8, 5000),
        'Flow Bytes/s': np.random.lognormal(10, 1, 5000),
        'Label': ['BENIGN'] * 4000 + ['ATTACK'] * 1000,
        'binary_label': ['BENIGN'] * 4000 + ['ATTACK'] * 1000
    }
    df = pd.DataFrame(synthetic_data)
    
    # 3. Generate Adversarial Samples
    print(f"Generating {config.adversarial_samples} adversarial samples...")
    generator = AdversarialGenerator(config)
    adv_df = generator.generate_adversarial_data(df)
    
    # 4. Save to CSV
    output_file = "adversarial_samples_10k.csv"
    adv_df.to_csv(output_file, index=False)
    print(f"Success! Generated {len(adv_df)} samples.")
    print(f"File saved: {output_file}")
