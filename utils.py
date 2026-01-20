"""
Utility functions
"""
import json
from pathlib import Path
import torch

def save_config(config, path):
    """Save configuration to JSON file"""
    config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                  for k, v in config.__dict__.items()}
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config(config_class, path):
    """Load configuration from JSON file"""
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    return config_class(**config_dict)

def setup_directories(config):
    """Create necessary directories"""
    directories = [
        config.checkpoint_dir,
        "./logs",
        "./results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Directories created: {directories}")

def estimate_memory_usage(model, batch_size, seq_length):
    """Estimate GPU memory usage"""
    # Rough estimation formula
    params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if next(model.parameters()).dtype == torch.float16 else 4
    
    param_memory = params * bytes_per_param
    activation_memory = batch_size * seq_length * model.config.hidden_size * 4 * 10  # Rough estimate
    
    total_memory = (param_memory + activation_memory) / (1024**3)  # Convert to GB
    
    print(f"Estimated memory usage: {total_memory:.2f} GB")
    print(f"Parameters: {params:,}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_length}")
    
    return total_memory
