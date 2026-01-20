"""
Configuration settings for Phi-3-mini CICIDS adversarial training
"""
import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model config
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    use_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    
    # Data config
    data_path: str = "./cicids"
    adv_data_path: str = "adversarial_samples_10k.csv"  # Your pre-generated file
    sample_fraction: float = 0.3  # 30% of data
    max_length: int = 384  # Reduced for RTX 3050 memory
    test_size: float = 0.2
    
    # Training config (optimized for RTX 3050 6GB)
    batch_size: int = 2  # Start with 2, adjust based on memory
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    num_train_epochs: int = 3  # Reduced for faster training
    max_steps: int = 1000  # Total training steps
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    
    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Checkpoint config
    checkpoint_dir: str = "./checkpoints"
    save_steps: int = 100
    eval_steps: int = 50
    save_total_limit: int = 3
    
    # Adversarial config
    adversarial_samples: int = 10000  # For reference only
    
    # Device config
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    
    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = True
    use_flash_attention: bool = False  # Phi-3 doesn't support flash attention
    
    # Training stability
    max_grad_norm: float = 1.0  # Gradient clipping
    
    def __post_init__(self):
        """Apply any post-initialization logic"""
        # Ensure paths use forward slashes
        self.data_path = self.data_path.replace("\\", "/")
        
        # Adjust settings based on available memory
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"Available VRAM: {total_memory_gb:.2f} GB")
            
            # Adjust for RTX 3050 6GB
            if total_memory_gb < 7:
                print("Applying RTX 3050 6GB optimizations...")
                self.batch_size = 1
                self.gradient_accumulation_steps = 16
                self.max_length = 256
                self.sample_fraction = 0.2
        else:
            print("No GPU detected, using CPU mode")
            self.batch_size = 1
            self.fp16 = False
            self.device_map = None

config = TrainingConfig()
