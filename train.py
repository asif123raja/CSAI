"""
Main training script with checkpointing
"""
from transformers import TrainingArguments, DataCollatorForLanguageModeling
import os
from pathlib import Path
import sys
import pandas as pd
import torch
from datasets import load_from_disk


# Add current directory to path
sys.path.append('.')

from config import config, TrainingConfig
from data_loader import CICIDSDataLoader
from adversarial_generator import AdversarialGenerator
from model_setup import ModelSetup
from trainer import AdversarialTrainer
from utils import setup_directories, save_config, estimate_memory_usage

# --- Monkey Patch for Transformers DynamicCache issue ---
# Phi-3 remote code uses get_usable_length which was removed in recent transformers
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, input_seq_length, layer_idx=None):
        if layer_idx is None:
            layer_idx = 0
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = get_usable_length
    print("[+] Applied monkey patch for DynamicCache.get_usable_length")
# --------------------------------------------------------

def validate_datasets(train_dataset, eval_dataset):
    """Validate dataset integrity before training"""
    print("\n=== Validating Datasets ===")
    
    # Check dataset sizes
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty!")
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty!")
    
    print(f"[+] Training dataset: {len(train_dataset)} samples")
    print(f"[+] Evaluation dataset: {len(eval_dataset)} samples")
    
    # Check sample structure
    sample = train_dataset[0]
    required_keys = ['input_ids', 'attention_mask', 'labels']
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Missing key '{key}' in dataset samples")
    
    print("[+] Dataset structure is valid")
    return True

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[-1])
                checkpoints.append((step, os.path.join(checkpoint_dir, item)))
            except:
                continue
    
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]  # Return path of latest checkpoint

def save_adversarial_data(adversarial_df, filename="adversarial_samples_10k.csv"):
    """Save adversarial data for reuse"""
    adversarial_df.to_csv(filename, index=False)
    print(f"Adversarial data saved to {filename}")

def check_and_optimize_memory():
    """Check available memory and optimize settings"""
    print("\n=== Memory Optimization ===")
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {total_memory:.2f} GB")
        
        # Get free memory
        torch.cuda.empty_cache()
        free_memory = torch.cuda.memory_reserved() / (1024**3)
        print(f"Currently allocated: {free_memory:.2f} GB")
        
        # Adjust config based on available memory
        if total_memory < 7:  # RTX 3050 6GB
            print("Applying RTX 3050 6GB optimizations...")
            config.batch_size = 1
            config.gradient_accumulation_steps = 16
            config.max_length = 256
            config.sample_fraction = 0.2
            print(f"  Batch size: {config.batch_size}")
            print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
            print(f"  Max length: {config.max_length}")
            print(f"  Sample fraction: {config.sample_fraction}")
            
    else:
        print("No GPU detected - training on CPU (very slow!)")
        config.batch_size = 1
        config.fp16 = False

def main():
    print("=== Phi-3 CICIDS Adversarial Training ===")
    # Check and optimize memory
    check_and_optimize_memory()
    # Setup directories
    setup_directories(config)
    save_config(config, "./config.json")
    
    # Check for resume flag
    resume_training = False
    latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
    
    if latest_checkpoint:
        print(f"\nFound existing checkpoint: {latest_checkpoint}")
        response = input("Resume training from checkpoint? (y/n): ").strip().lower()
        if response == 'y':
            resume_training = True
            print(f"Will resume from: {latest_checkpoint}")
    
    # Initialize components
    data_loader = CICIDSDataLoader(config)
    adv_generator = AdversarialGenerator(config)
    model_setup = ModelSetup(config)
    
    # Prepare datasets with checks
    print("\n=== Preparing Datasets ===")
    processed_dir = Path("./processed_data_cleaned")
    train_path = processed_dir / "train"
    eval_path = processed_dir / "eval"
    
    if train_path.exists() and eval_path.exists():
        print("Loading pre-processed datasets from disk...")
        train_dataset = load_from_disk(str(train_path))
        eval_dataset = load_from_disk(str(eval_path))
    else:
        # Load data (always fresh if not cached)
        print("\n=== Loading CICIDS Data ===")
        df = data_loader.load_cicids_data()
        
        # Load or Generate adversarial data
        print("\n=== Handling Adversarial Samples ===")
        adv_file = "adversarial_samples_10k.csv"
        
        if os.path.exists(adv_file):
            print(f"Loading existing adversarial samples from {adv_file}...")
            adversarial_df = pd.read_csv(adv_file)
        else:
            print("Generating new adversarial samples...")
            adversarial_df = adv_generator.generate_adversarial_data(df)
            save_adversarial_data(adversarial_df, adv_file)
        
        print(f"Using {len(adversarial_df)} adversarial samples")
        
        # Initialize tokenizer for data preparation
        print("Initializing tokenizer for data processing...")
        tokenizer = model_setup.setup_tokenizer()
        
        train_dataset, eval_dataset = data_loader.prepare_datasets(
            df, tokenizer, adversarial_df
        )
        
        # Save for future use
        print("Saving processed datasets to disk...")
        processed_dir.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(str(train_path))
        eval_dataset.save_to_disk(str(eval_path))

    # Setup model and tokenizer (moved after data loading)
    print("\n=== Setting up Model ===")
    model, tokenizer = model_setup.setup_model_and_tokenizer()
    
    # Estimate memory usage
    estimate_memory_usage(model, config.batch_size, config.max_length)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    # Validate datasets before training
    validate_datasets(train_dataset, eval_dataset)
    # Training arguments with proper checkpointing
    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=True,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        
        # IMPORTANT: For proper resumption
        overwrite_output_dir=False,  # Don't overwrite existing checkpoints
        resume_from_checkpoint=latest_checkpoint if resume_training else None,
        ignore_data_skip=False,  # Important for resuming data loading
    )
    
    # Initialize trainer
    # Initialize trainer with correct data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        checkpoint_dir=config.checkpoint_dir
    )
    
    # If resuming, restore trainer state
    if resume_training and latest_checkpoint:
        print(f"\n=== Resuming Training ===")
        print(f"Resuming from: {latest_checkpoint}")
        
        # Load trainer state if it exists
        trainer_state_path = os.path.join(latest_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            print("Found trainer state, resuming...")
    
    # Train model
    print("\n=== Starting Training ===")
    
    try:
        trainer.train(resume_from_checkpoint=latest_checkpoint if resume_training else None)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")
        current_step = trainer.state.global_step if hasattr(trainer.state, 'global_step') else "unknown"
        trainer.save_model(os.path.join(config.checkpoint_dir, f"interrupted-{current_step}"))
        print(f"Checkpoint saved at step {current_step}")
        return
    
    # Save final model
    print("\n=== Saving Final Model ===")
    final_path = "./phi3-cicids-adversarial-final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {final_path}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    
    # Save training summary
    with open("training_summary.txt", "w") as f:
        f.write(f"Training completed successfully\n")
        f.write(f"Final model: {final_path}\n")
        f.write(f"Checkpoints: {config.checkpoint_dir}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Evaluation samples: {len(eval_dataset)}\n")
        f.write(f"Adversarial samples: {len(adversarial_df)}\n")

if __name__ == "__main__":
    # Update config for your system
    config.data_path = "./cicids"  # Corrected path
    config.sample_fraction = 0.3  # 30% for RTX 3050 6GB
    config.batch_size = 1  # Start with 2, increase if memory allows
    config.adversarial_samples = 10000  # Fixed variable name (not adversarial_samples_10k)
    config.max_steps = 1000  # Total training steps
    config.save_steps = 25   # Save every 25 steps to protect progress
    config.eval_steps = 25   # Evaluate every 25 steps
    config.gradient_accumulation_steps = 8  # Effective batch size = 2 * 8 = 16
    
    # For RTX 3050 6GB - memory optimization
    config.max_length = 384  # Reduced from 512 for memory
    config.gradient_checkpointing = True  # Save memory
    config.fp16 = True  # Use mixed precision
    
    # Enable these only if you have enough memory
    config.use_flash_attention = False  # Disable if OOM errors
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining was interrupted by user (Ctrl+C)")
        print("Checkpoints have been saved. Run again to resume.")
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Reduce batch_size to 1 if out of memory")
        print("2. Reduce sample_fraction to 0.2")
        print("3. Reduce max_length to 256")
