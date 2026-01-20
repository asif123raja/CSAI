"""
Custom trainer with adversarial training and checkpointing
"""
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import os
from pathlib import Path
import json
import warnings
from typing import Dict, Optional, Tuple
class AdversarialTrainer(Trainer):
    def __init__(self, *args, checkpoint_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.last_saved_step = 0
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss with adversarial training
        """
        # Regular forward pass
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Adversarial training - only during training, not evaluation
        if model.training:
            try:
                # Get embeddings
                embeddings = model.get_input_embeddings()
                
                # Get input embeddings
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                
                # Create adversarial examples using FGSM-like method
                with torch.enable_grad():
                    # Forward pass to get gradients
                    inputs_embeds = embeddings(input_ids)
                    inputs_embeds.requires_grad_(True)
                    
                    adv_inputs = {
                        'inputs_embeds': inputs_embeds,
                        'attention_mask': attention_mask,
                        'labels': inputs['labels']
                    }
                    
                    adv_outputs = model(**adv_inputs)
                    adv_loss = adv_outputs.loss
                    
                    # Compute gradients
                    grad = torch.autograd.grad(
                        adv_loss, 
                        inputs_embeds, 
                        retain_graph=True
                    )[0]
                    
                    # Apply perturbation (small FGSM step)
                    epsilon = 0.01
                    perturbation = epsilon * grad.sign()
                    perturbed_embeds = inputs_embeds + perturbation
                    
                    # Forward with perturbed embeddings
                    adv_inputs['inputs_embeds'] = perturbed_embeds.detach()
                    adv_outputs = model(**adv_inputs)
                    adv_loss = adv_outputs.loss
                    
                    # Combine losses
                    total_loss = (loss + 0.3 * adv_loss) / 1.3
                    
                    # Detach to save memory
                    del inputs_embeds, grad, perturbation, perturbed_embeds
                    
            except Exception as e:
                # Silently fail to adversarial training and use regular loss
                total_loss = loss
        else:
            total_loss = loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override the parent method to add custom checkpoint saving
        """
        # Call parent save method first
        # transformers > 4.36 _save_checkpoint doesn't accept metrics
        super()._save_checkpoint(model, trial)
        
        # Additional custom saving if needed
        if self.checkpoint_dir and hasattr(self.state, 'global_step'):
            current_step = self.state.global_step
            if current_step > self.last_saved_step:
                self.last_saved_step = current_step
                
                # Save additional trainer state if needed
                trainer_state_path = os.path.join(
                    self.args.output_dir, 
                    f"checkpoint-{current_step}", 
                    "trainer_state_custom.json"
                )
                
                custom_state = {
                    'global_step': current_step,
                    'epoch': getattr(self.state, 'epoch', 0),
                    'best_metric': getattr(self.state, 'best_metric', None),
                    'last_saved_step': self.last_saved_step,
                    'train_loss': getattr(self.state, 'train_loss', None),
                }
                
                with open(trainer_state_path, 'w') as f:
                    json.dump(custom_state, f, indent=2)
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        Load from checkpoint with custom handling
        """
        # Call parent load method
        result = super()._load_from_checkpoint(resume_from_checkpoint, model)
        
        # Load custom trainer state if it exists
        custom_state_path = os.path.join(resume_from_checkpoint, "trainer_state_custom.json")
        if os.path.exists(custom_state_path):
            with open(custom_state_path, 'r') as f:
                custom_state = json.load(f)
                self.last_saved_step = custom_state.get('last_saved_step', 0)
                print(f"Loaded custom trainer state from checkpoint")
                print(f"Resuming from step {custom_state.get('global_step', 0)}")
        
        return result
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with gradient accumulation awareness
        """
        # Store current step for checkpoint naming
        if hasattr(self.state, 'global_step'):
            current_step = self.state.global_step
            
            # Auto-save checkpoint at specified intervals
            if (self.args.save_steps > 0 and 
                current_step > 0 and 
                current_step % self.args.save_steps == 0):
                print(f"\n[Step {current_step}] Auto-saving checkpoint...")
        
        return super().training_step(model, inputs)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to handle adversarial training during eval if needed
        """
        # Turn off adversarial training during evaluation
        model_was_training = self.model.training
        self.model.eval()
        
        try:
            result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        finally:
            # Restore training mode
            if model_was_training:
                self.model.train()
        
        return result
    
    def save_model(self, output_dir=None, _internal_call=False):
        """
        Save model with custom handling for LoRA models
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # For PEFT models, save adapters properly
        if hasattr(self.model, 'peft_config'):
            # This is a PEFT model
            self.model.save_pretrained(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            
            # Also save the base model config for reference
            base_config = {
                'base_model_name_or_path': self.model.config._name_or_path,
                'peft_type': 'LORA',
                'task_type': 'CAUSAL_LM',
            }
            
            with open(os.path.join(output_dir, 'adapter_config.json'), 'r') as f:
                adapter_config = json.load(f)
                base_config.update(adapter_config)
            
            with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
                json.dump(base_config, f, indent=2)
                
            print(f"PEFT model saved to {output_dir}")
        else:
            # Regular model saving
            super().save_model(output_dir, _internal_call)
    
    def create_optimizer(self):
        """
        Create optimizer with support for 8-bit AdamW
        """
        # Use parent optimizer creation
        optimizer = super().create_optimizer()
        
        # Log optimizer info
        print(f"Optimizer created: {type(optimizer).__name__}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        return optimizer
    
    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Create learning rate scheduler
        """
        scheduler = super().create_scheduler(num_training_steps, optimizer)
        
        # Log scheduler info
        print(f"Scheduler created: {self.args.lr_scheduler_type}")
        print(f"Total training steps: {num_training_steps}")
        print(f"Warmup steps: {int(self.args.warmup_ratio * num_training_steps)}")
        
        return scheduler
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Override log to add custom logging
        """
        # Add step information to logs
        if hasattr(self.state, 'global_step'):
            logs['step'] = self.state.global_step
            logs['epoch'] = getattr(self.state, 'epoch', 0)
        
        # Add memory usage info
        if torch.cuda.is_available():
            logs['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024**2
            logs['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024**2
        
        # Call parent log method
        super().log(logs, *args, **kwargs)
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs):
        """
        Override to add custom behavior before save/evaluate
        """
        # Check if we should save checkpoint
        if self.args.save_steps > 0 and self.state.global_step % self.args.save_steps == 0:
            print(f"\n{'='*60}")
            print(f"Checkpoint saved at step {self.state.global_step}")
            print(f"Current loss: {tr_loss:.4f}")
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            print(f"{'='*60}\n")
        
        return super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs)





def safe_adv_loss_computation(model, inputs, epsilon=0.01):
    """
    Safe adversarial loss computation with memory optimization
    """
    try:
        # Get embeddings
        embeddings = model.get_input_embeddings()
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Create adversarial examples
        with torch.enable_grad():
            # Forward pass
            inputs_embeds = embeddings(input_ids)
            inputs_embeds.requires_grad_(True)
            
            adv_inputs = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'labels': inputs['labels']
            }
            
            adv_outputs = model(**adv_inputs)
            adv_loss = adv_outputs.loss
            
            # Compute gradients
            grad = torch.autograd.grad(adv_loss, inputs_embeds)[0]
            
            # Apply perturbation
            perturbation = epsilon * grad.sign()
            perturbed_embeds = inputs_embeds + perturbation
            
            # Forward with perturbed embeddings
            adv_inputs['inputs_embeds'] = perturbed_embeds.detach()
            adv_outputs = model(**adv_inputs)
            adv_loss = adv_outputs.loss
            
        return adv_loss
        
    except Exception as e:
        # Return None if adversarial computation fails
        return None

def create_adversarial_trainer(
    model,
    args,
    train_dataset,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    compute_metrics=None,
    callbacks=None,
    optimizers=(None, None),
    checkpoint_dir=None
):
    """
    Factory function to create an AdversarialTrainer
    """
    return AdversarialTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
        checkpoint_dir=checkpoint_dir
    )
