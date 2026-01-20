"""
Model initialization and LoRA configuration
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

class ModelSetup:
    def __init__(self, config):
        self.config = config
    
    def setup_tokenizer(self):
        """Initialize tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,
                trust_remote_code=True
            )"""
Model initialization and LoRA configuration
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

class ModelSetup:
    def __init__(self, config):
        self.config = config
    
    def setup_tokenizer(self):
        """Initialize tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            # Try without fast tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[+] Tokenizer loaded: {self.config.model_name}")
        return tokenizer
    
    def setup_model(self):
        """Initialize model with quantization"""
        print(f"Loading model: {self.config.model_name}")
        
        # Configure quantization
        bnb_config = None
        if self.config.use_4bit:
            print("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.config.torch_dtype
            )
        
        # Load model - Phi-3 doesn't support flash_attention_2 parameter
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                trust_remote_code=True,
                # Phi-3 doesn't support use_flash_attention_2 parameter
                # Remove this line or handle it differently
            )
        except TypeError as e:
            if "use_flash_attention_2" in str(e):
                print("⚠️  Phi-3 doesn't support flash_attention_2, loading without it...")
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=bnb_config,
                    device_map=self.config.device_map,
                    torch_dtype=self.config.torch_dtype,
                    trust_remote_code=True,
                )
            else:
                raise e
        
        # Set padding token ID
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        
        print(f"[+] Model loaded successfully")
        return model
    
    def apply_lora(self, model):
        """Apply LoRA to model"""
        print("Applying LoRA configuration...")
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA for Phi-3
        # Phi-3 uses these common transformer layers
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            # Try different target modules for Phi-3
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj",     # MLP layers
            ],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"[+] LoRA applied successfully")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def setup_model_and_tokenizer(self):
        """Complete setup of model and tokenizer"""
        print("\n" + "="*50)
        print("MODEL SETUP")
        print("="*50)
        
        tokenizer = self.setup_tokenizer()
        model = self.setup_model()
        model = self.apply_lora(model)
        
        print("="*50)
        return model, tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            # Try without fast tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[+] Tokenizer loaded: {self.config.model_name}")
        return tokenizer
    
    def setup_model(self):
        """Initialize model with quantization"""
        print(f"Loading model: {self.config.model_name}")
        
        # Configure quantization
        bnb_config = None
        if self.config.use_4bit:
            print("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.config.torch_dtype
            )
        
        # Load model - Phi-3 doesn't support flash_attention_2 parameter
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                trust_remote_code=True,
                # Phi-3 doesn't support use_flash_attention_2 parameter
                # Remove this line or handle it differently
            )
        except TypeError as e:
            if "use_flash_attention_2" in str(e):
                print("⚠️  Phi-3 doesn't support flash_attention_2, loading without it...")
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=bnb_config,
                    device_map=self.config.device_map,
                    torch_dtype=self.config.torch_dtype,
                    trust_remote_code=True,
                )
            else:
                raise e
        
        # Set padding token ID
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        
        print(f"[+] Model loaded successfully")
        return model
    
    def apply_lora(self, model):
        """Apply LoRA to model"""
        print("Applying LoRA configuration...")
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA for Phi-3
        # Phi-3 uses these common transformer layers
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            # Try different target modules for Phi-3
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj",     # MLP layers
            ],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"[+] LoRA applied successfully")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    def setup_model_and_tokenizer(self):
        """Complete setup of model and tokenizer"""
        print("\n" + "="*50)
        print("MODEL SETUP")
        print("="*50)
        
        tokenizer = self.setup_tokenizer()
        model = self.setup_model()
        model = self.apply_lora(model)
        
        print("="*50)

        return model, tokenizer
