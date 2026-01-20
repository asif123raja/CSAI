"""
Inference script for testing the trained model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import config

class CICIDSClassifier:
    def __init__(self, model_path, use_lora=True):
        self.model_path = model_path
        self.use_lora = use_lora
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        print("Loading model...")
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter if using
        if self.use_lora:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.model_path,
                device_map="auto"
            )
            print("Loaded LoRA adapter")
        else:
            self.model.load_state_dict(
                torch.load(f"{self.model_path}/pytorch_model.bin")
            )
        
        self.model.eval()
        print("Model loaded successfully")
    
    def classify_traffic(self, features):
        """Classify network traffic"""
        prompt = self.create_prompt(features)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=config.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def create_prompt(self, features):
        """Create classification prompt"""
        return f"""Analyze this network flow for security threats:

Flow Characteristics:
- Source Port: {features.get('src_port', 'Unknown')}
- Destination Port: {features.get('dst_port', 'Unknown')}
- Protocol: {features.get('protocol', 'Unknown')}
- Duration: {features.get('duration', 'Unknown')} microseconds
- Forward Packets: {features.get('fwd_packets', 'Unknown')}
- Backward Packets: {features.get('bwd_packets', 'Unknown')}
- Forward Bytes: {features.get('fwd_bytes', 'Unknown')}
- Backward Bytes: {features.get('bwd_bytes', 'Unknown')}
- Bytes per Second: {features.get('flow_bytes_sec', 'Unknown')}

Task: Classify this network flow as BENIGN, MALICIOUS, or ADVERSARIAL_ATTACK.
Provide your reasoning and key indicators."""

def main():
    # Test the model
    classifier = CICIDSClassifier(
        "./phi3-cicids-adversarial-final",
        use_lora=True
    )
    
    # Test cases
    test_cases = [
        {
            'name': 'Normal HTTP Traffic',
            'features': {
                'src_port': '54321',
                'dst_port': '80',
                'protocol': '6',
                'duration': '1500000',
                'fwd_packets': '12',
                'bwd_packets': '10',
                'fwd_bytes': '6000',
                'bwd_bytes': '5500',
                'flow_bytes_sec': '8000'
            }
        },
        {
            'name': 'Suspicious DDoS-like Traffic',
            'features': {
                'src_port': '54321',
                'dst_port': '80',
                'protocol': '6',
                'duration': '50000',
                'fwd_packets': '1000',
                'bwd_packets': '5',
                'fwd_bytes': '64000',
                'bwd_bytes': '320',
                'flow_bytes_sec': '500000'
            }
        },
        {
            'name': 'Adversarial Example',
            'features': {
                'src_port': '54321',
                'dst_port': '22',
                'protocol': '6',
                'duration': '800000',
                'fwd_packets': '50',
                'bwd_packets': '48',
                'fwd_bytes': '35000',
                'bwd_bytes': '34000',
                'flow_bytes_sec': '45000'
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']}")
        print(f"{'='*60}")
        
        result = classifier.classify_traffic(test_case['features'])
        print(result)
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
