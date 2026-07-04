#!/usr/bin/env python3
"""
Test the fixed loader functionality
"""

import sys
from pathlib import Path
from attacks.loader import load_attack

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_fixed_loader():
    """Test that the loader works correctly with enhanced mock model"""
    print("🧪 Testing Fixed Attack Loader\n")
    
    # Enhanced mock model with all required interfaces
    class EnhancedMockModel:
        def __init__(self):
            self.device = "cpu"
            self.model = None
            self.config = {}
            
            # Create proper tokenizer mock
            class MockTokenizer:
                def __init__(self):
                    self.pad_token = "[PAD]"
                    self.eos_token = "[EOS]"
                    self.bos_token = "[BOS]"
                    self.unk_token = "[UNK]"
                    self.vocab_size = 50257
                
                def encode(self, text, **kwargs):
                    return [1, 2, 3, 4, 5]
                
                def decode(self, token_ids, **kwargs):
                    return "mock decoded text"
                
                def __call__(self, text, **kwargs):
                    return {"input_ids": [[1, 2, 3, 4, 5]]}
            
            self.tokenizer = MockTokenizer()
        
        def get_tokenizer(self):
            """Required by white-box attacks"""
            return self.tokenizer
        
        def generate(self, *args, **kwargs):
            return ["mock response"]
        
        def __call__(self, *args, **kwargs):
            return {"logits": None}
    
    model = EnhancedMockModel()
    model_params = {"temperature": 0, "max_tokens": 50}
    
    # Test a few representative attacks
    test_attacks = [
        ("ArtPrompt", "black_box"),
        ("PAIR", "black_box"), 
        ("AutoDAN", "white_box"),
        ("COLD", "white_box"),
        ("GCGAttack", "white_box"),
        ("no_attack", "special")
    ]
    
    results = {"success": 0, "failed": 0}
    
    for attack_name, attack_type in test_attacks:
        print(f"🔧 Testing {attack_name} ({attack_type})...")
        try:
            if attack_name == "no_attack":
                attack = load_attack("no_attack", target_model=model)
            else:
                attack = load_attack(
                    attack_name,
                    target_model=model,
                    target_model_parameters=model_params
                )
            print(f"  ✅ Success: {type(attack).__name__}")
            results["success"] += 1
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results["failed"] += 1
    
    print(f"\n📊 Results: {results['success']}/{results['success']+results['failed']} passed")
    return results["failed"] == 0

if __name__ == "__main__":
    success = test_fixed_loader()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    sys.exit(0 if success else 1)