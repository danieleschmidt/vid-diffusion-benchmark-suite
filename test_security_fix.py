#!/usr/bin/env python3
"""Test security validation fix specifically."""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_security_validation():
    """Test that security validation catches all malicious patterns."""
    try:
        from vid_diffusion_bench.generation1_enhancements import SafetyValidator
        
        # Test malicious input filtering
        malicious_prompts = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "exec('rm -rf /')",
            "__import__('os').system('ls')",
            "eval(malicious_code)",
            "Normal safe prompt"
        ]
        
        print("Testing security validation...")
        print(f"Input prompts: {malicious_prompts}")
        
        safe_prompts = SafetyValidator.validate_prompts(malicious_prompts)
        print(f"Safe prompts: {safe_prompts}")
        
        # Should filter out all malicious prompts, leaving only 1 safe prompt
        expected_safe_count = 1
        actual_safe_count = len(safe_prompts)
        
        print(f"Expected safe prompts: {expected_safe_count}")
        print(f"Actual safe prompts: {actual_safe_count}")
        
        if actual_safe_count == expected_safe_count:
            print("✅ Security validation works correctly - filtered out all malicious prompts")
            
            # Verify the safe prompt is indeed the expected one
            if "Normal safe prompt" in safe_prompts:
                print("✅ Correct safe prompt preserved")
                return True
            else:
                print("❌ Wrong prompt preserved")
                return False
        else:
            print(f"❌ Security validation failed - expected {expected_safe_count} safe prompts, got {actual_safe_count}")
            
            # Check which patterns are still getting through
            for prompt in safe_prompts:
                for pattern in ['<script', 'javascript:', 'exec(', '__import__', 'eval(']:
                    if pattern in prompt.lower():
                        print(f"❌ Malicious pattern '{pattern}' found in safe prompt: {prompt}")
            
            return False
            
    except Exception as e:
        print(f"❌ Security validation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_security_validation()
    sys.exit(0 if success else 1)