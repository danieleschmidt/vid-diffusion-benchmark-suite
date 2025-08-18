#!/usr/bin/env python3
"""Isolated test for security validation fix."""

import logging

logger = logging.getLogger(__name__)

class SafetyValidator:
    """Input validation and safety checks."""
    
    @staticmethod
    def validate_prompts(prompts, max_length=1000, max_count=100):
        """Validate and sanitize prompts."""
        if len(prompts) > max_count:
            logger.warning(f"Too many prompts ({len(prompts)}), limiting to {max_count}")
            prompts = prompts[:max_count]
        
        validated = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                logger.warning(f"Skipping non-string prompt at index {i}")
                continue
                
            if len(prompt) > max_length:
                logger.warning(f"Truncating long prompt at index {i}")
                prompt = prompt[:max_length] + "..."
            
            # Basic safety filtering
            if any(pattern in prompt.lower() for pattern in ['<script', 'javascript:', 'eval(', 'exec(', '__import__']):
                logger.warning(f"Skipping potentially unsafe prompt at index {i}")
                continue
                
            validated.append(prompt.strip())
        
        return validated

def test_security_validation():
    """Test that security validation catches all malicious patterns."""
    print("Testing security validation...")
    
    # Test malicious input filtering
    malicious_prompts = [
        "<script>alert('xss')</script>",
        "javascript:void(0)",
        "exec('rm -rf /')",
        "__import__('os').system('ls')",
        "eval(malicious_code)",
        "Normal safe prompt"
    ]
    
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

if __name__ == "__main__":
    import sys
    success = test_security_validation()
    sys.exit(0 if success else 1)