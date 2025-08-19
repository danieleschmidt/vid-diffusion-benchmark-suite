"""Basic tests that work without heavy dependencies."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_project_structure():
    """Test basic project structure."""
    src_dir = Path(__file__).parent.parent / "src" / "vid_diffusion_bench"
    
    # Check essential files exist
    essential_files = [
        "__init__.py",
        "exceptions.py", 
        "cli.py",
        "benchmark.py",
        "models/__init__.py",
        "models/base.py",
        "models/registry.py"
    ]
    
    for file_path in essential_files:
        full_path = src_dir / file_path
        assert full_path.exists(), f"Essential file missing: {file_path}"
        print(f"✓ Found {file_path}")

def test_exception_classes():
    """Test exception classes can be imported."""
    try:
        # Mock torch to avoid dependency
        sys.modules['torch'] = type('MockTorch', (), {
            'Tensor': object,
            'cuda': type('cuda', (), {'is_available': lambda: False})
        })
        sys.modules['torchvision'] = object()
        sys.modules['numpy'] = type('MockNumpy', (), {
            'ndarray': object,
            'float32': float
        })
        
        from vid_diffusion_bench.exceptions import (
            VidBenchError, ModelError, ModelNotFoundError, 
            ValidationError, MetricsError
        )
        
        # Test basic exception functionality
        error = ModelNotFoundError("test-model", ["model1", "model2"])
        assert error.model_name == "test-model"
        assert "test-model" in str(error)
        
        # Test error dict conversion
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ModelNotFoundError"
        assert "test-model" in error_dict["message"]
        
        print("✓ Exception classes work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Exception test failed: {e}")
        return False

def test_configuration_files():
    """Test configuration files are valid."""
    project_root = Path(__file__).parent.parent
    
    # Test pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml missing"
    
    content = pyproject_path.read_text()
    assert "vid-diffusion-benchmark-suite" in content
    assert "python" in content.lower()
    
    print("✓ pyproject.toml is valid")
    
    # Test README.md
    readme_path = project_root / "README.md"
    assert readme_path.exists(), "README.md missing"
    
    readme_content = readme_path.read_text()
    assert "Video Diffusion" in readme_content
    assert "benchmark" in readme_content.lower()
    
    print("✓ README.md is valid")

def test_docs_structure():
    """Test documentation structure."""
    docs_dir = Path(__file__).parent.parent / "docs"
    assert docs_dir.exists(), "docs directory missing"
    
    # Check for key documentation files
    doc_files = list(docs_dir.glob("*.md"))
    assert len(doc_files) > 0, "No documentation files found"
    
    print(f"✓ Found {len(doc_files)} documentation files")

def run_all_tests():
    """Run all basic tests."""
    tests = [
        test_project_structure,
        test_exception_classes, 
        test_configuration_files,
        test_docs_structure
    ]
    
    passed = 0
    failed = 0
    
    print("Running basic structure tests...")
    print("=" * 50)
    
    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__}...")
            result = test_func()
            if result is not False:
                passed += 1
                print(f"✓ {test_func.__name__} PASSED")
            else:
                failed += 1
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests PASSED!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)