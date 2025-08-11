"""
Basic structure tests that don't require heavy ML dependencies.

Tests the fundamental package structure, imports, and basic functionality
without requiring torch, transformers, or other heavy dependencies.
"""

import sys
import importlib
from pathlib import Path
import pytest


class TestPackageStructure:
    """Test basic package structure and imports."""
    
    def test_package_directory_exists(self):
        """Test that the main package directory exists."""
        package_dir = Path(__file__).parent.parent / "src" / "vid_diffusion_bench"
        assert package_dir.exists(), "Package directory should exist"
        assert package_dir.is_dir(), "Package directory should be a directory"
    
    def test_init_file_exists(self):
        """Test that __init__.py exists in the main package."""
        init_file = Path(__file__).parent.parent / "src" / "vid_diffusion_bench" / "__init__.py"
        assert init_file.exists(), "__init__.py should exist"
        assert init_file.is_file(), "__init__.py should be a file"
    
    def test_core_modules_exist(self):
        """Test that core module files exist."""
        package_dir = Path(__file__).parent.parent / "src" / "vid_diffusion_bench"
        
        core_modules = [
            "benchmark.py",
            "metrics.py", 
            "prompts.py",
            "profiler.py",
            "cli.py"
        ]
        
        for module in core_modules:
            module_path = package_dir / module
            assert module_path.exists(), f"{module} should exist"
    
    def test_subpackages_exist(self):
        """Test that expected subpackages exist."""
        package_dir = Path(__file__).parent.parent / "src" / "vid_diffusion_bench"
        
        subpackages = [
            "models",
            "api", 
            "monitoring",
            "security",
            "research"
        ]
        
        for subpackage in subpackages:
            subpackage_path = package_dir / subpackage
            assert subpackage_path.exists(), f"{subpackage} subpackage should exist"
            assert (subpackage_path / "__init__.py").exists(), f"{subpackage}/__init__.py should exist"


class TestBasicImports:
    """Test basic imports without heavy dependencies."""
    
    def test_package_metadata(self):
        """Test that package metadata is accessible."""
        # Mock the heavy dependencies
        import sys
        from unittest.mock import Mock
        
        # Mock torch and related modules
        sys.modules['torch'] = Mock()
        sys.modules['transformers'] = Mock()
        sys.modules['diffusers'] = Mock()
        sys.modules['accelerate'] = Mock()
        
        try:
            # Import package metadata
            import vid_diffusion_bench
            assert hasattr(vid_diffusion_bench, '__version__'), "Package should have __version__"
            assert hasattr(vid_diffusion_bench, '__author__'), "Package should have __author__"
            assert vid_diffusion_bench.__version__ == "0.1.0", "Version should match pyproject.toml"
        finally:
            # Clean up mocks
            for module in ['torch', 'transformers', 'diffusers', 'accelerate']:
                if module in sys.modules:
                    del sys.modules[module]
    
    def test_cli_module_structure(self):
        """Test CLI module has expected structure."""
        cli_file = Path(__file__).parent.parent / "src" / "vid_diffusion_bench" / "cli.py"
        assert cli_file.exists(), "CLI module should exist"
        
        # Read and check for expected patterns
        content = cli_file.read_text()
        assert "click" in content or "argparse" in content or "typer" in content, "CLI should use a command line library"
        assert "def main" in content or "def cli" in content, "CLI should have main/cli function"


class TestConfigurationFiles:
    """Test configuration and setup files."""
    
    def test_pyproject_toml_structure(self):
        """Test pyproject.toml has expected structure."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"
        
        content = pyproject_path.read_text()
        assert "[project]" in content, "Should have [project] section"
        assert "vid-diffusion-benchmark-suite" in content, "Should have correct package name"
        assert "torch" in content, "Should list torch as dependency"
    
    def test_readme_exists(self):
        """Test README.md exists and has content."""
        readme_path = Path(__file__).parent.parent / "README.md"
        assert readme_path.exists(), "README.md should exist"
        
        content = readme_path.read_text()
        assert len(content) > 100, "README should have substantial content"
        assert "Video Diffusion" in content, "README should mention Video Diffusion"
    
    def test_docker_files_exist(self):
        """Test Docker configuration files exist."""
        repo_root = Path(__file__).parent.parent
        
        docker_files = [
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.dev.yml"
        ]
        
        for docker_file in docker_files:
            docker_path = repo_root / docker_file
            assert docker_path.exists(), f"{docker_file} should exist"


class TestDirectoryStructure:
    """Test overall directory structure."""
    
    def test_tests_directory_structure(self):
        """Test tests directory has expected structure."""
        tests_dir = Path(__file__).parent
        
        expected_test_files = [
            "test_benchmark.py",
            "test_metrics.py",
            "test_prompts.py",
            "test_profiler.py"
        ]
        
        for test_file in expected_test_files:
            test_path = tests_dir / test_file
            assert test_path.exists(), f"{test_file} should exist"
    
    def test_docs_directory_structure(self):
        """Test docs directory exists and has content."""
        docs_dir = Path(__file__).parent.parent / "docs"
        assert docs_dir.exists(), "docs directory should exist"
        
        # Check for key documentation files
        expected_docs = [
            "API_REFERENCE.md",
            "DEPLOYMENT.md", 
            "TESTING.md"
        ]
        
        for doc_file in expected_docs:
            doc_path = docs_dir / doc_file
            assert doc_path.exists(), f"{doc_file} should exist in docs"
    
    def test_scripts_directory(self):
        """Test scripts directory has utility scripts."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        assert scripts_dir.exists(), "scripts directory should exist"
        
        # Check for setup script
        setup_script = scripts_dir / "setup.sh"
        assert setup_script.exists(), "setup.sh should exist"


class TestCodeQuality:
    """Test basic code quality aspects."""
    
    def test_python_files_are_valid(self):
        """Test that all Python files have valid syntax."""
        src_dir = Path(__file__).parent.parent / "src" / "vid_diffusion_bench"
        
        python_files = list(src_dir.rglob("*.py"))
        assert len(python_files) > 0, "Should find Python files"
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file}: {e}")
    
    def test_no_obvious_security_issues(self):
        """Test for obvious security anti-patterns."""
        src_dir = Path(__file__).parent.parent / "src"
        
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()
            
            # Check for dangerous patterns
            dangerous_patterns = [
                "eval(",
                "exec(",
                "os.system(",
                "subprocess.call(",
                "__import__(",
                "pickle.loads(",
            ]
            
            for pattern in dangerous_patterns:
                if pattern in content:
                    # Allow if it's in a comment or string literal context
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern in line and not (line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'")):
                            # Special cases: .eval() method call and regex patterns are safe
                            if pattern == "eval(" and (".eval(" in line or "r'" in line or 'r"' in line):
                                continue
                            if pattern == "exec(" and ("r'" in line or 'r"' in line):
                                continue
                            pytest.fail(f"Potentially dangerous pattern '{pattern}' found in {py_file}:{i+1}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])