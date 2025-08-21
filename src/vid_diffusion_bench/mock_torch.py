"""Mock torch module for testing without PyTorch dependency."""

import numpy as np
from typing import Tuple, List, Any


class MockTensor:
    """Mock tensor class that mimics torch.Tensor interface."""
    
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data)
        else:
            self.data = np.array([data])
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def size(self) -> Tuple[int, ...]:
        return self.shape
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])
    
    def __setitem__(self, key, value):
        if isinstance(value, MockTensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
    
    def float(self):
        return MockTensor(self.data.astype(np.float32))
    
    def half(self):
        return MockTensor(self.data.astype(np.float16))
    
    def cuda(self):
        return self  # Mock - just return self
    
    def cpu(self):
        return self
    
    def to(self, device):
        return self  # Mock - just return self


def zeros(*shape) -> MockTensor:
    """Create tensor of zeros."""
    return MockTensor(np.zeros(shape))


def ones(*shape) -> MockTensor:
    """Create tensor of ones."""
    return MockTensor(np.ones(shape))


def randn(*shape) -> MockTensor:
    """Create tensor of random normal values."""
    return MockTensor(np.random.randn(*shape))


def rand(*shape) -> MockTensor:
    """Create tensor of random uniform values."""
    return MockTensor(np.random.rand(*shape))


def stack(tensors, dim=0) -> MockTensor:
    """Stack tensors along dimension."""
    arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
    return MockTensor(np.stack(arrays, axis=dim))


def linspace(start, end, steps) -> MockTensor:
    """Create linearly spaced tensor."""
    return MockTensor(np.linspace(start, end, steps))


def sin(tensor) -> MockTensor:
    """Apply sin function."""
    if isinstance(tensor, MockTensor):
        return MockTensor(np.sin(tensor.data))
    return MockTensor(np.sin(tensor))


def cos(tensor) -> MockTensor:
    """Apply cos function."""
    if isinstance(tensor, MockTensor):
        return MockTensor(np.cos(tensor.data))
    return MockTensor(np.cos(tensor))


def tensor(data) -> MockTensor:
    """Create tensor from data."""
    return MockTensor(data)


# Mock CUDA module
class MockCuda:
    """Mock torch.cuda module."""
    
    @staticmethod
    def is_available() -> bool:
        return False  # Always return False for testing
    
    @staticmethod
    def device_count() -> int:
        return 0
    
    @staticmethod
    def empty_cache():
        pass  # No-op
    
    @staticmethod
    def memory_allocated(device=None) -> int:
        return 0
    
    @staticmethod
    def max_memory_allocated(device=None) -> int:
        return 0
    
    @staticmethod
    def memory_stats(device=None) -> dict:
        return {}


# Mock nn module
class MockNN:
    """Mock torch.nn module."""
    
    class Module:
        """Mock neural network module."""
        
        def __init__(self):
            pass
        
        def eval(self):
            return self
        
        def train(self):
            return self
        
        def half(self):
            return self
        
        def cuda(self):
            return self
        
        def to(self, device):
            return self


# Create mock torch module
class MockTorch:
    """Mock torch module for testing."""
    
    # Tensor creation functions
    zeros = zeros
    ones = ones
    randn = randn
    rand = rand
    stack = stack
    linspace = linspace
    tensor = tensor
    
    # Math functions
    sin = sin
    cos = cos
    
    # Mock submodules
    cuda = MockCuda()
    nn = MockNN()
    
    # Mock attributes
    float16 = "float16"
    float32 = "float32"
    
    @staticmethod
    def compile(model, mode="reduce-overhead"):
        """Mock torch.compile (PyTorch 2.0+)."""
        return model  # Just return the model unchanged


# Mock F (functional)
class MockF:
    """Mock torch.nn.functional module."""
    
    @staticmethod
    def relu(x):
        if isinstance(x, MockTensor):
            return MockTensor(np.maximum(0, x.data))
        return MockTensor(np.maximum(0, np.array(x)))


# Add F to nn
MockTorch.nn.functional = MockF()
setattr(MockTorch.nn, 'F', MockF())

# Export the mock torch as 'torch'
torch = MockTorch()