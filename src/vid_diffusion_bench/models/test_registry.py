"""Test-only model registry without heavy dependencies."""

from typing import Dict, List, Type

# Simple registry for testing
_TEST_MODEL_REGISTRY: Dict[str, Dict] = {
    "mock-fast": {
        "name": "mock-fast",
        "type": "mock",
        "requirements": {
            "vram_gb": 2.0,
            "precision": "fp32",
            "max_frames": 64
        }
    },
    "mock-high-quality": {
        "name": "mock-high-quality", 
        "type": "mock",
        "requirements": {
            "vram_gb": 8.0,
            "precision": "fp16",
            "max_frames": 32
        }
    },
    "mock-memory-intensive": {
        "name": "mock-memory-intensive",
        "type": "mock", 
        "requirements": {
            "vram_gb": 32.0,
            "precision": "fp16",
            "max_frames": 16
        }
    },
    "mock-efficient": {
        "name": "mock-efficient",
        "type": "mock",
        "requirements": {
            "vram_gb": 1.0,
            "precision": "fp32",
            "max_frames": 128
        }
    },
    "cogvideo-5b": {
        "name": "cogvideo-5b",
        "type": "real",
        "requirements": {
            "vram_gb": 24.0,
            "precision": "fp16", 
            "max_frames": 49
        }
    },
    "modelscope-t2v": {
        "name": "modelscope-t2v",
        "type": "real",
        "requirements": {
            "vram_gb": 12.0,
            "precision": "fp16",
            "max_frames": 16
        }
    },
    "zeroscope-v2": {
        "name": "zeroscope-v2",
        "type": "real",
        "requirements": {
            "vram_gb": 8.0,
            "precision": "fp16",
            "max_frames": 24
        }
    },
    "animatediff-v2": {
        "name": "animatediff-v2",
        "type": "real",
        "requirements": {
            "vram_gb": 16.0,
            "precision": "fp16",
            "max_frames": 16
        }
    },
    "text2video-zero": {
        "name": "text2video-zero",
        "type": "real",
        "requirements": {
            "vram_gb": 6.0,
            "precision": "fp16",
            "max_frames": 8
        }
    },
    "pika-lumiere-xl": {
        "name": "pika-lumiere-xl",
        "type": "proprietary",
        "requirements": {
            "vram_gb": 40.0,
            "precision": "fp16",
            "max_frames": 240
        }
    },
    "dreamvideo-v3": {
        "name": "dreamvideo-v3",
        "type": "sota",
        "requirements": {
            "vram_gb": 24.0,
            "precision": "fp16",
            "max_frames": 128
        }
    },
    "svd-xt-1.1": {
        "name": "svd-xt-1.1",
        "type": "real",
        "requirements": {
            "vram_gb": 20.0,
            "precision": "fp16",
            "max_frames": 25
        }
    },
    "svd-base": {
        "name": "svd-base",
        "type": "real",
        "requirements": {
            "vram_gb": 16.0,
            "precision": "fp16",
            "max_frames": 14
        }
    }
}


def test_list_models() -> List[str]:
    """List all test models."""
    return list(_TEST_MODEL_REGISTRY.keys())


def test_get_model(name: str) -> Dict:
    """Get test model info."""
    if name not in _TEST_MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found. Available: {test_list_models()}")
    return _TEST_MODEL_REGISTRY[name]


def test_model_count() -> int:
    """Get total model count."""
    return len(_TEST_MODEL_REGISTRY)


def test_model_types() -> List[str]:
    """Get unique model types."""
    return list(set(model["type"] for model in _TEST_MODEL_REGISTRY.values()))