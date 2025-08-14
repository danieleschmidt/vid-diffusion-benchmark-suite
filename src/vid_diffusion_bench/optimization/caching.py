"""Caching system for models, results, and intermediate data."""

import os
import pickle
import hashlib
import json
import time
import threading
from typing import Any, Dict, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import OrderedDict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

import torch
import numpy as np

from ..monitoring.logging import get_structured_logger
from ..exceptions import VidBenchError

logger = get_structured_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LRUCache:
    """In-memory LRU cache with size limits."""
    
    def __init__(self, max_size_bytes: int = 1024 * 1024 * 1024):  # 1GB default
        self.max_size_bytes = max_size_bytes
        self.current_size = 0
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._metadata: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
                
            # Check expiration
            metadata = self._metadata[key]
            if metadata.is_expired():
                self._remove(key)
                return None
                
            # Update access info
            metadata.last_accessed = datetime.utcnow()
            metadata.access_count += 1
            
            # Move to end (most recently used)
            value = self._cache[key]
            del self._cache[key]
            self._cache[key] = value
            
            return value
            
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put item in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove(key)
                
            # Check if item is too large
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Cache item too large: {size_bytes} bytes")
                return False
                
            # Make space if needed
            while (self.current_size + size_bytes) > self.max_size_bytes and self._cache:
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)
                
            # Add new entry
            self._cache[key] = value
            self._metadata[key] = CacheEntry(
                key=key,
                size_bytes=size_bytes,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                ttl_seconds=ttl_seconds
            )
            self.current_size += size_bytes
            
            return True
            
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            return self._remove(key)
            
    def _remove(self, key: str) -> bool:
        """Internal remove method."""
        if key not in self._cache:
            return False
            
        metadata = self._metadata[key]
        self.current_size -= metadata.size_bytes
        
        del self._cache[key]
        del self._metadata[key]
        
        return True
        
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
            self.current_size = 0
            
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, metadata in self._metadata.items()
                if metadata.is_expired()
            ]
            
            for key in expired_keys:
                self._remove(key)
                
            return len(expired_keys)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "size_bytes": self.current_size,
                "size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": self.current_size / self.max_size_bytes if self.max_size_bytes > 0 else 0
            }
            
    @staticmethod
    def _calculate_size(obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (str, bytes)):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(LRUCache._calculate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(LRUCache._calculate_size(k) + LRUCache._calculate_size(v) 
                      for k, v in obj.items())
        else:
            # Rough estimate for other objects
            try:
                return len(pickle.dumps(obj))
            except:
                return 1024  # Default estimate


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._metadata_file = self.cache_dir / "metadata.json"
        self._metadata: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._load_metadata()
        
    def _load_metadata(self):
        """Load cache metadata from disk."""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, 'r') as f:
                    self._metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            self._metadata = {}
            
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 chars for directory structure
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        subdir = key_hash[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{key_hash}.cache"
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        with self._lock:
            if key not in self._metadata:
                return None
                
            metadata = self._metadata[key]
            
            # Check expiration
            if metadata.get('ttl_seconds'):
                created_at = datetime.fromisoformat(metadata['created_at'])
                if (datetime.utcnow() - created_at).total_seconds() > metadata['ttl_seconds']:
                    self.remove(key)
                    return None
                    
            # Load from disk
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                # Metadata exists but file doesn't - clean up
                del self._metadata[key]
                self._save_metadata()
                return None
                
            try:
                with open(cache_path, 'rb') as f:
        # SECURITY: pickle.loads() can execute arbitrary code. Only use with trusted data.
                    value = pickle.load(f)
                    
                # Update access info
                metadata['last_accessed'] = datetime.utcnow().isoformat()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                self._save_metadata()
                
                return value
                
            except Exception as e:
                logger.error(f"Failed to load cache entry {key}: {e}")
                self.remove(key)
                return None
                
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put item in disk cache."""
        with self._lock:
            try:
                # Serialize to bytes
                data = pickle.dumps(value)
                size_bytes = len(data)
                
                # Check size limits
                if size_bytes > self.max_size_bytes:
                    logger.warning(f"Cache item too large: {size_bytes} bytes")
                    return False
                    
                # Make space if needed
                self._cleanup_space(size_bytes)
                
                # Write to disk
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'wb') as f:
                    f.write(data)
                    
                # Update metadata
                self._metadata[key] = {
                    'key': key,
                    'size_bytes': size_bytes,
                    'created_at': datetime.utcnow().isoformat(),
                    'last_accessed': datetime.utcnow().isoformat(),
                    'access_count': 1,
                    'ttl_seconds': ttl_seconds
                }
                self._save_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache item {key}: {e}")
                return False
                
    def remove(self, key: str) -> bool:
        """Remove item from disk cache."""
        with self._lock:
            if key not in self._metadata:
                return False
                
            # Remove file
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                
            # Remove metadata
            del self._metadata[key]
            self._save_metadata()
            
            return True
            
    def _cleanup_space(self, needed_bytes: int):
        """Clean up space for new entry."""
        current_size = sum(meta['size_bytes'] for meta in self._metadata.values())
        
        while (current_size + needed_bytes) > self.max_size_bytes and self._metadata:
            # Find least recently used entry
            lru_key = min(
                self._metadata.keys(),
                key=lambda k: self._metadata[k]['last_accessed']
            )
            
            removed_size = self._metadata[lru_key]['size_bytes']
            self.remove(lru_key)
            current_size -= removed_size
            
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired_keys = []
            now = datetime.utcnow()
            
            for key, metadata in self._metadata.items():
                if metadata.get('ttl_seconds'):
                    created_at = datetime.fromisoformat(metadata['created_at'])
                    if (now - created_at).total_seconds() > metadata['ttl_seconds']:
                        expired_keys.append(key)
                        
            for key in expired_keys:
                self.remove(key)
                
            return len(expired_keys)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(meta['size_bytes'] for meta in self._metadata.values())
            return {
                "entries": len(self._metadata),
                "size_bytes": total_size,
                "size_mb": total_size / (1024 * 1024),
                "max_size_gb": self.max_size_bytes / (1024 * 1024 * 1024),
                "utilization": total_size / self.max_size_bytes if self.max_size_bytes > 0 else 0
            }


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "vidbench:"):
        if not REDIS_AVAILABLE:
            raise VidBenchError("Redis not available")
            
        self.client = redis.from_url(redis_url)
        self.prefix = prefix
        
        # Test connection
        try:
            self.client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise VidBenchError(f"Redis connection failed: {e}")
            
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache."""
        try:
            data = self.client.get(self._make_key(key))
            if data is None:
                return None
        # SECURITY: pickle.loads() can execute arbitrary code. Only use with trusted data.
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get failed for {key}: {e}")
            return None
            
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put item in Redis cache."""
        try:
            data = pickle.dumps(value)
            redis_key = self._make_key(key)
            
            if ttl_seconds:
                self.client.setex(redis_key, ttl_seconds, data)
            else:
                self.client.set(redis_key, data)
                
            return True
        except Exception as e:
            logger.error(f"Redis put failed for {key}: {e}")
            return False
            
    def remove(self, key: str) -> bool:
        """Remove item from Redis cache."""
        try:
            result = self.client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis remove failed for {key}: {e}")
            return False
            
    def clear(self, pattern: str = "*"):
        """Clear cache entries matching pattern."""
        try:
            keys = self.client.keys(f"{self.prefix}{pattern}")
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        try:
            info = self.client.info()
            keys_count = len(self.client.keys(f"{self.prefix}*"))
            
            return {
                "entries": keys_count,
                "memory_used_mb": info.get("used_memory", 0) / (1024 * 1024),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            logger.error(f"Redis stats failed: {e}")
            return {}


class ModelCache:
    """Specialized cache for model instances and weights."""
    
    def __init__(self, cache_dir: str = "./cache/models"):
        self.memory_cache = LRUCache(max_size_bytes=2 * 1024 * 1024 * 1024)  # 2GB
        self.disk_cache = DiskCache(cache_dir, max_size_gb=50.0)  # 50GB
        self._loaded_models: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
    def get_model(self, model_name: str, model_config: Dict[str, Any]) -> Optional[Any]:
        """Get cached model instance."""
        cache_key = self._make_model_key(model_name, model_config)
        
        with self._lock:
            # Check if model is already loaded
            if cache_key in self._loaded_models:
                return self._loaded_models[cache_key]
                
            # Try memory cache
            model = self.memory_cache.get(cache_key)
            if model is not None:
                self._loaded_models[cache_key] = model
                return model
                
            # Try disk cache
            model = self.disk_cache.get(cache_key)
            if model is not None:
                # Load back to memory
                self.memory_cache.put(cache_key, model, ttl_seconds=3600)
                self._loaded_models[cache_key] = model
                return model
                
            return None
            
    def cache_model(self, model_name: str, model_config: Dict[str, Any], model: Any) -> bool:
        """Cache model instance."""
        cache_key = self._make_model_key(model_name, model_config)
        
        with self._lock:
            # Store in loaded models
            self._loaded_models[cache_key] = model
            
            # Cache in memory (short TTL)
            memory_success = self.memory_cache.put(cache_key, model, ttl_seconds=1800)
            
            # Cache on disk (longer TTL)
            disk_success = self.disk_cache.put(cache_key, model, ttl_seconds=86400)
            
            return memory_success or disk_success
            
    def evict_model(self, model_name: str, model_config: Dict[str, Any]) -> bool:
        """Evict model from all caches."""
        cache_key = self._make_model_key(model_name, model_config)
        
        with self._lock:
            # Remove from loaded models
            if cache_key in self._loaded_models:
                del self._loaded_models[cache_key]
                
            # Remove from caches
            memory_removed = self.memory_cache.remove(cache_key)
            disk_removed = self.disk_cache.remove(cache_key)
            
            return memory_removed or disk_removed
            
    def _make_model_key(self, model_name: str, model_config: Dict[str, Any]) -> str:
        """Create cache key for model."""
        config_str = json.dumps(model_config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return f"model:{model_name}:{config_hash}"
        
    def get_stats(self) -> Dict[str, Any]:
        """Get model cache statistics."""
        return {
            "loaded_models": len(self._loaded_models),
            "memory_cache": self.memory_cache.get_stats(),
            "disk_cache": self.disk_cache.get_stats()
        }


class ResultsCache:
    """Cache for benchmark results and computed metrics."""
    
    def __init__(self, cache_dir: str = "./cache/results"):
        self.disk_cache = DiskCache(cache_dir, max_size_gb=5.0)
        
        # Try to initialize Redis cache
        self.redis_cache = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(redis_url, "vidbench:results:")
            except Exception as e:
                logger.warning(f"Redis cache unavailable: {e}")
                
    def get_result(self, prompt: str, model_name: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get cached benchmark result."""
        cache_key = self._make_result_key(prompt, model_name, config)
        
        # Try Redis first (fastest)
        if self.redis_cache:
            result = self.redis_cache.get(cache_key)
            if result is not None:
                return result
                
        # Try disk cache
        return self.disk_cache.get(cache_key)
        
    def cache_result(self, prompt: str, model_name: str, config: Dict[str, Any], result: Any) -> bool:
        """Cache benchmark result."""
        cache_key = self._make_result_key(prompt, model_name, config)
        
        # Cache in Redis (short TTL)
        redis_success = False
        if self.redis_cache:
            redis_success = self.redis_cache.put(cache_key, result, ttl_seconds=3600)
            
        # Cache on disk (longer TTL)
        disk_success = self.disk_cache.put(cache_key, result, ttl_seconds=604800)  # 1 week
        
        return redis_success or disk_success
        
    def _make_result_key(self, prompt: str, model_name: str, config: Dict[str, Any]) -> str:
        """Create cache key for result."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return f"result:{model_name}:{prompt_hash}:{config_hash}"
        
    def get_stats(self) -> Dict[str, Any]:
        """Get results cache statistics."""
        stats = {"disk_cache": self.disk_cache.get_stats()}
        if self.redis_cache:
            stats["redis_cache"] = self.redis_cache.get_stats()
        return stats


class CacheManager:
    """Central cache management."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_cache = ModelCache(str(self.cache_dir / "models"))
        self.results_cache = ResultsCache(str(self.cache_dir / "results"))
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(300)  # 5 minutes
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        
    def cleanup_expired(self):
        """Clean up expired cache entries."""
        try:
            # Cleanup memory caches
            self.model_cache.memory_cache.cleanup_expired()
            
            # Cleanup disk caches
            model_expired = self.model_cache.disk_cache.cleanup_expired()
            results_expired = self.results_cache.disk_cache.cleanup_expired()
            
            if model_expired > 0 or results_expired > 0:
                logger.info(f"Cleaned up {model_expired} model and {results_expired} result cache entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "model_cache": self.model_cache.get_stats(),
            "results_cache": self.results_cache.get_stats()
        }
        
    def clear_all(self):
        """Clear all caches."""
        try:
            self.model_cache.memory_cache.clear()
            self.model_cache.disk_cache._metadata.clear()
            self.results_cache.disk_cache._metadata.clear()
            
            if self.results_cache.redis_cache:
                self.results_cache.redis_cache.clear()
                
            logger.info("All caches cleared")
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")


# Global cache manager
cache_manager = CacheManager()