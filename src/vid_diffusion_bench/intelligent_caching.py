"""Intelligent caching system with ML-driven optimization.

Advanced caching strategies that learn from usage patterns and automatically
optimize for different workloads and access patterns.
"""

import asyncio
import time
import json
import hashlib
import threading
import pickle
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from pathlib import Path
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    write_time: float = 0.0
    read_time: float = 0.0
    memory_usage: int = 0
    last_access: float = field(default_factory=time.time)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    @property
    def avg_read_time(self) -> float:
        """Average read time per operation."""
        return self.read_time / max(1, self.hits)
        
    @property
    def avg_write_time(self) -> float:
        """Average write time per operation."""
        return self.write_time / max(1, self.hits + self.misses)


@dataclass
class CacheEntry:
    """Smart cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    access_pattern: List[float] = field(default_factory=list)
    compute_cost: float = 0.0
    size: int = 0
    priority: float = 1.0
    tags: List[str] = field(default_factory=list)
    
    def access(self):
        """Record cache access."""
        now = time.time()
        self.last_accessed = now
        self.access_count += 1
        self.access_pattern.append(now)
        
        # Keep only recent access pattern (last 100 accesses)
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]
            
    @property
    def age(self) -> float:
        """Age of cache entry in seconds."""
        return time.time() - self.created_at
        
    @property
    def recency(self) -> float:
        """Time since last access."""
        return time.time() - self.last_accessed
        
    @property
    def frequency_score(self) -> float:
        """Frequency-based score for LFU eviction."""
        if not self.access_pattern:
            return 0.0
            
        # Weight recent accesses more heavily
        now = time.time()
        weighted_accesses = sum(
            1.0 / (1.0 + (now - access_time) / 3600)  # Hour-based decay
            for access_time in self.access_pattern
        )
        return weighted_accesses
        
    @property
    def value_score(self) -> float:
        """Value-based score considering cost and utility."""
        base_score = self.compute_cost * self.access_count / max(1, self.recency)
        return base_score * self.priority


class IntelligentCache:
    """ML-driven adaptive cache with multiple eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: int = 1024 * 1024 * 1024,  # 1GB
        ttl: Optional[float] = None,
        eviction_strategy: str = "adaptive"
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.ttl = ttl
        self.eviction_strategy = eviction_strategy
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._metrics = CacheMetrics()
        self._lock = threading.RLock()
        
        # Adaptive strategy state
        self._strategy_performance: Dict[str, float] = defaultdict(float)
        self._current_strategy = "lru"
        self._strategy_switch_threshold = 0.1
        self._last_strategy_evaluation = time.time()
        
        # Access pattern learning
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._prediction_cache: Dict[str, float] = {}
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent prefetching."""
        start_time = time.time()
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if self.ttl and entry.age > self.ttl:
                    del self._cache[key]
                    self._metrics.misses += 1
                    return default
                    
                # Record access and move to end (for LRU)
                entry.access()
                self._cache.move_to_end(key)
                
                self._metrics.hits += 1
                self._metrics.read_time += time.time() - start_time
                
                # Trigger predictive prefetching
                self._maybe_prefetch(key)
                
                return entry.value
            else:
                self._metrics.misses += 1
                return default
                
    def set(
        self,
        key: str,
        value: Any,
        compute_cost: float = 1.0,
        priority: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> None:
        """Set value in cache with intelligent metadata."""
        start_time = time.time()
        
        with self._lock:
            # Calculate value size
            try:
                size = len(pickle.dumps(value))
            except:
                size = 1024  # Fallback estimate
                
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                compute_cost=compute_cost,
                size=size,
                priority=priority,
                tags=tags or []
            )
            
            # Update cache
            if key in self._cache:
                old_entry = self._cache[key]
                self._metrics.memory_usage -= old_entry.size
                
            self._cache[key] = entry
            self._metrics.memory_usage += size
            self._metrics.write_time += time.time() - start_time
            
            # Trigger eviction if needed
            self._enforce_limits()
            
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._metrics.memory_usage -= entry.size
                return True
            return False
            
    def clear(self, tags: Optional[List[str]] = None) -> int:
        """Clear cache entries, optionally by tags."""
        with self._lock:
            if tags is None:
                count = len(self._cache)
                self._cache.clear()
                self._metrics.memory_usage = 0
                return count
            else:
                keys_to_remove = [
                    key for key, entry in self._cache.items()
                    if any(tag in entry.tags for tag in tags)
                ]
                for key in keys_to_remove:
                    entry = self._cache.pop(key)
                    self._metrics.memory_usage -= entry.size
                return len(keys_to_remove)
                
    def _enforce_limits(self) -> None:
        """Enforce cache size and memory limits with adaptive eviction."""
        while (len(self._cache) > self.max_size or 
               self._metrics.memory_usage > self.max_memory):
            
            if self.eviction_strategy == "adaptive":
                self._adaptive_eviction()
            else:
                self._evict_by_strategy(self.eviction_strategy)
                
    def _adaptive_eviction(self) -> None:
        """Dynamically choose best eviction strategy."""
        # Evaluate strategy performance periodically
        if time.time() - self._last_strategy_evaluation > 300:  # 5 minutes
            self._evaluate_strategies()
            self._last_strategy_evaluation = time.time()
            
        self._evict_by_strategy(self._current_strategy)
        
    def _evaluate_strategies(self) -> None:
        """Evaluate and select best eviction strategy."""
        strategies = ["lru", "lfu", "value_based", "hybrid"]
        
        # Simulate each strategy and score
        for strategy in strategies:
            score = self._simulate_strategy_performance(strategy)
            self._strategy_performance[strategy] = score
            
        # Select best performing strategy
        best_strategy = max(
            self._strategy_performance.items(),
            key=lambda x: x[1]
        )[0]
        
        # Switch if significantly better
        current_score = self._strategy_performance.get(self._current_strategy, 0)
        best_score = self._strategy_performance[best_strategy]
        
        if best_score > current_score * (1 + self._strategy_switch_threshold):
            logger.info(f"Switching eviction strategy from {self._current_strategy} to {best_strategy}")
            self._current_strategy = best_strategy
            
    def _simulate_strategy_performance(self, strategy: str) -> float:
        """Simulate strategy performance based on access patterns."""
        if not self._cache:
            return 0.0
            
        # Score based on predicted future value
        total_score = 0.0
        for entry in self._cache.values():
            if strategy == "lru":
                score = 1.0 / (1.0 + entry.recency)
            elif strategy == "lfu":
                score = entry.frequency_score
            elif strategy == "value_based":
                score = entry.value_score
            elif strategy == "hybrid":
                score = (entry.frequency_score + entry.value_score) / 2
            else:
                score = 1.0
                
            total_score += score
            
        return total_score / len(self._cache)
        
    def _evict_by_strategy(self, strategy: str) -> None:
        """Evict entry using specified strategy."""
        if not self._cache:
            return
            
        if strategy == "lru":
            # Remove least recently used (first in OrderedDict)
            key, entry = self._cache.popitem(last=False)
        elif strategy == "lfu":
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].frequency_score)
            entry = self._cache.pop(key)
        elif strategy == "value_based":
            # Remove lowest value entry
            key = min(self._cache.keys(), key=lambda k: self._cache[k].value_score)
            entry = self._cache.pop(key)
        elif strategy == "hybrid":
            # Combine frequency and value scoring
            def hybrid_score(k):
                e = self._cache[k]
                return (e.frequency_score + e.value_score) / 2
            key = min(self._cache.keys(), key=hybrid_score)
            entry = self._cache.pop(key)
        else:
            # Default to LRU
            key, entry = self._cache.popitem(last=False)
            
        self._metrics.memory_usage -= entry.size
        self._metrics.evictions += 1
        
        logger.debug(f"Evicted cache entry: {key} (strategy: {strategy})")
        
    def _maybe_prefetch(self, accessed_key: str) -> None:
        """Predictive prefetching based on access patterns."""
        # Record access pattern
        self._access_patterns[accessed_key].append(time.time())
        
        # Limit pattern history
        if len(self._access_patterns[accessed_key]) > 100:
            self._access_patterns[accessed_key] = self._access_patterns[accessed_key][-100:]
            
        # Find correlated keys for prefetching
        correlated_keys = self._find_correlated_keys(accessed_key)
        
        for key, correlation in correlated_keys[:3]:  # Top 3 correlated
            if key not in self._cache and correlation > 0.7:
                logger.debug(f"Prefetch candidate: {key} (correlation: {correlation:.2f})")
                # Note: Actual prefetching would require external data source
                
    def _find_correlated_keys(self, key: str) -> List[Tuple[str, float]]:
        """Find keys with correlated access patterns."""
        if key not in self._access_patterns:
            return []
            
        key_pattern = self._access_patterns[key]
        if len(key_pattern) < 5:  # Need sufficient data
            return []
            
        correlations = []
        for other_key, other_pattern in self._access_patterns.items():
            if other_key == key or len(other_pattern) < 5:
                continue
                
            # Calculate time-based correlation
            correlation = self._calculate_pattern_correlation(key_pattern, other_pattern)
            if correlation > 0.3:  # Threshold for meaningful correlation
                correlations.append((other_key, correlation))
                
        return sorted(correlations, key=lambda x: x[1], reverse=True)
        
    def _calculate_pattern_correlation(self, pattern1: List[float], pattern2: List[float]) -> float:
        """Calculate correlation between access patterns."""
        if len(pattern1) < 2 or len(pattern2) < 2:
            return 0.0
            
        # Convert to intervals between accesses
        intervals1 = [pattern1[i+1] - pattern1[i] for i in range(len(pattern1)-1)]
        intervals2 = [pattern2[i+1] - pattern2[i] for i in range(len(pattern2)-1)]
        
        if not intervals1 or not intervals2:
            return 0.0
            
        # Calculate Pearson correlation coefficient
        try:
            return statistics.correlation(intervals1[-10:], intervals2[-10:])
        except:
            return 0.0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage': self._metrics.memory_usage,
                'max_memory': self.max_memory,
                'hit_rate': self._metrics.hit_rate,
                'hits': self._metrics.hits,
                'misses': self._metrics.misses,
                'evictions': self._metrics.evictions,
                'avg_read_time': self._metrics.avg_read_time,
                'avg_write_time': self._metrics.avg_write_time,
                'current_strategy': self._current_strategy,
                'strategy_performance': dict(self._strategy_performance),
                'access_patterns_tracked': len(self._access_patterns)
            }
            
    def optimize(self) -> Dict[str, Any]:
        """Run optimization routines and return recommendations."""
        with self._lock:
            optimizations = []
            
            # Analyze hit rate
            if self._metrics.hit_rate < 0.8:
                optimizations.append({
                    'type': 'hit_rate',
                    'message': f'Low hit rate ({self._metrics.hit_rate:.2%}). Consider increasing cache size.',
                    'recommendation': 'increase_size'
                })
                
            # Analyze memory usage
            memory_usage_ratio = self._metrics.memory_usage / self.max_memory
            if memory_usage_ratio > 0.9:
                optimizations.append({
                    'type': 'memory',
                    'message': f'High memory usage ({memory_usage_ratio:.1%}). Consider TTL or priority tuning.',
                    'recommendation': 'reduce_memory'
                })
                
            # Analyze eviction rate
            if self._metrics.evictions > self._metrics.hits * 0.1:
                optimizations.append({
                    'type': 'evictions',
                    'message': 'High eviction rate detected. Cache may be too small for workload.',
                    'recommendation': 'increase_size_or_optimize_data'
                })
                
            # Strategy performance analysis
            strategy_scores = self._strategy_performance
            if strategy_scores:
                best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
                if best_strategy[0] != self._current_strategy:
                    optimizations.append({
                        'type': 'strategy',
                        'message': f'Strategy {best_strategy[0]} may perform better than {self._current_strategy}.',
                        'recommendation': f'switch_to_{best_strategy[0]}'
                    })
                    
            return {
                'optimizations': optimizations,
                'stats': self.get_stats(),
                'recommendations_count': len(optimizations)
            }


class DistributedCache:
    """Distributed cache with intelligent partitioning."""
    
    def __init__(self, node_id: str, nodes: List[str]):
        self.node_id = node_id
        self.nodes = nodes
        self.local_cache = IntelligentCache()
        self._node_weights: Dict[str, float] = {node: 1.0 for node in nodes}
        
    def _get_node_for_key(self, key: str) -> str:
        """Determine which node should handle this key."""
        # Consistent hashing with weighted nodes
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Apply weights to hash ring
        weighted_nodes = []
        for node in self.nodes:
            weight = self._node_weights[node]
            for i in range(int(weight * 100)):  # Scale weight
                node_hash = int(hashlib.md5(f"{node}:{i}".encode()).hexdigest(), 16)
                weighted_nodes.append((node_hash, node))
                
        weighted_nodes.sort()
        
        # Find next node in ring
        for node_hash, node in weighted_nodes:
            if hash_value <= node_hash:
                return node
                
        return weighted_nodes[0][1]  # Wrap around
        
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from distributed cache."""
        target_node = self._get_node_for_key(key)
        
        if target_node == self.node_id:
            return self.local_cache.get(key, default)
        else:
            # Remote cache access (would implement network call)
            return await self._remote_get(target_node, key, default)
            
    async def set(self, key: str, value: Any, **kwargs) -> None:
        """Set value in distributed cache."""
        target_node = self._get_node_for_key(key)
        
        if target_node == self.node_id:
            self.local_cache.set(key, value, **kwargs)
        else:
            await self._remote_set(target_node, key, value, **kwargs)
            
    async def _remote_get(self, node: str, key: str, default: Any) -> Any:
        """Get value from remote node (placeholder for network implementation)."""
        # Would implement actual network call
        logger.debug(f"Remote cache get: {node}/{key}")
        return default
        
    async def _remote_set(self, node: str, key: str, value: Any, **kwargs) -> None:
        """Set value on remote node (placeholder for network implementation)."""
        # Would implement actual network call
        logger.debug(f"Remote cache set: {node}/{key}")
        pass


# Global cache instances
model_cache = IntelligentCache(max_size=100, max_memory=2*1024*1024*1024)  # 2GB for models
result_cache = IntelligentCache(max_size=10000, max_memory=512*1024*1024)  # 512MB for results
metadata_cache = IntelligentCache(max_size=50000, ttl=3600)  # 1 hour TTL for metadata