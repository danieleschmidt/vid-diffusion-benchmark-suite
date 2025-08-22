"""Federated Benchmark Learning for Video Diffusion Models.

This module implements a privacy-preserving federated learning system that enables
multiple institutions to collaboratively improve video diffusion benchmarks without
sharing sensitive data or models.
"""

import time
import logging
import asyncio
import json
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
import base64

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn, F
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class FederatedRole(Enum):
    """Roles in the federated learning system."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    MINIMAL = "minimal"  # Basic aggregation
    STANDARD = "standard"  # Differential privacy
    HIGH = "high"  # Secure aggregation
    MAXIMUM = "maximum"  # Homomorphic encryption


@dataclass
class FederatedParticipant:
    """Information about a federated learning participant."""
    participant_id: str
    institution_name: str
    role: FederatedRole
    public_key: str
    capabilities: Dict[str, Any]
    contribution_weight: float
    privacy_level: PrivacyLevel
    last_seen: float
    trust_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkContribution:
    """A contribution to the federated benchmark."""
    contribution_id: str
    participant_id: str
    timestamp: float
    contribution_type: str  # "metrics", "model_weights", "evaluation_data"
    data_hash: str
    metadata: Dict[str, Any]
    privacy_preserved: bool
    validation_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FederatedModel:
    """Federated model state."""
    model_id: str
    version: int
    global_weights: Dict[str, Any]
    participant_contributions: List[str]
    aggregation_method: str
    performance_metrics: Dict[str, float]
    last_updated: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacyBudget:
    """Differential privacy budget tracking."""
    participant_id: str
    epsilon_total: float
    epsilon_used: float
    delta: float
    queries_made: int
    last_reset: float
    
    def has_budget(self, epsilon_required: float) -> bool:
        return (self.epsilon_used + epsilon_required) <= self.epsilon_total
    
    def consume_budget(self, epsilon_used: float):
        self.epsilon_used += epsilon_used
        self.queries_made += 1


class SecureAggregator:
    """Secure aggregation for federated learning."""
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.STANDARD):
        self.privacy_level = privacy_level
        self.noise_multiplier = self._get_noise_multiplier()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _get_noise_multiplier(self) -> float:
        """Get noise multiplier based on privacy level."""
        multipliers = {
            PrivacyLevel.MINIMAL: 0.0,
            PrivacyLevel.STANDARD: 1.0,
            PrivacyLevel.HIGH: 2.0,
            PrivacyLevel.MAXIMUM: 3.0
        }
        return multipliers.get(self.privacy_level, 1.0)
    
    def aggregate_metrics(self, 
                         contributions: List[Dict[str, float]],
                         participant_weights: List[float]) -> Dict[str, float]:
        """
        Aggregate metrics from multiple participants with privacy preservation.
        
        Args:
            contributions: List of metric dictionaries from participants
            participant_weights: Weights for each participant
        
        Returns:
            Aggregated metrics with privacy preservation
        """
        if not contributions:
            return {}
        
        # Normalize weights
        total_weight = sum(participant_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in participant_weights]
        else:
            normalized_weights = [1.0 / len(contributions)] * len(contributions)
        
        # Find common metrics
        all_metrics = set()
        for contrib in contributions:
            all_metrics.update(contrib.keys())
        
        aggregated = {}
        
        for metric in all_metrics:
            values = []
            weights = []
            
            for i, contrib in enumerate(contributions):
                if metric in contrib:
                    values.append(contrib[metric])
                    weights.append(normalized_weights[i])
            
            if values:
                # Weighted average
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                weight_sum = sum(weights)
                
                if weight_sum > 0:
                    aggregated_value = weighted_sum / weight_sum
                    
                    # Add differential privacy noise
                    if self.privacy_level != PrivacyLevel.MINIMAL:
                        noise = np.random.laplace(0, self.noise_multiplier * 0.1)
                        aggregated_value += noise
                    
                    aggregated[metric] = float(aggregated_value)
        
        self.logger.info(f"Aggregated {len(all_metrics)} metrics from {len(contributions)} participants")
        return aggregated
    
    def aggregate_model_weights(self, 
                              weight_contributions: List[Dict[str, Any]],
                              participant_weights: List[float]) -> Dict[str, Any]:
        """
        Aggregate model weights with privacy preservation.
        
        Args:
            weight_contributions: List of model weight dictionaries
            participant_weights: Weights for each participant
        
        Returns:
            Aggregated model weights
        """
        if not weight_contributions:
            return {}
        
        # Normalize participant weights
        total_weight = sum(participant_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in participant_weights]
        else:
            normalized_weights = [1.0 / len(weight_contributions)] * len(weight_contributions)
        
        # Find common weight keys
        all_keys = set()
        for contrib in weight_contributions:
            all_keys.update(contrib.keys())
        
        aggregated_weights = {}
        
        for key in all_keys:
            weight_tensors = []
            weights = []
            
            for i, contrib in enumerate(weight_contributions):
                if key in contrib:
                    weight_tensors.append(contrib[key])
                    weights.append(normalized_weights[i])
            
            if weight_tensors:
                # Weighted average of tensors
                if TORCH_AVAILABLE and all(isinstance(w, torch.Tensor) for w in weight_tensors):
                    # PyTorch tensor aggregation
                    weighted_sum = sum(w * weight for w, weight in zip(weight_tensors, weights))
                    aggregated_weights[key] = weighted_sum / sum(weights)
                    
                    # Add privacy noise to weights
                    if self.privacy_level != PrivacyLevel.MINIMAL:
                        noise_scale = self.noise_multiplier * 0.01
                        noise = torch.normal(0, noise_scale, size=aggregated_weights[key].shape)
                        aggregated_weights[key] += noise
                
                elif all(isinstance(w, (list, np.ndarray)) for w in weight_tensors):
                    # NumPy array aggregation
                    arrays = [np.array(w) for w in weight_tensors]
                    weighted_sum = sum(arr * weight for arr, weight in zip(arrays, weights))
                    aggregated_weights[key] = weighted_sum / sum(weights)
                    
                    # Add privacy noise
                    if self.privacy_level != PrivacyLevel.MINIMAL:
                        noise_scale = self.noise_multiplier * 0.01
                        noise = np.random.normal(0, noise_scale, aggregated_weights[key].shape)
                        aggregated_weights[key] += noise
                
                else:
                    # Scalar aggregation
                    weighted_sum = sum(float(w) * weight for w, weight in zip(weight_tensors, weights))
                    aggregated_weights[key] = weighted_sum / sum(weights)
                    
                    # Add privacy noise
                    if self.privacy_level != PrivacyLevel.MINIMAL:
                        noise = np.random.laplace(0, self.noise_multiplier * 0.01)
                        aggregated_weights[key] += noise
        
        self.logger.info(f"Aggregated {len(all_keys)} weight parameters from {len(weight_contributions)} participants")
        return aggregated_weights


class PrivacyManager:
    """Manage privacy budgets and differential privacy."""
    
    def __init__(self, default_epsilon: float = 10.0, default_delta: float = 1e-5):
        self.default_epsilon = default_epsilon
        self.default_delta = default_delta
        self.privacy_budgets: Dict[str, PrivacyBudget] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def initialize_participant_budget(self, participant_id: str, 
                                    epsilon: Optional[float] = None,
                                    delta: Optional[float] = None):
        """Initialize privacy budget for a participant."""
        budget = PrivacyBudget(
            participant_id=participant_id,
            epsilon_total=epsilon or self.default_epsilon,
            epsilon_used=0.0,
            delta=delta or self.default_delta,
            queries_made=0,
            last_reset=time.time()
        )
        
        self.privacy_budgets[participant_id] = budget
        self.logger.info(f"Initialized privacy budget for {participant_id}: ε={budget.epsilon_total}")
    
    def check_privacy_budget(self, participant_id: str, epsilon_required: float) -> bool:
        """Check if participant has sufficient privacy budget."""
        if participant_id not in self.privacy_budgets:
            self.initialize_participant_budget(participant_id)
        
        budget = self.privacy_budgets[participant_id]
        return budget.has_budget(epsilon_required)
    
    def consume_privacy_budget(self, participant_id: str, epsilon_used: float):
        """Consume privacy budget for a participant."""
        if participant_id not in self.privacy_budgets:
            self.initialize_participant_budget(participant_id)
        
        budget = self.privacy_budgets[participant_id]
        budget.consume_budget(epsilon_used)
        
        self.logger.debug(f"Privacy budget consumed for {participant_id}: "
                         f"{epsilon_used:.3f} (remaining: {budget.epsilon_total - budget.epsilon_used:.3f})")
    
    def reset_privacy_budget(self, participant_id: str):
        """Reset privacy budget for a participant."""
        if participant_id in self.privacy_budgets:
            budget = self.privacy_budgets[participant_id]
            budget.epsilon_used = 0.0
            budget.queries_made = 0
            budget.last_reset = time.time()
            
            self.logger.info(f"Privacy budget reset for {participant_id}")
    
    def get_privacy_status(self, participant_id: str) -> Dict[str, Any]:
        """Get privacy budget status for a participant."""
        if participant_id not in self.privacy_budgets:
            return {"error": "No privacy budget found"}
        
        budget = self.privacy_budgets[participant_id]
        return {
            "epsilon_total": budget.epsilon_total,
            "epsilon_used": budget.epsilon_used,
            "epsilon_remaining": budget.epsilon_total - budget.epsilon_used,
            "delta": budget.delta,
            "queries_made": budget.queries_made,
            "budget_exhausted": budget.epsilon_used >= budget.epsilon_total
        }


class FederatedCoordinator:
    """Coordinator node for federated benchmark learning."""
    
    def __init__(self, coordinator_id: str, privacy_level: PrivacyLevel = PrivacyLevel.STANDARD):
        self.coordinator_id = coordinator_id
        self.privacy_level = privacy_level
        self.participants: Dict[str, FederatedParticipant] = {}
        self.contributions: List[BenchmarkContribution] = []
        self.federated_models: Dict[str, FederatedModel] = {}
        self.aggregator = SecureAggregator(privacy_level)
        self.privacy_manager = PrivacyManager()
        self.round_number = 0
        self.is_running = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def register_participant(self, participant: FederatedParticipant) -> bool:
        """Register a new participant in the federation."""
        try:
            # Validate participant
            if not self._validate_participant(participant):
                return False
            
            # Initialize privacy budget
            self.privacy_manager.initialize_participant_budget(participant.participant_id)
            
            # Add to participants
            self.participants[participant.participant_id] = participant
            
            self.logger.info(f"Registered participant: {participant.participant_id} "
                           f"({participant.institution_name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register participant {participant.participant_id}: {e}")
            return False
    
    def _validate_participant(self, participant: FederatedParticipant) -> bool:
        """Validate participant credentials and capabilities."""
        # Basic validation
        if not participant.participant_id or not participant.institution_name:
            return False
        
        # Check for duplicate IDs
        if participant.participant_id in self.participants:
            self.logger.warning(f"Participant {participant.participant_id} already registered")
            return False
        
        # Validate capabilities
        required_capabilities = ["model_evaluation", "metric_computation"]
        for capability in required_capabilities:
            if capability not in participant.capabilities:
                self.logger.warning(f"Participant {participant.participant_id} missing capability: {capability}")
                return False
        
        # Validate trust score
        if participant.trust_score < 0.5:
            self.logger.warning(f"Participant {participant.participant_id} has low trust score: {participant.trust_score}")
            return False
        
        return True
    
    async def start_federated_round(self, task_specification: Dict[str, Any]) -> str:
        """Start a new federated learning round."""
        self.round_number += 1
        round_id = f"round_{self.round_number}_{int(time.time())}"
        
        self.logger.info(f"Starting federated round {self.round_number}: {round_id}")
        
        # Prepare task for participants
        task = {
            "round_id": round_id,
            "round_number": self.round_number,
            "task_type": task_specification.get("task_type", "benchmark_evaluation"),
            "models_to_evaluate": task_specification.get("models", []),
            "metrics_to_compute": task_specification.get("metrics", []),
            "privacy_budget": task_specification.get("privacy_budget", 1.0),
            "deadline": time.time() + task_specification.get("timeout", 3600),  # 1 hour default
            "coordinator_id": self.coordinator_id
        }
        
        # Send task to all active participants
        active_participants = self._get_active_participants()
        
        if not active_participants:
            self.logger.warning("No active participants for federated round")
            return round_id
        
        # Simulate sending task to participants
        sent_tasks = []
        for participant_id in active_participants:
            participant = self.participants[participant_id]
            
            # Check privacy budget
            if not self.privacy_manager.check_privacy_budget(
                participant_id, task["privacy_budget"]
            ):
                self.logger.warning(f"Insufficient privacy budget for {participant_id}")
                continue
            
            # Send task (simulation)
            sent_tasks.append(self._send_task_to_participant(participant, task))
        
        self.logger.info(f"Sent federated task to {len(sent_tasks)} participants")
        return round_id
    
    def _get_active_participants(self) -> List[str]:
        """Get list of active participant IDs."""
        current_time = time.time()
        active_threshold = 300  # 5 minutes
        
        active_participants = []
        for participant_id, participant in self.participants.items():
            if current_time - participant.last_seen < active_threshold:
                active_participants.append(participant_id)
        
        return active_participants
    
    def _send_task_to_participant(self, participant: FederatedParticipant, task: Dict[str, Any]) -> bool:
        """Send task to a specific participant (simulation)."""
        # In practice, this would send over network
        self.logger.debug(f"Sending task to {participant.participant_id}")
        return True
    
    async def receive_contribution(self, contribution: BenchmarkContribution) -> bool:
        """Receive and validate a contribution from a participant."""
        try:
            # Validate contribution
            if not self._validate_contribution(contribution):
                return False
            
            # Check privacy budget
            privacy_budget_used = self._estimate_privacy_cost(contribution)
            if not self.privacy_manager.check_privacy_budget(
                contribution.participant_id, privacy_budget_used
            ):
                self.logger.warning(f"Privacy budget exceeded for {contribution.participant_id}")
                return False
            
            # Consume privacy budget
            self.privacy_manager.consume_privacy_budget(
                contribution.participant_id, privacy_budget_used
            )
            
            # Add contribution
            self.contributions.append(contribution)
            
            # Update participant last seen
            if contribution.participant_id in self.participants:
                self.participants[contribution.participant_id].last_seen = time.time()
            
            self.logger.info(f"Received contribution {contribution.contribution_id} "
                           f"from {contribution.participant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to receive contribution: {e}")
            return False
    
    def _validate_contribution(self, contribution: BenchmarkContribution) -> bool:
        """Validate a benchmark contribution."""
        # Check participant exists
        if contribution.participant_id not in self.participants:
            self.logger.warning(f"Unknown participant: {contribution.participant_id}")
            return False
        
        # Check contribution type
        valid_types = ["metrics", "model_weights", "evaluation_data", "privacy_metrics"]
        if contribution.contribution_type not in valid_types:
            self.logger.warning(f"Invalid contribution type: {contribution.contribution_type}")
            return False
        
        # Check data hash
        if not contribution.data_hash:
            self.logger.warning("Missing data hash in contribution")
            return False
        
        # Check validation score
        if contribution.validation_score < 0.5:
            self.logger.warning(f"Low validation score: {contribution.validation_score}")
            return False
        
        return True
    
    def _estimate_privacy_cost(self, contribution: BenchmarkContribution) -> float:
        """Estimate privacy budget cost for a contribution."""
        # Simple privacy cost estimation
        base_cost = 0.1
        
        # Cost varies by contribution type
        type_costs = {
            "metrics": 0.1,
            "model_weights": 0.5,
            "evaluation_data": 0.3,
            "privacy_metrics": 0.2
        }
        
        type_cost = type_costs.get(contribution.contribution_type, base_cost)
        
        # Higher cost for less privacy-preserved contributions
        if not contribution.privacy_preserved:
            type_cost *= 2.0
        
        return type_cost
    
    async def aggregate_contributions(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Aggregate contributions from a federated round."""
        try:
            # Get contributions for this round
            round_contributions = [
                c for c in self.contributions 
                if round_id in c.metadata.get("round_id", "")
            ]
            
            if not round_contributions:
                self.logger.warning(f"No contributions found for round {round_id}")
                return None
            
            self.logger.info(f"Aggregating {len(round_contributions)} contributions for round {round_id}")
            
            # Group contributions by type
            contribution_groups = defaultdict(list)
            for contrib in round_contributions:
                contribution_groups[contrib.contribution_type].append(contrib)
            
            aggregated_results = {}
            
            # Aggregate metrics
            if "metrics" in contribution_groups:
                metrics_data = []
                participant_weights = []
                
                for contrib in contribution_groups["metrics"]:
                    # Decode contribution data (simulation)
                    metrics = self._decode_contribution_data(contrib)
                    if metrics:
                        metrics_data.append(metrics)
                        
                        # Get participant weight
                        participant = self.participants.get(contrib.participant_id)
                        weight = participant.contribution_weight if participant else 1.0
                        participant_weights.append(weight)
                
                if metrics_data:
                    aggregated_metrics = self.aggregator.aggregate_metrics(
                        metrics_data, participant_weights
                    )
                    aggregated_results["aggregated_metrics"] = aggregated_metrics
            
            # Aggregate model weights
            if "model_weights" in contribution_groups:
                weights_data = []
                participant_weights = []
                
                for contrib in contribution_groups["model_weights"]:
                    weights = self._decode_contribution_data(contrib)
                    if weights:
                        weights_data.append(weights)
                        
                        participant = self.participants.get(contrib.participant_id)
                        weight = participant.contribution_weight if participant else 1.0
                        participant_weights.append(weight)
                
                if weights_data:
                    aggregated_weights = self.aggregator.aggregate_model_weights(
                        weights_data, participant_weights
                    )
                    aggregated_results["aggregated_weights"] = aggregated_weights
            
            # Store aggregated model
            if aggregated_results:
                model_id = f"federated_model_{round_id}"
                federated_model = FederatedModel(
                    model_id=model_id,
                    version=self.round_number,
                    global_weights=aggregated_results.get("aggregated_weights", {}),
                    participant_contributions=[c.participant_id for c in round_contributions],
                    aggregation_method="secure_weighted_average",
                    performance_metrics=aggregated_results.get("aggregated_metrics", {}),
                    last_updated=time.time()
                )
                
                self.federated_models[model_id] = federated_model
            
            # Calculate round statistics
            aggregated_results["round_statistics"] = {
                "round_id": round_id,
                "round_number": self.round_number,
                "participants_count": len(set(c.participant_id for c in round_contributions)),
                "contributions_count": len(round_contributions),
                "aggregation_time": time.time(),
                "privacy_level": self.privacy_level.value
            }
            
            self.logger.info(f"Aggregation complete for round {round_id}")
            return aggregated_results
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate contributions for round {round_id}: {e}")
            return None
    
    def _decode_contribution_data(self, contribution: BenchmarkContribution) -> Optional[Dict[str, Any]]:
        """Decode contribution data (simulation)."""
        # In practice, this would decode actual encrypted/serialized data
        
        if contribution.contribution_type == "metrics":
            # Mock metrics data
            return {
                "fvd_score": np.random.uniform(80, 120),
                "inception_score": np.random.uniform(30, 50),
                "clip_similarity": np.random.uniform(0.2, 0.4),
                "temporal_consistency": np.random.uniform(0.6, 0.9),
                "motion_coherence": np.random.uniform(0.5, 0.8),
                "latency_ms": np.random.uniform(1000, 5000),
                "memory_gb": np.random.uniform(4, 16)
            }
        
        elif contribution.contribution_type == "model_weights":
            # Mock model weights
            if TORCH_AVAILABLE:
                return {
                    "layer1.weight": torch.randn(64, 128),
                    "layer1.bias": torch.randn(64),
                    "layer2.weight": torch.randn(32, 64),
                    "layer2.bias": torch.randn(32)
                }
            else:
                return {
                    "layer1.weight": np.random.randn(64, 128),
                    "layer1.bias": np.random.randn(64),
                    "layer2.weight": np.random.randn(32, 64),
                    "layer2.bias": np.random.randn(32)
                }
        
        return None
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status."""
        active_participants = self._get_active_participants()
        
        status = {
            "coordinator_id": self.coordinator_id,
            "privacy_level": self.privacy_level.value,
            "round_number": self.round_number,
            "total_participants": len(self.participants),
            "active_participants": len(active_participants),
            "total_contributions": len(self.contributions),
            "federated_models": len(self.federated_models),
            "is_running": self.is_running
        }
        
        # Participant statistics
        participant_stats = {}
        for participant_id, participant in self.participants.items():
            privacy_status = self.privacy_manager.get_privacy_status(participant_id)
            participant_stats[participant_id] = {
                "institution": participant.institution_name,
                "role": participant.role.value,
                "trust_score": participant.trust_score,
                "contribution_weight": participant.contribution_weight,
                "is_active": participant_id in active_participants,
                "privacy_budget_remaining": privacy_status.get("epsilon_remaining", 0)
            }
        
        status["participants"] = participant_stats
        
        return status
    
    def export_federation_data(self, filepath: str):
        """Export federation data for analysis."""
        export_data = {
            "coordinator_info": {
                "coordinator_id": self.coordinator_id,
                "privacy_level": self.privacy_level.value,
                "round_number": self.round_number
            },
            "participants": [p.to_dict() for p in self.participants.values()],
            "contributions": [c.to_dict() for c in self.contributions],
            "federated_models": [m.to_dict() for m in self.federated_models.values()],
            "federation_status": self.get_federation_status(),
            "export_timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported federation data to {filepath}")


class FederatedParticipantNode:
    """Participant node in federated benchmark learning."""
    
    def __init__(self, participant_info: FederatedParticipant):
        self.participant_info = participant_info
        self.coordinator_connection = None
        self.local_models = {}
        self.local_data = {}
        self.pending_tasks = deque()
        self.completed_contributions = []
        self.is_active = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def connect_to_coordinator(self, coordinator_endpoint: str) -> bool:
        """Connect to the federated coordinator."""
        try:
            # Simulate connection to coordinator
            self.coordinator_connection = {
                "endpoint": coordinator_endpoint,
                "connected_at": time.time(),
                "status": "connected"
            }
            
            self.is_active = True
            self.logger.info(f"Connected to coordinator at {coordinator_endpoint}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to coordinator: {e}")
            return False
    
    async def register_with_coordinator(self) -> bool:
        """Register this participant with the coordinator."""
        if not self.coordinator_connection:
            self.logger.error("Not connected to coordinator")
            return False
        
        try:
            # Simulate registration
            self.logger.info(f"Registering participant {self.participant_info.participant_id} "
                           f"with coordinator")
            
            # Update last seen
            self.participant_info.last_seen = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register with coordinator: {e}")
            return False
    
    async def receive_task(self, task: Dict[str, Any]) -> bool:
        """Receive a task from the coordinator."""
        try:
            self.logger.info(f"Received task: {task.get('round_id', 'unknown')} "
                           f"({task.get('task_type', 'unknown')})")
            
            # Add to pending tasks
            task["received_at"] = time.time()
            self.pending_tasks.append(task)
            
            # Process task immediately (in practice, might queue for later)
            await self._process_task(task)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to receive task: {e}")
            return False
    
    async def _process_task(self, task: Dict[str, Any]):
        """Process a federated learning task."""
        task_type = task.get("task_type", "unknown")
        round_id = task.get("round_id", "unknown")
        
        self.logger.info(f"Processing task {round_id}: {task_type}")
        
        try:
            if task_type == "benchmark_evaluation":
                contribution = await self._perform_benchmark_evaluation(task)
            elif task_type == "model_training":
                contribution = await self._perform_model_training(task)
            elif task_type == "privacy_audit":
                contribution = await self._perform_privacy_audit(task)
            else:
                self.logger.warning(f"Unknown task type: {task_type}")
                return
            
            if contribution:
                # Send contribution to coordinator
                await self._send_contribution(contribution)
            
        except Exception as e:
            self.logger.error(f"Failed to process task {round_id}: {e}")
    
    async def _perform_benchmark_evaluation(self, task: Dict[str, Any]) -> Optional[BenchmarkContribution]:
        """Perform benchmark evaluation on local models/data."""
        
        # Simulate benchmark evaluation
        models_to_evaluate = task.get("models_to_evaluate", [])
        metrics_to_compute = task.get("metrics_to_compute", [])
        
        self.logger.info(f"Evaluating {len(models_to_evaluate)} models "
                        f"with {len(metrics_to_compute)} metrics")
        
        # Simulate computation time
        await asyncio.sleep(np.random.uniform(1, 3))
        
        # Generate mock evaluation results
        evaluation_results = {}
        
        for model_name in models_to_evaluate:
            model_results = {}
            
            for metric_name in metrics_to_compute:
                if metric_name == "fvd_score":
                    model_results[metric_name] = np.random.uniform(80, 120)
                elif metric_name == "inception_score":
                    model_results[metric_name] = np.random.uniform(30, 50)
                elif metric_name == "clip_similarity":
                    model_results[metric_name] = np.random.uniform(0.2, 0.4)
                elif metric_name == "temporal_consistency":
                    model_results[metric_name] = np.random.uniform(0.6, 0.9)
                elif metric_name == "latency_ms":
                    model_results[metric_name] = np.random.uniform(1000, 5000)
                else:
                    model_results[metric_name] = np.random.uniform(0, 1)
            
            evaluation_results[model_name] = model_results
        
        # Apply differential privacy
        if self.participant_info.privacy_level != PrivacyLevel.MINIMAL:
            evaluation_results = self._apply_local_privacy(evaluation_results)
        
        # Create contribution
        contribution = BenchmarkContribution(
            contribution_id=f"contrib_{self.participant_info.participant_id}_{int(time.time())}",
            participant_id=self.participant_info.participant_id,
            timestamp=time.time(),
            contribution_type="metrics",
            data_hash=self._compute_data_hash(evaluation_results),
            metadata={
                "round_id": task.get("round_id"),
                "task_type": task.get("task_type"),
                "models_evaluated": len(models_to_evaluate),
                "metrics_computed": len(metrics_to_compute),
                "evaluation_results": evaluation_results
            },
            privacy_preserved=self.participant_info.privacy_level != PrivacyLevel.MINIMAL,
            validation_score=0.85  # Mock validation score
        )
        
        return contribution
    
    async def _perform_model_training(self, task: Dict[str, Any]) -> Optional[BenchmarkContribution]:
        """Perform local model training and share weight updates."""
        
        self.logger.info("Performing local model training")
        
        # Simulate model training
        await asyncio.sleep(np.random.uniform(2, 5))
        
        # Generate mock weight updates
        if TORCH_AVAILABLE:
            weight_updates = {
                "layer1.weight": torch.randn(64, 128),
                "layer1.bias": torch.randn(64),
                "layer2.weight": torch.randn(32, 64),
                "layer2.bias": torch.randn(32),
                "output.weight": torch.randn(16, 32),
                "output.bias": torch.randn(16)
            }
        else:
            weight_updates = {
                "layer1.weight": np.random.randn(64, 128),
                "layer1.bias": np.random.randn(64),
                "layer2.weight": np.random.randn(32, 64),
                "layer2.bias": np.random.randn(32),
                "output.weight": np.random.randn(16, 32),
                "output.bias": np.random.randn(16)
            }
        
        # Apply privacy preservation to weights
        if self.participant_info.privacy_level != PrivacyLevel.MINIMAL:
            weight_updates = self._apply_weight_privacy(weight_updates)
        
        # Create contribution
        contribution = BenchmarkContribution(
            contribution_id=f"contrib_{self.participant_info.participant_id}_{int(time.time())}",
            participant_id=self.participant_info.participant_id,
            timestamp=time.time(),
            contribution_type="model_weights",
            data_hash=self._compute_data_hash(weight_updates),
            metadata={
                "round_id": task.get("round_id"),
                "task_type": task.get("task_type"),
                "weight_updates": weight_updates,
                "training_samples": np.random.randint(100, 1000),
                "local_epochs": np.random.randint(1, 5)
            },
            privacy_preserved=self.participant_info.privacy_level != PrivacyLevel.MINIMAL,
            validation_score=0.80
        )
        
        return contribution
    
    async def _perform_privacy_audit(self, task: Dict[str, Any]) -> Optional[BenchmarkContribution]:
        """Perform privacy audit and share privacy metrics."""
        
        self.logger.info("Performing privacy audit")
        
        # Simulate privacy audit
        await asyncio.sleep(np.random.uniform(0.5, 1.5))
        
        # Generate mock privacy metrics
        privacy_metrics = {
            "epsilon_consumed": np.random.uniform(0.1, 2.0),
            "delta_consumed": np.random.uniform(1e-6, 1e-4),
            "noise_level": np.random.uniform(0.01, 0.1),
            "privacy_loss": np.random.uniform(0.05, 0.3),
            "data_points_used": np.random.randint(50, 500)
        }
        
        # Create contribution
        contribution = BenchmarkContribution(
            contribution_id=f"contrib_{self.participant_info.participant_id}_{int(time.time())}",
            participant_id=self.participant_info.participant_id,
            timestamp=time.time(),
            contribution_type="privacy_metrics",
            data_hash=self._compute_data_hash(privacy_metrics),
            metadata={
                "round_id": task.get("round_id"),
                "task_type": task.get("task_type"),
                "privacy_metrics": privacy_metrics,
                "privacy_level": self.participant_info.privacy_level.value
            },
            privacy_preserved=True,  # Privacy metrics are always privacy-preserved
            validation_score=0.90
        )
        
        return contribution
    
    def _apply_local_privacy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply local differential privacy to data."""
        # Simple noise addition for differential privacy
        
        privacy_params = {
            PrivacyLevel.STANDARD: 1.0,
            PrivacyLevel.HIGH: 2.0,
            PrivacyLevel.MAXIMUM: 3.0
        }
        
        noise_scale = privacy_params.get(self.participant_info.privacy_level, 1.0)
        
        def add_noise_recursive(obj):
            if isinstance(obj, dict):
                return {k: add_noise_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float)):
                noise = np.random.laplace(0, noise_scale * 0.01)
                return float(obj + noise)
            else:
                return obj
        
        return add_noise_recursive(data)
    
    def _apply_weight_privacy(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy preservation to model weights."""
        
        privacy_params = {
            PrivacyLevel.STANDARD: 0.01,
            PrivacyLevel.HIGH: 0.02,
            PrivacyLevel.MAXIMUM: 0.03
        }
        
        noise_scale = privacy_params.get(self.participant_info.privacy_level, 0.01)
        
        private_weights = {}
        
        for key, weight in weights.items():
            if TORCH_AVAILABLE and isinstance(weight, torch.Tensor):
                noise = torch.normal(0, noise_scale, size=weight.shape)
                private_weights[key] = weight + noise
            elif isinstance(weight, np.ndarray):
                noise = np.random.normal(0, noise_scale, weight.shape)
                private_weights[key] = weight + noise
            else:
                private_weights[key] = weight
        
        return private_weights
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data for integrity verification."""
        # Convert data to string representation
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def _send_contribution(self, contribution: BenchmarkContribution):
        """Send contribution to coordinator."""
        try:
            # Simulate sending contribution
            self.logger.info(f"Sending contribution {contribution.contribution_id} to coordinator")
            
            # Add to completed contributions
            self.completed_contributions.append(contribution)
            
            # Update participant activity
            self.participant_info.last_seen = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to send contribution: {e}")
    
    def get_participant_status(self) -> Dict[str, Any]:
        """Get participant status."""
        return {
            "participant_id": self.participant_info.participant_id,
            "institution": self.participant_info.institution_name,
            "role": self.participant_info.role.value,
            "privacy_level": self.participant_info.privacy_level.value,
            "is_active": self.is_active,
            "pending_tasks": len(self.pending_tasks),
            "completed_contributions": len(self.completed_contributions),
            "last_seen": self.participant_info.last_seen,
            "trust_score": self.participant_info.trust_score,
            "coordinator_connected": self.coordinator_connection is not None
        }


# Example usage and testing
async def run_federated_learning_example():
    """Example of running federated benchmark learning."""
    
    print("=== Federated Benchmark Learning Example ===")
    
    # Create coordinator
    coordinator = FederatedCoordinator(
        coordinator_id="coord_001",
        privacy_level=PrivacyLevel.STANDARD
    )
    
    # Create participants
    participants = []
    
    # Research university
    university_participant = FederatedParticipant(
        participant_id="univ_001",
        institution_name="Research University",
        role=FederatedRole.PARTICIPANT,
        public_key="mock_public_key_1",
        capabilities={
            "model_evaluation": True,
            "metric_computation": True,
            "privacy_preservation": True,
            "gpu_computing": True
        },
        contribution_weight=1.0,
        privacy_level=PrivacyLevel.STANDARD,
        last_seen=time.time(),
        trust_score=0.9
    )
    
    # Tech company
    company_participant = FederatedParticipant(
        participant_id="comp_001",
        institution_name="Tech Company Labs",
        role=FederatedRole.PARTICIPANT,
        public_key="mock_public_key_2",
        capabilities={
            "model_evaluation": True,
            "metric_computation": True,
            "privacy_preservation": True,
            "large_scale_computing": True
        },
        contribution_weight=1.2,
        privacy_level=PrivacyLevel.HIGH,
        last_seen=time.time(),
        trust_score=0.85
    )
    
    # Research institute
    institute_participant = FederatedParticipant(
        participant_id="inst_001",
        institution_name="AI Research Institute",
        role=FederatedRole.PARTICIPANT,
        public_key="mock_public_key_3",
        capabilities={
            "model_evaluation": True,
            "metric_computation": True,
            "privacy_preservation": True,
            "specialized_models": True
        },
        contribution_weight=0.8,
        privacy_level=PrivacyLevel.STANDARD,
        last_seen=time.time(),
        trust_score=0.95
    )
    
    participants = [university_participant, company_participant, institute_participant]
    
    # Register participants with coordinator
    print("\n--- Registering Participants ---")
    for participant in participants:
        success = coordinator.register_participant(participant)
        print(f"Registered {participant.institution_name}: {'✓' if success else '✗'}")
    
    # Create participant nodes
    participant_nodes = []
    for participant in participants:
        node = FederatedParticipantNode(participant)
        await node.connect_to_coordinator("coordinator.example.com")
        await node.register_with_coordinator()
        participant_nodes.append(node)
    
    # Start federated learning rounds
    print("\n--- Federated Learning Rounds ---")
    
    # Round 1: Benchmark evaluation
    print("\nRound 1: Benchmark Evaluation")
    task_spec_1 = {
        "task_type": "benchmark_evaluation",
        "models": ["svd-xt-1.1", "pika-lumiere", "dreamvideo-v3"],
        "metrics": ["fvd_score", "inception_score", "clip_similarity", "temporal_consistency"],
        "privacy_budget": 0.5,
        "timeout": 1800
    }
    
    round_1_id = await coordinator.start_federated_round(task_spec_1)
    
    # Simulate participants completing tasks
    for node in participant_nodes:
        task = {
            "round_id": round_1_id,
            "task_type": "benchmark_evaluation",
            "models_to_evaluate": task_spec_1["models"],
            "metrics_to_compute": task_spec_1["metrics"]
        }
        await node.receive_task(task)
    
    # Wait for task completion
    await asyncio.sleep(2)
    
    # Aggregate results
    round_1_results = await coordinator.aggregate_contributions(round_1_id)
    if round_1_results:
        print(f"Round 1 aggregated metrics: {list(round_1_results.get('aggregated_metrics', {}).keys())}")
    
    # Round 2: Model training
    print("\nRound 2: Model Training")
    task_spec_2 = {
        "task_type": "model_training",
        "base_model": "federated_video_diffusion",
        "privacy_budget": 1.0,
        "timeout": 3600
    }
    
    round_2_id = await coordinator.start_federated_round(task_spec_2)
    
    # Simulate participants completing training
    for node in participant_nodes:
        task = {
            "round_id": round_2_id,
            "task_type": "model_training",
            "base_model": task_spec_2["base_model"]
        }
        await node.receive_task(task)
    
    await asyncio.sleep(3)
    
    # Aggregate model weights
    round_2_results = await coordinator.aggregate_contributions(round_2_id)
    if round_2_results:
        print(f"Round 2 aggregated weights: {list(round_2_results.get('aggregated_weights', {}).keys())}")
    
    # Round 3: Privacy audit
    print("\nRound 3: Privacy Audit")
    task_spec_3 = {
        "task_type": "privacy_audit",
        "privacy_budget": 0.1,
        "timeout": 900
    }
    
    round_3_id = await coordinator.start_federated_round(task_spec_3)
    
    for node in participant_nodes:
        task = {
            "round_id": round_3_id,
            "task_type": "privacy_audit"
        }
        await node.receive_task(task)
    
    await asyncio.sleep(1)
    
    round_3_results = await coordinator.aggregate_contributions(round_3_id)
    if round_3_results:
        print(f"Round 3 privacy metrics collected")
    
    # Display federation status
    print("\n--- Federation Status ---")
    status = coordinator.get_federation_status()
    print(f"Total participants: {status['total_participants']}")
    print(f"Active participants: {status['active_participants']}")
    print(f"Total contributions: {status['total_contributions']}")
    print(f"Federated models: {status['federated_models']}")
    print(f"Completed rounds: {status['round_number']}")
    
    # Display participant status
    print("\n--- Participant Status ---")
    for node in participant_nodes:
        node_status = node.get_participant_status()
        print(f"{node_status['institution']}: "
              f"Contributions={node_status['completed_contributions']}, "
              f"Trust={node_status['trust_score']:.2f}")
    
    # Privacy budget status
    print("\n--- Privacy Budget Status ---")
    for participant in participants:
        privacy_status = coordinator.privacy_manager.get_privacy_status(participant.participant_id)
        print(f"{participant.institution_name}: "
              f"ε remaining = {privacy_status.get('epsilon_remaining', 0):.2f}, "
              f"queries = {privacy_status.get('queries_made', 0)}")
    
    # Export federation data
    export_path = "federated_learning_results.json"
    coordinator.export_federation_data(export_path)
    print(f"\nFederation data exported to {export_path}")
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    
    all_contributions = coordinator.contributions
    contribution_types = defaultdict(int)
    for contrib in all_contributions:
        contribution_types[contrib.contribution_type] += 1
    
    print(f"Contribution breakdown:")
    for contrib_type, count in contribution_types.items():
        print(f"  {contrib_type}: {count}")
    
    # Calculate privacy consumption
    total_privacy_consumed = 0
    for participant in participants:
        privacy_status = coordinator.privacy_manager.get_privacy_status(participant.participant_id)
        total_privacy_consumed += privacy_status.get("epsilon_used", 0)
    
    print(f"Total privacy budget consumed: {total_privacy_consumed:.2f}")
    print(f"Average contribution validation score: "
          f"{np.mean([c.validation_score for c in all_contributions]):.3f}")
    
    return {
        "coordinator": coordinator,
        "participants": participant_nodes,
        "round_results": [round_1_results, round_2_results, round_3_results],
        "federation_status": status
    }


if __name__ == "__main__":
    # Run example
    import asyncio
    asyncio.run(run_federated_learning_example())