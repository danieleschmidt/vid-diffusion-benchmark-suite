"""Comprehensive tests for federated benchmarking functionality."""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.vid_diffusion_bench.federated_benchmark import (
    FederatedCoordinator,
    FederatedParticipant,
    FederatedConfig,
    PrivacyBudget,
    SecureResult,
    CryptoManager,
    DifferentialPrivacyEngine,
    BenchmarkPhase,
    create_federated_session,
    join_federated_session
)
from src.vid_diffusion_bench.benchmark import BenchmarkResult


class TestCryptoManager:
    """Test cryptographic operations."""
    
    def test_initialization(self):
        """Test crypto manager initialization."""
        crypto = CryptoManager()
        
        assert crypto.private_key is not None
        assert crypto.public_key is not None
        assert crypto.symmetric_key is not None
        assert crypto.cipher is not None
    
    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption."""
        crypto = CryptoManager()
        original_data = b"test data for encryption"
        
        # Encrypt data
        encrypted = crypto.encrypt_data(original_data)
        assert encrypted != original_data
        assert len(encrypted) > len(original_data)  # Encrypted data should be larger
        
        # Decrypt data
        decrypted = crypto.decrypt_data(encrypted)
        assert decrypted == original_data
    
    def test_sign_verify_data(self):
        """Test data signing and verification."""
        crypto = CryptoManager()
        data = b"test data for signing"
        
        # Sign data
        signature = crypto.sign_data(data)
        assert len(signature) > 0
        
        # Verify signature
        is_valid = crypto.verify_signature(data, signature, crypto.public_key)
        assert is_valid
        
        # Verify with wrong data should fail
        wrong_data = b"wrong data"
        is_valid = crypto.verify_signature(wrong_data, signature, crypto.public_key)
        assert not is_valid
    
    def test_public_key_export(self):
        """Test public key export."""
        crypto = CryptoManager()
        
        pem_key = crypto.get_public_key_pem()
        assert isinstance(pem_key, bytes)
        assert b"BEGIN PUBLIC KEY" in pem_key
        assert b"END PUBLIC KEY" in pem_key
    
    def test_key_exchange(self):
        """Test key exchange between parties."""
        crypto1 = CryptoManager()
        crypto2 = CryptoManager()
        
        # Exchange keys
        pub_key_1 = crypto1.get_public_key_pem()
        pub_key_2 = crypto2.get_public_key_pem()
        
        encrypted_key = crypto1.exchange_keys(pub_key_2)
        assert len(encrypted_key) > 0


class TestDifferentialPrivacyEngine:
    """Test differential privacy implementation."""
    
    def test_initialization(self):
        """Test DP engine initialization."""
        dp = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
        
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
        assert dp.budget.epsilon == 1.0
        assert dp.budget.spent == 0.0
    
    def test_laplace_noise(self):
        """Test Laplace noise addition."""
        dp = DifferentialPrivacyEngine(epsilon=1.0)
        original_value = 10.0
        
        noisy_value = dp.add_laplace_noise(original_value, sensitivity=1.0, allocated_epsilon=0.1)
        
        # Value should be different (with high probability)
        assert noisy_value != original_value
        
        # Budget should be updated
        assert dp.budget.spent == 0.1
    
    def test_gaussian_noise(self):
        """Test Gaussian noise addition."""
        dp = DifferentialPrivacyEngine(epsilon=1.0)
        original_value = 5.0
        
        noisy_value = dp.add_gaussian_noise(original_value, sensitivity=1.0, allocated_epsilon=0.2)
        
        # Value should be different
        assert noisy_value != original_value
        
        # Budget should be updated
        assert dp.budget.spent == 0.2
    
    def test_budget_exhaustion(self):
        """Test privacy budget exhaustion."""
        dp = DifferentialPrivacyEngine(epsilon=1.0)
        
        # Use up most of the budget
        dp.add_laplace_noise(10.0, 1.0, 0.9)
        
        # Trying to use more than remaining budget should raise error
        with pytest.raises(ValueError, match="Privacy budget exceeded"):
            dp.add_laplace_noise(10.0, 1.0, 0.2)
    
    def test_privatize_result(self):
        """Test benchmark result privatization."""
        dp = DifferentialPrivacyEngine(epsilon=2.0)
        
        # Create mock benchmark result
        result = BenchmarkResult("test_model", ["prompt1"])
        result.set_metrics(0.8, 0.7, 0.9, 0.85)
        result.set_performance(100.0, 5.0, 16.0, 200.0)
        
        original_fvd = result.metrics["fvd"]
        
        # Privatize result
        privatized = dp.privatize_result(result, allocated_epsilon=0.5)
        
        # Metrics should be modified (with high probability)
        assert privatized.metrics["fvd"] != original_fvd
        
        # Budget should be updated
        assert dp.budget.spent > 0


class TestFederatedConfig:
    """Test federated configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = FederatedConfig(
            session_id="test_session",
            coordinator_endpoint="localhost:8080"
        )
        
        assert config.session_id == "test_session"
        assert config.coordinator_endpoint == "localhost:8080"
        assert config.participants == []
        assert config.privacy_level == "differential"
        assert config.consensus_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FederatedConfig(
            session_id="custom_session",
            coordinator_endpoint="cluster.example.com:9000",
            participants=["node1", "node2", "node3"],
            privacy_level="secure_aggregation",
            consensus_threshold=0.9,
            timeout_minutes=120
        )
        
        assert config.participants == ["node1", "node2", "node3"]
        assert config.privacy_level == "secure_aggregation"
        assert config.consensus_threshold == 0.9
        assert config.timeout_minutes == 120


class TestFederatedCoordinator:
    """Test federated coordinator functionality."""
    
    @pytest.fixture
    def coordinator_config(self):
        """Create test coordinator configuration."""
        return FederatedConfig(
            session_id="test_session",
            coordinator_endpoint="localhost:8080",
            participants=["participant1", "participant2"],
            timeout_minutes=10
        )
    
    @pytest.fixture
    def coordinator(self, coordinator_config):
        """Create coordinator instance."""
        return FederatedCoordinator(coordinator_config)
    
    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.config.session_id == "test_session"
        assert coordinator.current_phase == BenchmarkPhase.SETUP
        assert len(coordinator.participants) == 0
        assert len(coordinator.results) == 0
    
    @patch('src.vid_diffusion_bench.federated_benchmark.time.time')
    @pytest.mark.asyncio
    async def test_setup_secure_channels(self, mock_time, coordinator):
        """Test secure channel setup."""
        mock_time.return_value = 1000.0
        
        # Mock the key exchange and verification methods
        coordinator._exchange_keys = AsyncMock(return_value=b"mock_public_key")
        coordinator._verify_participant_identity = AsyncMock(return_value=True)
        
        await coordinator._setup_secure_channels()
        
        # Should have registered participants
        assert len(coordinator.participants) == len(coordinator.config.participants)
        for participant_id in coordinator.config.participants:
            assert participant_id in coordinator.participants
            assert coordinator.participants[participant_id]["status"] == "verified"
    
    @pytest.mark.asyncio
    async def test_register_participants(self, coordinator):
        """Test participant registration."""
        # Pre-populate with verified participants
        for participant_id in coordinator.config.participants:
            coordinator.participants[participant_id] = {
                "public_key": b"test_key",
                "status": "verified",
                "last_heartbeat": time.time()
            }
        
        await coordinator._register_participants()
        
        # All participants should remain registered
        assert len(coordinator.participants) == len(coordinator.config.participants)
    
    @pytest.mark.asyncio
    async def test_distribute_benchmark_spec(self, coordinator):
        """Test benchmark specification distribution."""
        # Mock the send method
        coordinator._send_to_participant = AsyncMock(return_value=True)
        
        # Add some participants
        for participant_id in coordinator.config.participants:
            coordinator.participants[participant_id] = {"status": "verified"}
        
        await coordinator._distribute_benchmark_spec()
        
        # Should have sent to all participants
        expected_calls = len(coordinator.participants)
        assert coordinator._send_to_participant.call_count == expected_calls
    
    def test_differential_private_aggregation(self, coordinator):
        """Test differential private aggregation."""
        # Create mock results
        results = []
        for i in range(3):
            result = {
                "metrics": {"fvd": 100 + i, "clip_score": 0.8 + i * 0.1},
                "performance": {"latency_ms": 2000 + i * 100}
            }
            results.append(result)
        
        aggregated = coordinator._differential_private_aggregation(results)
        
        assert "metrics" in aggregated
        assert "performance" in aggregated
        assert "metadata" in aggregated
        assert aggregated["metadata"]["aggregation_method"] == "differential_private"
        assert aggregated["metadata"]["participant_count"] == 3
    
    def test_simple_aggregation(self, coordinator):
        """Test simple aggregation without privacy."""
        results = []
        for i in range(3):
            result = {
                "metrics": {"fvd": 100 + i * 10, "clip_score": 0.7 + i * 0.05},
                "performance": {"latency_ms": 2000 + i * 200}
            }
            results.append(result)
        
        aggregated = coordinator._simple_aggregation(results)
        
        assert "metrics" in aggregated
        assert "performance" in aggregated
        assert aggregated["metadata"]["aggregation_method"] == "simple_average"
        
        # Check statistical aggregation
        fvd_stats = aggregated["metrics"]["fvd"]
        assert fvd_stats["mean"] == pytest.approx(110.0, rel=1e-2)
        assert fvd_stats["count"] == 3
    
    @pytest.mark.asyncio
    async def test_validate_consensus(self, coordinator):
        """Test consensus validation."""
        # Create aggregated results with sufficient participants
        aggregated_results = {
            "metadata": {"participant_count": 2},  # Meets 80% of 2 participants
            "metrics": {
                "fvd": {"mean": 100, "std": 5},
                "clip_score": {"mean": 0.8, "std": 0.02}
            }
        }
        
        # Mock privacy level to none for variance checking
        coordinator.config.privacy_level = "none"
        
        result = await coordinator._validate_consensus(aggregated_results)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_consensus_insufficient_participants(self, coordinator):
        """Test consensus validation with insufficient participants."""
        aggregated_results = {
            "metadata": {"participant_count": 1},  # Less than 80% of 2 participants
        }
        
        result = await coordinator._validate_consensus(aggregated_results)
        assert result is False


class TestFederatedParticipant:
    """Test federated participant functionality."""
    
    @pytest.fixture
    def participant(self):
        """Create participant instance."""
        return FederatedParticipant("test_participant", "localhost:8080")
    
    def test_initialization(self, participant):
        """Test participant initialization."""
        assert participant.participant_id == "test_participant"
        assert participant.coordinator_endpoint == "localhost:8080"
        assert participant.crypto is not None
        assert not participant.session_active
    
    @patch('src.vid_diffusion_bench.federated_benchmark.BenchmarkSuite')
    @pytest.mark.asyncio
    async def test_execute_benchmark(self, mock_suite, participant):
        """Test benchmark execution."""
        # Mock benchmark suite
        mock_result = BenchmarkResult("test_model", ["prompt1"])
        mock_result.set_metrics(0.8, 0.7, 0.9, 0.85)
        
        mock_suite_instance = Mock()
        mock_suite_instance.evaluate_model = Mock(return_value=mock_result)
        mock_suite.return_value = mock_suite_instance
        
        # Execute benchmark
        spec = {
            "models": ["test_model"],
            "prompts": ["test prompt"],
            "settings": {"num_frames": 16, "fps": 8}
        }
        
        result = await participant._execute_benchmark(spec)
        
        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "test_model"
        assert mock_suite_instance.evaluate_model.called
    
    @pytest.mark.asyncio
    async def test_submit_results(self, participant):
        """Test result submission."""
        # Create test result
        result = BenchmarkResult("test_model", ["prompt1"])
        result.set_metrics(0.8, 0.7, 0.9, 0.85)
        
        # Mock the submission - in real implementation would send over network
        await participant._submit_results(result)
        
        # Test passes if no exception is raised


class TestFederatedBenchmarkIntegration:
    """Integration tests for federated benchmarking."""
    
    @patch('src.vid_diffusion_bench.federated_benchmark.FederatedCoordinator')
    @pytest.mark.asyncio
    async def test_create_federated_session(self, mock_coordinator_class):
        """Test federated session creation."""
        mock_coordinator = Mock()
        mock_coordinator.start_session = AsyncMock(return_value=True)
        mock_coordinator_class.return_value = mock_coordinator
        
        success = await create_federated_session(
            session_id="test_session",
            participants=["node1", "node2"],
            benchmark_spec={"models": ["test_model"]},
            privacy_level="differential"
        )
        
        assert success is True
        assert mock_coordinator.start_session.called
    
    @patch('src.vid_diffusion_bench.federated_benchmark.FederatedParticipant')
    @pytest.mark.asyncio
    async def test_join_federated_session(self, mock_participant_class):
        """Test joining federated session."""
        mock_participant = Mock()
        mock_participant.join_session = AsyncMock(return_value=True)
        mock_participant_class.return_value = mock_participant
        
        success = await join_federated_session(
            participant_id="test_participant",
            session_id="test_session"
        )
        
        assert success is True
        assert mock_participant.join_session.called


class TestFederatedBenchmarkSecurity:
    """Security tests for federated benchmarking."""
    
    def test_secure_result_creation(self):
        """Test secure result creation."""
        crypto = CryptoManager()
        
        # Create test data
        data = {"metrics": {"fvd": 100}, "participant": "test"}
        data_json = json.dumps(data)
        
        # Encrypt and sign
        encrypted_data = crypto.encrypt_data(data_json.encode())
        signature = crypto.sign_data(data_json.encode())
        
        secure_result = SecureResult(
            participant_id="test_participant",
            encrypted_data=encrypted_data,
            signature=signature,
            timestamp="2025-01-01T00:00:00Z",
            metadata_hash="test_hash"
        )
        
        assert secure_result.participant_id == "test_participant"
        assert len(secure_result.encrypted_data) > 0
        assert len(secure_result.signature) > 0
    
    def test_privacy_budget_management(self):
        """Test privacy budget management."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        
        # Allocate budget
        budget.allocations["task1"] = 0.3
        budget.allocations["task2"] = 0.2
        budget.spent = 0.5
        
        assert budget.spent == 0.5
        assert sum(budget.allocations.values()) == 0.5
        assert budget.epsilon - budget.spent == 0.5  # Remaining budget
    
    def test_crypto_manager_different_keys(self):
        """Test that different crypto managers have different keys."""
        crypto1 = CryptoManager()
        crypto2 = CryptoManager()
        
        key1 = crypto1.get_public_key_pem()
        key2 = crypto2.get_public_key_pem()
        
        assert key1 != key2
        
        # Data encrypted with one key cannot be decrypted with another
        data = b"test data"
        encrypted1 = crypto1.encrypt_data(data)
        
        with pytest.raises(Exception):
            crypto2.decrypt_data(encrypted1)


class TestFederatedBenchmarkPerformance:
    """Performance tests for federated benchmarking."""
    
    def test_crypto_operations_performance(self):
        """Test performance of cryptographic operations."""
        crypto = CryptoManager()
        data = b"test data" * 100  # 900 bytes
        
        # Test encryption performance
        start_time = time.time()
        for _ in range(10):
            encrypted = crypto.encrypt_data(data)
            decrypted = crypto.decrypt_data(encrypted)
        encrypt_time = time.time() - start_time
        
        # Should be reasonably fast
        assert encrypt_time < 1.0  # Less than 1 second for 10 operations
        
        # Test signing performance
        start_time = time.time()
        for _ in range(10):
            signature = crypto.sign_data(data)
            is_valid = crypto.verify_signature(data, signature, crypto.public_key)
            assert is_valid
        sign_time = time.time() - start_time
        
        assert sign_time < 1.0  # Less than 1 second for 10 operations
    
    def test_differential_privacy_performance(self):
        """Test performance of differential privacy operations."""
        dp = DifferentialPrivacyEngine(epsilon=10.0)  # Large budget for testing
        
        # Test noise addition performance
        start_time = time.time()
        for i in range(100):
            noisy_value = dp.add_laplace_noise(float(i), 1.0, 0.05)
        noise_time = time.time() - start_time
        
        assert noise_time < 1.0  # Should be very fast


if __name__ == "__main__":
    pytest.main([__file__])