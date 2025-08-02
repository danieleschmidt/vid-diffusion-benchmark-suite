# Test Fixtures

This directory contains test fixtures and mock data for the Video Diffusion Benchmark Suite test suite.

## Directory Structure

```
fixtures/
├── README.md                    # This file
├── sample_videos/              # Sample video files for testing
│   ├── short_video.mp4         # 5-second test video
│   ├── standard_video.mp4      # 25-frame standard test video
│   └── high_res_video.mp4      # High resolution test video
├── mock_models/               # Mock model configurations
│   ├── mock_svd.yaml          # Mock Stable Video Diffusion config
│   ├── mock_cogvideo.yaml     # Mock CogVideo config
│   └── mock_custom.yaml       # Custom mock model config
├── test_prompts/              # Test prompt datasets
│   ├── unit_test_prompts.json # Prompts for unit testing
│   ├── integration_prompts.json # Prompts for integration testing
│   └── performance_prompts.json # Prompts for performance testing
├── expected_outputs/          # Expected test outputs
│   ├── metrics/               # Expected metric values
│   ├── reports/               # Expected report formats
│   └── api_responses/         # Expected API response formats
└── mock_data/                 # Mock data for various components
    ├── mock_metrics.json      # Mock metric calculation results
    ├── mock_profiles.json     # Mock performance profiles
    └── mock_leaderboard.json  # Mock leaderboard data
```

## Usage Guidelines

### Video Fixtures
- Use `short_video.mp4` for quick unit tests
- Use `standard_video.mp4` for standard evaluation pipeline tests
- Use `high_res_video.mp4` for memory and performance testing

### Model Fixtures
- Mock model configurations should match real model interfaces
- Include all required parameters and metadata
- Use consistent naming conventions

### Prompt Fixtures
- Keep prompts diverse but predictable for testing
- Include edge cases and common scenarios
- Maintain consistent structure across files

### Expected Outputs
- Update expected outputs when making changes to evaluation logic
- Include both successful and error scenarios
- Document any assumptions or dependencies

## Best Practices

1. **Reproducibility**: All fixtures should produce deterministic results
2. **Maintenance**: Regular review and update of fixtures
3. **Size**: Keep fixture files small but representative
4. **Documentation**: Document the purpose and usage of each fixture
5. **Version Control**: Track changes to fixtures for regression testing

## Creating New Fixtures

When adding new fixtures:
1. Follow the existing directory structure
2. Add clear documentation about the fixture's purpose
3. Ensure the fixture is self-contained and doesn't depend on external resources
4. Add corresponding tests that use the fixture
5. Update this README with information about the new fixture