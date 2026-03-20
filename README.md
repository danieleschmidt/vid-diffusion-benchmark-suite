# vid-diffusion-benchmark-suite

Unified benchmark suite for evaluating video diffusion models.

## Metrics

| Metric | Description |
|--------|-------------|
| **FVD Proxy** | Frame-level FID proxy via pixel statistics (lower = better) |
| **Temporal Consistency** | Mean cosine similarity between consecutive frames [0,1] |
| **VRAM Estimate** | `params × dtype_bytes × activation_multiplier / 1024²` MB |
| **Latency** | Wall-clock inference time in milliseconds |

## Usage

```python
from vid_bench.models import GradientModel
report = GradientModel().run(num_frames=16, height=64, width=64)
print(report.to_json())
```

### CLI

```bash
python -m vid_bench.runner --models gradient small-fast --frames 16 --output results.json
```

## Adding a New Model

```python
from vid_bench.benchmark import VideoModelBenchmark
import numpy as np

class MyModel(VideoModelBenchmark):
    model_name = "my-model"
    def get_param_count(self): return 500_000_000
    def generate_frames(self, prompt, num_frames, height, width):
        # Your generation logic
        return np.zeros((num_frames, height, width, 3), dtype=np.uint8)
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```
