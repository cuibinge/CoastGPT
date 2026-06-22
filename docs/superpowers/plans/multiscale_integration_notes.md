# Multi-Scale Training Integration Notes

The `Dataset/multiscale_sampler.py` module is ready. To integrate into Stage 3 training:

## Integration points in train_stage_three.py

1. Import the sampler:
```python
from Dataset.multiscale_sampler import build_multiscale_sampler
```

2. After config is loaded, build the sampler:
```python
ms_sampler = build_multiscale_sampler(config)
```

3. In the training loop, before each batch:
```python
current_h, current_w = ms_sampler.sample()
# Resize batch images to (current_h, current_w)
# Rebuild Mask R-CNN with current min_size/max_size and anchor sizes
```

4. For epoch-level switching (simpler than per-batch):
```python
# At the start of each epoch:
epoch_size = ms_sampler.sizes[epoch % len(ms_sampler.sizes)]
model = build_aqua_maskrcnn(
    adapter,
    min_size=epoch_size[0],
    max_size=epoch_size[1],
    anchor_relative_scales=config.anchor_generator.scales,
)
```

## Usage

When multi_scale.enabled=true in config:
- Sizes pool: [224, 280, 336, 392]
- Each epoch uses one size (round-robin)
- Anchor sizes auto-scale via anchor_generator.scales (relative mode)
