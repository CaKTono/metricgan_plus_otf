# audio_degradation module
# GPU-accelerated audio degradation for MetricGAN+

from .gpu_audio_effects import (
    GPUNoiseMixer,
    GPUClipping,
    GPUBitcrush,
    GPUDistortion,
    GPUPacketLoss,
    GPUReverb,
    GPUBandwidthLimitation,
    GPUChorus,
    GPUParametricEQ,
)

from .gpu_degradation_pipeline import (
    GPUDegradationPipeline,
    HybridDegradationPipeline,
    DEFAULT_DEGRADATION_CONFIG,
)

from .cached_dataset import (
    CachedAudioDegradationDataset,
    gpu_collate_fn,
    create_gpu_dataloader,
)

__all__ = [
    # Effects
    'GPUNoiseMixer',
    'GPUClipping',
    'GPUBitcrush',
    'GPUDistortion',
    'GPUPacketLoss',
    'GPUReverb',
    'GPUBandwidthLimitation',
    'GPUChorus',
    'GPUParametricEQ',
    # Pipeline
    'GPUDegradationPipeline',
    'HybridDegradationPipeline',
    'DEFAULT_DEGRADATION_CONFIG',
    # Dataset
    'CachedAudioDegradationDataset',
    'gpu_collate_fn',
    'create_gpu_dataloader',
]
