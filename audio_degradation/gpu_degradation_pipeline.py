# gpu_degradation_pipeline.py
# Main GPU-accelerated degradation pipeline for MetricGAN+
# Compatible with PyTorch 1.7+

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any

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


# Default degradation configuration (matching CPU pipeline)
DEFAULT_DEGRADATION_CONFIG = {
    # Noise
    "p_noise": 0.9,
    "snr_min": -5,
    "snr_max": 20,
    # Reverb
    "p_reverb": 0.5,
    # Clipping
    "p_clipping": 0.2,
    "clipping_min_db": -20,
    "clipping_max_db": 0,
    # Bandwidth limitation
    "p_bandwidth_limitation": 0.2,
    "bandwidth_limitation_rates": [4000, 8000, 16000, 22050, 32000],
    # Bitcrush
    "p_bitcrush": 0.0,
    "bitcrush_min_bits": 3,
    "bitcrush_max_bits": 8,
    # Chorus
    "p_chorus": 0.05,
    "chorus_rate_min": 0.1,
    "chorus_rate_max": 3.0,
    "chorus_depth_min": 0.0,
    "chorus_depth_max": 1.0,
    "chorus_delay_min": 1,
    "chorus_delay_max": 30,
    "chorus_feedback_min": -0.5,
    "chorus_feedback_max": 0.5,
    "chorus_mix_min": 0.4,
    "chorus_mix_max": 0.6,
    # Distortion
    "p_distortion": 0.05,
    "distortion_min_db": 5,
    "distortion_max_db": 20,
    # EQ
    "p_eq": 0.1,
    "eq_min_times": 1,
    "eq_max_times": 3,
    "eq_min_length": 0.5,
    "eq_max_length": 1.0,
    # Packet loss
    "p_pl": 0.05,
    "pl_min_ratio": 0.05,
    "pl_max_ratio": 0.1,
    "pl_min_length": 0.05,
    "pl_max_length": 0.1,
}


class GPUDegradationPipeline(nn.Module):
    """
    GPU-accelerated audio degradation pipeline.

    Applies a chain of audio degradation effects on GPU, matching
    the behavior of the CPU-based audio_degradation_pipeline.py.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        sample_rate: int = 16000,
        device: str = 'cuda'
    ):
        """
        Initialize the GPU degradation pipeline.

        Args:
            config: Degradation configuration dictionary
            sample_rate: Audio sample rate in Hz
            device: Target device ('cuda' or 'cpu')
        """
        super().__init__()

        self.config = config if config is not None else DEFAULT_DEGRADATION_CONFIG.copy()
        self.sample_rate = sample_rate
        self.device = device

        # Initialize all effect modules
        self.noise_mixer = GPUNoiseMixer(use_vad=True)
        self.clipping = GPUClipping()
        self.bitcrush = GPUBitcrush()
        self.distortion = GPUDistortion()
        self.packet_loss = GPUPacketLoss(sample_rate=sample_rate)
        self.reverb = GPUReverb(sample_rate=sample_rate)
        self.bandwidth_limiter = GPUBandwidthLimitation(orig_sr=sample_rate)
        self.chorus = GPUChorus(sample_rate=sample_rate)
        self.eq = GPUParametricEQ(sample_rate=sample_rate)

    def _random_uniform(self, low: float, high: float, size: int = 1) -> torch.Tensor:
        """Generate uniform random values on the correct device."""
        return torch.rand(size, device=self.device) * (high - low) + low

    def _random_choice(self, options: list, size: int = 1) -> list:
        """Random choice from list."""
        indices = torch.randint(0, len(options), (size,))
        return [options[i] for i in indices]

    def _should_apply(self, probability: float, batch_size: int) -> torch.Tensor:
        """Determine which samples in batch should have effect applied."""
        return torch.rand(batch_size, device=self.device) < probability

    def forward(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        rir: Optional[torch.Tensor] = None,
        rir_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply degradation pipeline to a batch of audio.

        Args:
            clean: (batch, time) clean audio
            noise: (batch, time) noise audio
            rir: (batch, rir_len) room impulse responses, or None
            rir_mask: (batch,) boolean mask indicating which samples have RIR

        Returns:
            degraded: (batch, time) degraded audio
            clean_processed: (batch, time) clean audio (potentially reverberant)
        """
        batch_size = clean.shape[0]
        degraded = clean.clone()
        clean_processed = clean.clone()

        # 1. Packet Loss
        apply_pl = self._should_apply(self.config["p_pl"], batch_size)
        if apply_pl.any():
            pl_ratio = self._random_uniform(
                self.config["pl_min_ratio"],
                self.config["pl_max_ratio"],
                batch_size
            )
            pl_result = self.packet_loss(
                degraded,
                pl_ratio,
                self.config["pl_min_length"],
                self.config["pl_max_length"]
            )
            degraded = torch.where(apply_pl.unsqueeze(-1), pl_result, degraded)

        # 2. EQ (applied to random segments)
        apply_eq = self._should_apply(self.config["p_eq"], batch_size)
        if apply_eq.any():
            eq_result = self.eq(degraded)
            degraded = torch.where(apply_eq.unsqueeze(-1), eq_result, degraded)

        # 3. Reverb
        if rir is not None:
            apply_reverb = self._should_apply(self.config["p_reverb"], batch_size)
            if rir_mask is not None:
                apply_reverb = apply_reverb & rir_mask

            if apply_reverb.any():
                # Process only samples that need reverb
                for b in range(batch_size):
                    if apply_reverb[b]:
                        rev_speech, rev_early = self.reverb(
                            clean_processed[b:b+1],
                            degraded[b:b+1],
                            rir[b:b+1]
                        )
                        degraded[b] = rev_speech.squeeze(0)
                        clean_processed[b] = rev_early.squeeze(0)

        # 4. Chorus
        apply_chorus = self._should_apply(self.config["p_chorus"], batch_size)
        if apply_chorus.any():
            rate_hz = self._random_uniform(
                self.config["chorus_rate_min"],
                self.config["chorus_rate_max"],
                batch_size
            )
            depth = self._random_uniform(
                self.config["chorus_depth_min"],
                self.config["chorus_depth_max"],
                batch_size
            )
            delay_ms = self._random_uniform(
                float(self.config["chorus_delay_min"]),
                float(self.config["chorus_delay_max"]),
                batch_size
            )
            feedback = self._random_uniform(
                self.config["chorus_feedback_min"],
                self.config["chorus_feedback_max"],
                batch_size
            )
            mix = self._random_uniform(
                self.config["chorus_mix_min"],
                self.config["chorus_mix_max"],
                batch_size
            )

            chorus_result = self.chorus(degraded, rate_hz, depth, delay_ms, feedback, mix)
            degraded = torch.where(apply_chorus.unsqueeze(-1), chorus_result, degraded)

        # 5. Noise
        apply_noise = self._should_apply(self.config["p_noise"], batch_size)
        if apply_noise.any():
            snr = self._random_uniform(
                self.config["snr_min"],
                self.config["snr_max"],
                batch_size
            )
            noisy, _ = self.noise_mixer(degraded, noise, snr)
            degraded = torch.where(apply_noise.unsqueeze(-1), noisy, degraded)

        # 6. Bitcrush
        apply_bitcrush = self._should_apply(self.config["p_bitcrush"], batch_size)
        if apply_bitcrush.any():
            bit_depth = self._random_uniform(
                float(self.config["bitcrush_min_bits"]),
                float(self.config["bitcrush_max_bits"]),
                batch_size
            ).round()
            bitcrush_result = self.bitcrush(degraded, bit_depth)
            degraded = torch.where(apply_bitcrush.unsqueeze(-1), bitcrush_result, degraded)

        # 7. Clipping
        apply_clipping = self._should_apply(self.config["p_clipping"], batch_size)
        if apply_clipping.any():
            threshold_db = self._random_uniform(
                self.config["clipping_min_db"],
                self.config["clipping_max_db"],
                batch_size
            )
            clipping_result = self.clipping(degraded, threshold_db)
            degraded = torch.where(apply_clipping.unsqueeze(-1), clipping_result, degraded)

        # 8. Distortion
        apply_distortion = self._should_apply(self.config["p_distortion"], batch_size)
        if apply_distortion.any():
            drive_db = self._random_uniform(
                float(self.config["distortion_min_db"]),
                float(self.config["distortion_max_db"]),
                batch_size
            )
            distortion_result = self.distortion(degraded, drive_db)
            degraded = torch.where(apply_distortion.unsqueeze(-1), distortion_result, degraded)

        # 9. Bandwidth Limitation
        apply_bw = self._should_apply(self.config["p_bandwidth_limitation"], batch_size)
        if apply_bw.any():
            # Apply bandwidth limitation per sample (different rates)
            target_rates = self._random_choice(
                self.config["bandwidth_limitation_rates"],
                batch_size
            )
            for b in range(batch_size):
                if apply_bw[b]:
                    degraded[b:b+1] = self.bandwidth_limiter(
                        degraded[b:b+1],
                        target_rates[b]
                    )

        # 10. Final Normalization
        # Normalize to prevent clipping
        max_vals = torch.max(
            torch.max(torch.abs(degraded), dim=-1).values,
            torch.max(
                torch.max(torch.abs(clean_processed), dim=-1).values,
                torch.max(torch.abs(noise), dim=-1).values
            )
        )
        max_vals = torch.clamp(max_vals, min=1e-10).unsqueeze(-1)
        scale = 1.0 / max_vals

        degraded = degraded * scale
        clean_processed = clean_processed * scale

        return degraded, clean_processed


class HybridDegradationPipeline:
    """
    Hybrid pipeline that can use either GPU or CPU degradation.
    Provides fallback to original CPU implementation if needed.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        sample_rate: int = 16000,
        device: str = 'cuda',
        use_gpu: bool = True,
        fallback_cpu: bool = False
    ):
        """
        Initialize hybrid pipeline.

        Args:
            config: Degradation configuration
            sample_rate: Audio sample rate
            device: Target device for GPU pipeline
            use_gpu: Whether to use GPU pipeline
            fallback_cpu: Whether to fallback to CPU on errors
        """
        self.config = config if config is not None else DEFAULT_DEGRADATION_CONFIG.copy()
        self.sample_rate = sample_rate
        self.device = device
        self.use_gpu = use_gpu
        self.fallback_cpu = fallback_cpu

        if use_gpu:
            self.gpu_pipeline = GPUDegradationPipeline(
                config=config,
                sample_rate=sample_rate,
                device=device
            )
            self.gpu_pipeline.to(device)
        else:
            self.gpu_pipeline = None

        # CPU pipeline will be imported on demand
        self._cpu_pipeline = None

    def _get_cpu_pipeline(self):
        """Lazy import of CPU pipeline."""
        if self._cpu_pipeline is None:
            from .audio_degradation_pipeline import process_from_audio_path
            self._cpu_pipeline = process_from_audio_path
        return self._cpu_pipeline

    def __call__(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        rir: Optional[torch.Tensor] = None,
        rir_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply degradation pipeline.

        Args:
            clean: (batch, time) clean audio tensor
            noise: (batch, time) noise audio tensor
            rir: (batch, rir_len) room impulse responses
            rir_mask: (batch,) boolean mask for RIR

        Returns:
            degraded: (batch, time) degraded audio
            clean_processed: (batch, time) processed clean audio
        """
        if self.use_gpu and self.gpu_pipeline is not None:
            try:
                with torch.no_grad():
                    return self.gpu_pipeline(clean, noise, rir, rir_mask)
            except Exception as e:
                if self.fallback_cpu:
                    print(f"GPU degradation failed: {e}, falling back to CPU")
                    return self._apply_cpu_fallback(clean, noise, rir)
                else:
                    raise
        else:
            return self._apply_cpu_fallback(clean, noise, rir)

    def _apply_cpu_fallback(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        rir: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CPU-based degradation as fallback."""
        # This is a simplified fallback - for full CPU support,
        # use the original AudioDegradationDataset
        import numpy as np
        from .audio_degradation_pipeline import (
            add_noise, add_reverberation_v2, bandwidth_limitation,
            pedalboard_equalizer
        )
        import pedalboard as pd
        import random

        batch_size = clean.shape[0]
        device = clean.device

        clean_np = clean.cpu().numpy()
        noise_np = noise.cpu().numpy()
        rir_np = rir.cpu().numpy() if rir is not None else None

        degraded_list = []
        clean_processed_list = []

        for b in range(batch_size):
            clean_sample = clean_np[b:b+1]
            noise_sample = noise_np[b:b+1]
            rir_sample = rir_np[b:b+1] if rir_np is not None else None

            noisy = clean_sample.copy()
            clean_out = clean_sample.copy()

            # Apply effects (simplified version)
            if random.random() < self.config["p_noise"]:
                snr = random.uniform(self.config["snr_min"], self.config["snr_max"])
                noisy, _ = add_noise(noisy, noise_sample, snr=snr, rng=np.random.default_rng())

            if rir_sample is not None and random.random() < self.config["p_reverb"]:
                noisy, clean_out = add_reverberation_v2(
                    clean_out, noisy, rir_sample, self.sample_rate
                )

            if random.random() < self.config["p_clipping"]:
                threshold_db = random.uniform(
                    self.config["clipping_min_db"],
                    self.config["clipping_max_db"]
                )
                noisy = pd.Clipping(threshold_db)(noisy, self.sample_rate)

            if random.random() < self.config["p_bandwidth_limitation"]:
                fs_new = random.choice(self.config["bandwidth_limitation_rates"])
                noisy = bandwidth_limitation(noisy, self.sample_rate, fs_new)

            # Normalize
            scale = 1 / max(
                np.max(np.abs(noisy)),
                np.max(np.abs(clean_out)),
                1e-10
            )
            noisy *= scale
            clean_out *= scale

            degraded_list.append(noisy)
            clean_processed_list.append(clean_out)

        degraded = torch.from_numpy(np.concatenate(degraded_list, axis=0)).to(device)
        clean_processed = torch.from_numpy(np.concatenate(clean_processed_list, axis=0)).to(device)

        return degraded, clean_processed
