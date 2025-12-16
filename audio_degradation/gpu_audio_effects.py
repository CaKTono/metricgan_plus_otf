# gpu_audio_effects.py
# GPU-accelerated audio effects for MetricGAN+ on-the-fly degradation
# Compatible with PyTorch 1.7+

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
import numpy as np
from typing import Optional, Tuple, List


class GPUNoiseMixer(nn.Module):
    """
    GPU-accelerated noise mixing with SNR control.
    Replaces add_noise() from audio_degradation_pipeline.py
    """
    def __init__(self, use_vad: bool = True, vad_threshold: float = 0.01):
        super().__init__()
        self.use_vad = use_vad
        self.vad_threshold = vad_threshold

    def detect_non_silence(self, x: torch.Tensor, frame_length: int = 1024) -> torch.Tensor:
        """Simple power-based voice activity detection on GPU."""
        if x.shape[-1] < frame_length:
            return torch.ones_like(x, dtype=torch.bool)

        # Compute frame power
        num_frames = x.shape[-1] // frame_length
        truncated = x[..., :num_frames * frame_length]
        frames = truncated.view(*x.shape[:-1], num_frames, frame_length)
        power = (frames ** 2).mean(dim=-1)

        # Detect non-silence
        mean_power = power.mean(dim=-1, keepdim=True)
        mean_power = torch.clamp(mean_power, min=1e-10)
        detect_frames = power / mean_power > self.vad_threshold

        # Expand back to original length
        detect = detect_frames.unsqueeze(-1).expand(*detect_frames.shape, frame_length)
        detect = detect.reshape(*x.shape[:-1], -1)

        # Pad to original length
        if detect.shape[-1] < x.shape[-1]:
            pad_size = x.shape[-1] - detect.shape[-1]
            detect = F.pad(detect.float(), (0, pad_size), value=1.0).bool()

        return detect

    def forward(
        self,
        speech: torch.Tensor,
        noise: torch.Tensor,
        snr_db: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix speech with noise at specified SNR.

        Args:
            speech: (batch, time) clean speech
            noise: (batch, time) noise signal
            snr_db: (batch,) or scalar SNR in dB

        Returns:
            noisy_speech: (batch, time) mixed signal
            scaled_noise: (batch, time) scaled noise
        """
        # Ensure noise matches speech length
        if noise.shape[-1] < speech.shape[-1]:
            # Repeat noise to match length
            repeats = (speech.shape[-1] // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)[..., :speech.shape[-1]]
        elif noise.shape[-1] > speech.shape[-1]:
            # Random crop noise
            max_offset = noise.shape[-1] - speech.shape[-1]
            offset = torch.randint(0, max_offset + 1, (1,)).item()
            noise = noise[..., offset:offset + speech.shape[-1]]

        # Compute power
        if self.use_vad:
            speech_mask = self.detect_non_silence(speech)
            noise_mask = self.detect_non_silence(noise)

            # Masked power computation
            speech_power = (speech ** 2 * speech_mask.float()).sum(dim=-1) / (speech_mask.sum(dim=-1).float() + 1e-10)
            noise_power = (noise ** 2 * noise_mask.float()).sum(dim=-1) / (noise_mask.sum(dim=-1).float() + 1e-10)
        else:
            speech_power = (speech ** 2).mean(dim=-1)
            noise_power = (noise ** 2).mean(dim=-1)

        # Compute scale factor
        if snr_db.dim() == 0:
            snr_db = snr_db.unsqueeze(0).expand(speech.shape[0])

        scale = 10 ** (-snr_db / 20) * torch.sqrt(speech_power / (noise_power + 1e-10))
        scale = scale.unsqueeze(-1)

        scaled_noise = scale * noise
        noisy_speech = speech + scaled_noise

        return noisy_speech, scaled_noise


class GPUClipping(nn.Module):
    """
    GPU-accelerated audio clipping.
    Replaces pedalboard.Clipping
    """
    def __init__(self):
        super().__init__()

    def forward(self, audio: torch.Tensor, threshold_db: torch.Tensor) -> torch.Tensor:
        """
        Apply clipping distortion.

        Args:
            audio: (batch, time) input audio
            threshold_db: (batch,) or scalar clipping threshold in dB

        Returns:
            clipped: (batch, time) clipped audio
        """
        # Convert dB to linear threshold
        if threshold_db.dim() == 0:
            threshold_db = threshold_db.unsqueeze(0).expand(audio.shape[0])

        threshold = 10 ** (threshold_db / 20)
        threshold = threshold.unsqueeze(-1)

        return torch.clamp(audio, -threshold, threshold)


class GPUBitcrush(nn.Module):
    """
    GPU-accelerated bitcrusher effect.
    Replaces pedalboard.Bitcrush
    """
    def __init__(self):
        super().__init__()

    def forward(self, audio: torch.Tensor, bit_depth: torch.Tensor) -> torch.Tensor:
        """
        Apply bitcrushing (quantization).

        Args:
            audio: (batch, time) input audio, normalized to [-1, 1]
            bit_depth: (batch,) or scalar bit depth (e.g., 8 for 8-bit)

        Returns:
            crushed: (batch, time) bitcrushed audio
        """
        if bit_depth.dim() == 0:
            bit_depth = bit_depth.unsqueeze(0).expand(audio.shape[0])

        # Quantization levels
        levels = (2 ** bit_depth).unsqueeze(-1)

        # Quantize: scale to [0, levels], round, scale back
        scaled = (audio + 1) / 2 * levels  # Map [-1, 1] to [0, levels]
        quantized = torch.round(scaled)
        crushed = quantized / levels * 2 - 1  # Map back to [-1, 1]

        return crushed


class GPUDistortion(nn.Module):
    """
    GPU-accelerated distortion effect using soft clipping (tanh).
    Replaces pedalboard.Distortion
    """
    def __init__(self):
        super().__init__()

    def forward(self, audio: torch.Tensor, drive_db: torch.Tensor) -> torch.Tensor:
        """
        Apply distortion using tanh waveshaping.

        Args:
            audio: (batch, time) input audio
            drive_db: (batch,) or scalar drive amount in dB

        Returns:
            distorted: (batch, time) distorted audio
        """
        if drive_db.dim() == 0:
            drive_db = drive_db.unsqueeze(0).expand(audio.shape[0])

        gain = 10 ** (drive_db / 20)
        gain = gain.unsqueeze(-1)

        # Apply tanh soft clipping
        distorted = torch.tanh(audio * gain)

        return distorted


class GPUPacketLoss(nn.Module):
    """
    GPU-accelerated packet loss simulation.
    Randomly zeros out segments of audio.
    """
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        audio: torch.Tensor,
        replace_ratio: torch.Tensor,
        min_length_sec: float = 0.05,
        max_length_sec: float = 0.1
    ) -> torch.Tensor:
        """
        Apply packet loss simulation.

        Args:
            audio: (batch, time) input audio
            replace_ratio: (batch,) ratio of audio to zero out
            min_length_sec: minimum segment length in seconds
            max_length_sec: maximum segment length in seconds

        Returns:
            damaged: (batch, time) audio with packet loss
        """
        batch_size, audio_len = audio.shape
        damaged = audio.clone()

        min_samples = int(min_length_sec * self.sample_rate)
        max_samples = int(max_length_sec * self.sample_rate)

        for b in range(batch_size):
            total_to_replace = int(audio_len * replace_ratio[b].item())
            replaced = 0

            while replaced < total_to_replace:
                # Random segment length
                seg_len = torch.randint(min_samples, max_samples + 1, (1,)).item()
                seg_len = min(seg_len, total_to_replace - replaced)

                # Random start position
                start = torch.randint(0, audio_len - seg_len + 1, (1,)).item()

                # Zero out segment
                damaged[b, start:start + seg_len] = 0
                replaced += seg_len

        return damaged


class GPUReverb(nn.Module):
    """
    GPU-accelerated reverb using FFT convolution.
    Replaces scipy.signal.fftconvolve
    """
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

    def fft_convolve(self, signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        FFT-based convolution on GPU.

        Args:
            signal: (batch, time) input signal
            kernel: (batch, kernel_len) or (kernel_len,) convolution kernel

        Returns:
            convolved: (batch, time) convolved signal (same length as input)
        """
        signal_len = signal.shape[-1]
        kernel_len = kernel.shape[-1]

        # Output length for full convolution
        n = signal_len + kernel_len - 1
        # Pad to power of 2 for efficient FFT
        n_fft = 2 ** int(math.ceil(math.log2(n)))

        # FFT
        signal_fft = torch.fft.rfft(signal, n=n_fft, dim=-1)

        # Handle kernel broadcasting
        if kernel.dim() == 1:
            kernel = kernel.unsqueeze(0).expand(signal.shape[0], -1)
        kernel_fft = torch.fft.rfft(kernel, n=n_fft, dim=-1)

        # Multiply in frequency domain
        result_fft = signal_fft * kernel_fft

        # Inverse FFT
        result = torch.fft.irfft(result_fft, n=n_fft, dim=-1)

        # Trim to original signal length
        return result[..., :signal_len]

    def forward(
        self,
        speech: torch.Tensor,
        noisy_speech: torch.Tensor,
        rir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply reverberation with early reflection extraction.
        Matches add_reverberation_v2() from CPU pipeline.

        Args:
            speech: (batch, time) clean speech
            noisy_speech: (batch, time) noisy speech (possibly with other effects)
            rir: (batch, rir_len) room impulse response

        Returns:
            reverberant_speech: (batch, time) reverberant version
            reverberant_early: (batch, time) early reflections only
        """
        batch_size = speech.shape[0]
        wav_len = speech.shape[-1]

        # Find delay (peak) in RIR
        delay_idx = torch.argmax(torch.abs(rir), dim=-1)

        # Early RIR extraction parameters
        delay_before = int(0.001 * self.sample_rate)  # 1ms before peak
        delay_after = int(0.005 * self.sample_rate)   # 5ms after peak

        reverberant_speech = torch.zeros_like(speech)
        reverberant_early = torch.zeros_like(speech)

        for b in range(batch_size):
            idx = delay_idx[b].item()
            idx_start = max(0, idx - delay_before)
            idx_end = idx + delay_after

            # Extract early RIR
            early_rir = rir[b, idx_start:idx_end]

            # Convolve
            rev_early = self.fft_convolve(speech[b:b+1], early_rir.unsqueeze(0))
            rev_full = self.fft_convolve(noisy_speech[b:b+1], rir[b:b+1])

            # Trim and align
            rev_full = rev_full[..., idx_start:idx_start + wav_len]
            rev_early = rev_early[..., :wav_len]

            # Normalize
            scale = torch.max(torch.abs(rev_full))
            if scale > 0:
                scale = 0.5 / scale
            else:
                scale = 1.0

            reverberant_speech[b] = rev_full.squeeze(0) * scale
            reverberant_early[b] = rev_early.squeeze(0) * scale

        return reverberant_speech, reverberant_early


class GPUBandwidthLimitation(nn.Module):
    """
    GPU-accelerated bandwidth limitation using torchaudio resampling.
    Replaces librosa.resample
    """
    def __init__(self, orig_sr: int = 16000):
        super().__init__()
        self.orig_sr = orig_sr
        self.target_rates = [4000, 8000, 16000, 22050, 32000]

        # Pre-create resamplers for common rates
        self.down_samplers = nn.ModuleDict()
        self.up_samplers = nn.ModuleDict()

        for rate in self.target_rates:
            if rate != orig_sr:
                self.down_samplers[str(rate)] = torchaudio.transforms.Resample(
                    orig_freq=orig_sr,
                    new_freq=rate,
                    resampling_method='sinc_interpolation'
                )
                self.up_samplers[str(rate)] = torchaudio.transforms.Resample(
                    orig_freq=rate,
                    new_freq=orig_sr,
                    resampling_method='sinc_interpolation'
                )

    def forward(self, audio: torch.Tensor, target_rate: int) -> torch.Tensor:
        """
        Apply bandwidth limitation by downsampling then upsampling.

        Args:
            audio: (batch, time) input audio
            target_rate: target sample rate for limitation

        Returns:
            limited: (batch, time) bandwidth-limited audio
        """
        if target_rate == self.orig_sr:
            return audio

        original_len = audio.shape[-1]

        # Get resamplers
        down = self.down_samplers[str(target_rate)]
        up = self.up_samplers[str(target_rate)]

        # Move resamplers to same device as audio
        down = down.to(audio.device)
        up = up.to(audio.device)

        # Downsample then upsample
        downsampled = down(audio)
        upsampled = up(downsampled)

        # Ensure same length
        if upsampled.shape[-1] > original_len:
            upsampled = upsampled[..., :original_len]
        elif upsampled.shape[-1] < original_len:
            upsampled = F.pad(upsampled, (0, original_len - upsampled.shape[-1]))

        return upsampled


class GPUChorus(nn.Module):
    """
    GPU-accelerated chorus effect using modulated delay line.
    Replaces pedalboard.Chorus
    """
    def __init__(self, sample_rate: int = 16000, max_delay_ms: float = 50.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)

    def forward(
        self,
        audio: torch.Tensor,
        rate_hz: torch.Tensor,
        depth: torch.Tensor,
        centre_delay_ms: torch.Tensor,
        feedback: torch.Tensor,
        mix: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply chorus effect with modulated delay.

        Args:
            audio: (batch, time) input audio
            rate_hz: (batch,) LFO rate in Hz
            depth: (batch,) modulation depth (0-1)
            centre_delay_ms: (batch,) center delay in milliseconds
            feedback: (batch,) feedback amount (-1 to 1)
            mix: (batch,) wet/dry mix (0-1)

        Returns:
            chorused: (batch, time) audio with chorus effect
        """
        batch_size, audio_len = audio.shape
        device = audio.device

        # Ensure parameters are proper shape
        if rate_hz.dim() == 0:
            rate_hz = rate_hz.unsqueeze(0).expand(batch_size)
        if depth.dim() == 0:
            depth = depth.unsqueeze(0).expand(batch_size)
        if centre_delay_ms.dim() == 0:
            centre_delay_ms = centre_delay_ms.unsqueeze(0).expand(batch_size)
        if feedback.dim() == 0:
            feedback = feedback.unsqueeze(0).expand(batch_size)
        if mix.dim() == 0:
            mix = mix.unsqueeze(0).expand(batch_size)

        # Generate time axis
        t = torch.arange(audio_len, device=device, dtype=audio.dtype) / self.sample_rate

        output = torch.zeros_like(audio)

        for b in range(batch_size):
            # Generate LFO (sinusoidal modulation)
            lfo = torch.sin(2 * math.pi * rate_hz[b] * t)

            # Compute delay in samples (time-varying)
            centre_delay_samples = centre_delay_ms[b] * self.sample_rate / 1000
            delay_samples = centre_delay_samples + lfo * depth[b] * centre_delay_samples
            delay_samples = torch.clamp(delay_samples, min=1, max=self.max_delay_samples)

            # Create delayed signal using linear interpolation
            delayed = torch.zeros(audio_len, device=device, dtype=audio.dtype)

            for i in range(audio_len):
                delay = delay_samples[i].item()
                read_pos = i - delay

                if read_pos >= 0:
                    idx_low = int(read_pos)
                    idx_high = min(idx_low + 1, audio_len - 1)
                    frac = read_pos - idx_low

                    delayed[i] = audio[b, idx_low] * (1 - frac) + audio[b, idx_high] * frac

            # Mix dry and wet
            output[b] = audio[b] * (1 - mix[b]) + delayed * mix[b]

        return output


class GPUParametricEQ(nn.Module):
    """
    GPU-accelerated parametric EQ using biquad filters.
    Replaces pedalboard_equalizer() function.
    """
    def __init__(self, sample_rate: int = 16000, num_bands: int = 10):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_bands = num_bands

    def compute_peaking_coeffs(
        self,
        fc: float,
        gain_db: float,
        q: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute biquad coefficients for peaking EQ filter."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * math.pi * fc / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        # Normalize
        b = torch.tensor([b0/a0, b1/a0, b2/a0])
        a = torch.tensor([a1/a0, a2/a0])

        return b, a

    def compute_lowshelf_coeffs(
        self,
        fc: float,
        gain_db: float,
        q: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute biquad coefficients for low-shelf filter."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * math.pi * fc / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2 * q)

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha

        b = torch.tensor([b0/a0, b1/a0, b2/a0])
        a = torch.tensor([a1/a0, a2/a0])

        return b, a

    def compute_highshelf_coeffs(
        self,
        fc: float,
        gain_db: float,
        q: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute biquad coefficients for high-shelf filter."""
        A = 10 ** (gain_db / 40)
        w0 = 2 * math.pi * fc / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2 * q)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha

        b = torch.tensor([b0/a0, b1/a0, b2/a0])
        a = torch.tensor([a1/a0, a2/a0])

        return b, a

    def apply_biquad(
        self,
        audio: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        """Apply biquad filter using torchaudio."""
        b = b.to(audio.device)
        a = a.to(audio.device)

        # torchaudio.functional.biquad expects: (waveform, b0, b1, b2, a0, a1, a2)
        # where a0 is always 1 (normalized)
        return torchaudio.functional.biquad(
            audio,
            b[0], b[1], b[2],
            1.0, a[0], a[1]
        )

    def forward(
        self,
        audio: torch.Tensor,
        center_freqs: Optional[List[float]] = None,
        gains_db: Optional[List[float]] = None,
        q_values: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Apply parametric EQ.

        Args:
            audio: (batch, time) input audio
            center_freqs: list of center frequencies for each band
            gains_db: list of gain values in dB for each band
            q_values: list of Q values for each band

        Returns:
            equalized: (batch, time) equalized audio
        """
        # Generate random parameters if not provided (matching original behavior)
        if center_freqs is None:
            center_freqs = [np.random.uniform(1, 12000) for _ in range(self.num_bands)]
        if gains_db is None:
            gains_db = [np.random.uniform(-12, 12) for _ in range(self.num_bands)]
        if q_values is None:
            q_min, q_max = 2, 5
            q_values = [q_min * ((q_max / q_min) ** np.random.uniform(0, 1)) for _ in range(self.num_bands)]

        # Apply filters
        output = audio

        # Low-shelf filter (first band)
        b, a = self.compute_lowshelf_coeffs(center_freqs[0], gains_db[0], q_values[0])
        output = self.apply_biquad(output, b, a)

        # Peaking filters (middle bands)
        for i in range(1, self.num_bands - 1):
            b, a = self.compute_peaking_coeffs(center_freqs[i], gains_db[i], q_values[i])
            output = self.apply_biquad(output, b, a)

        # High-shelf filter (last band)
        b, a = self.compute_highshelf_coeffs(
            center_freqs[self.num_bands - 1],
            gains_db[self.num_bands - 1],
            q_values[self.num_bands - 1]
        )
        output = self.apply_biquad(output, b, a)

        return output
