# cached_dataset.py
# GPU-optimized dataset with audio caching for MetricGAN+
# Compatible with PyTorch 1.7+

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Optional, Dict, Any, Tuple
from collections import OrderedDict
import threading


class LRUCache:
    """Simple LRU cache implementation."""

    def __init__(self, max_size: int = 10000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[torch.Tensor]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: torch.Tensor):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove oldest item
                    self.cache.popitem(last=False)
                self.cache[key] = value

    def __len__(self):
        return len(self.cache)


class CachedAudioDegradationDataset(Dataset):
    """
    GPU-optimized dataset with pre-loaded audio pools and caching.

    Key optimizations:
    1. Pre-loads noise and RIR pools into RAM at initialization
    2. LRU cache for speech files to avoid repeated disk I/O
    3. Returns raw tensors - degradation happens in training loop on GPU
    4. Uses torchaudio for faster audio loading
    """

    def __init__(
        self,
        speech_list: str,
        noise_list: str,
        rir_list: Optional[str],
        degradation_config: Dict[str, Any],
        seq_len: int = (512 - 1) * 256 + 512,
        sr: int = 16000,
        cache_speech: bool = True,
        max_cache_size: int = 10000,
        preload_noise: bool = True,
        preload_rir: bool = True,
        max_noise_pool: int = 1000,
        max_rir_pool: int = 500,
        silence_threshold: float = -40.0,
        silence_ratio_threshold: float = 0.5,
    ):
        """
        Initialize the cached dataset.

        Args:
            speech_list: Path to the speech .scp file
            noise_list: Path to the noise .scp file
            rir_list: Path to the RIR .scp file or None
            degradation_config: Configuration for audio degradation
            seq_len: Sequence length for audio samples
            sr: Sample rate
            cache_speech: Whether to cache speech files in memory
            max_cache_size: Maximum number of speech files to cache
            preload_noise: Whether to preload noise files into memory
            preload_rir: Whether to preload RIR files into memory
            max_noise_pool: Maximum number of noise files to preload
            max_rir_pool: Maximum number of RIR files to preload
            silence_threshold: dB threshold for silence detection
            silence_ratio_threshold: Maximum allowable silence ratio
        """
        # Parse file lists
        self.speech_list, self.noise_list, self.rir_list = self._parse_scp_files(
            speech_list, noise_list, rir_list
        )

        self.degradation_config = degradation_config
        self.seq_len = int(seq_len * (sr / 44100))  # Adjust for sample rate
        self.sr = sr
        self.silence_threshold = silence_threshold
        self.silence_ratio_threshold = silence_ratio_threshold

        # Caching
        self.cache_speech = cache_speech
        if cache_speech:
            self.speech_cache = LRUCache(max_size=max_cache_size)
        else:
            self.speech_cache = None

        # Pre-load noise pool
        self.noise_pool = None
        if preload_noise and self.noise_list:
            print(f"Pre-loading noise pool (up to {max_noise_pool} files)...")
            self.noise_pool = self._preload_audio_pool(
                self.noise_list,
                max_items=max_noise_pool
            )
            print(f"Loaded {len(self.noise_pool)} noise files into memory")

        # Pre-load RIR pool
        self.rir_pool = None
        if preload_rir and self.rir_list:
            print(f"Pre-loading RIR pool (up to {max_rir_pool} files)...")
            self.rir_pool = self._preload_audio_pool(
                self.rir_list,
                max_items=max_rir_pool
            )
            print(f"Loaded {len(self.rir_pool)} RIR files into memory")

    def _parse_scp_files(
        self,
        speech_file: str,
        noise_file: str,
        rir_file: Optional[str]
    ) -> Tuple[List[str], List[str], Optional[List[str]]]:
        """Parse .scp files to get audio file paths."""

        def parse_file(file_path: str) -> List[str]:
            paths = []
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=2)
                    if len(parts) >= 3:
                        paths.append(parts[2])
                    elif len(parts) == 1:
                        # Just a path
                        paths.append(parts[0])
            return paths

        speech_paths = parse_file(speech_file)
        noise_paths = parse_file(noise_file)
        rir_paths = parse_file(rir_file) if rir_file else None

        return speech_paths, noise_paths, rir_paths

    def _load_audio(self, path: str) -> Optional[torch.Tensor]:
        """Load audio file using torchaudio."""
        try:
            audio, sr = torchaudio.load(path)

            # Resample if needed
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sr
                )
                audio = resampler(audio)

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Squeeze to 1D
            audio = audio.squeeze(0)

            return audio
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None

    def _preload_audio_pool(
        self,
        file_list: List[str],
        max_items: int = 1000
    ) -> List[torch.Tensor]:
        """Pre-load audio files into memory."""
        pool = []
        selected = file_list[:max_items] if len(file_list) > max_items else file_list

        for path in selected:
            audio = self._load_audio(path)
            if audio is not None:
                pool.append(audio)

        return pool

    def _get_cached_audio(self, path: str) -> Optional[torch.Tensor]:
        """Get audio with optional LRU caching."""
        if self.speech_cache is not None:
            cached = self.speech_cache.get(path)
            if cached is not None:
                return cached.clone()

        audio = self._load_audio(path)

        if audio is not None and self.speech_cache is not None:
            self.speech_cache.put(path, audio)

        return audio

    def _pad_or_truncate(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad or truncate audio to seq_len."""
        length = audio.shape[-1]

        if length < self.seq_len:
            # Repeat to fill
            repeats = (self.seq_len // length) + 1
            audio = audio.repeat(repeats)[:self.seq_len]
        elif length > self.seq_len:
            # Random crop
            offset = random.randint(0, length - self.seq_len)
            audio = audio[offset:offset + self.seq_len]

        return audio

    def __len__(self) -> int:
        return len(self.speech_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample for GPU processing.

        Returns a dictionary with:
        - clean: clean speech tensor
        - noise: noise tensor
        - rir: RIR tensor (or None)
        - has_rir: whether RIR is available
        - idx: sample index
        """
        # Get clean speech
        speech_path = self.speech_list[idx]
        clean = self._get_cached_audio(speech_path)

        if clean is None:
            # Return dummy data if loading fails
            clean = torch.zeros(self.seq_len)

        clean = self._pad_or_truncate(clean)

        # Get noise
        if self.noise_pool:
            noise = random.choice(self.noise_pool).clone()
        else:
            noise_path = random.choice(self.noise_list)
            noise = self._load_audio(noise_path)
            if noise is None:
                noise = torch.zeros(self.seq_len)

        noise = self._pad_or_truncate(noise)

        # Get RIR
        has_rir = False
        rir = None
        if self.rir_pool:
            rir = random.choice(self.rir_pool).clone()
            has_rir = True
        elif self.rir_list:
            rir_path = random.choice(self.rir_list)
            rir = self._load_audio(rir_path)
            has_rir = rir is not None

        return {
            'clean': clean,
            'noise': noise,
            'rir': rir,
            'has_rir': has_rir,
            'idx': idx
        }


def gpu_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for GPU degradation pipeline.

    Handles variable-length RIRs by padding to max length.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary ready for GPU processing
    """
    # Stack clean and noise (same length)
    clean = torch.stack([item['clean'] for item in batch])
    noise = torch.stack([item['noise'] for item in batch])

    # Handle RIRs (variable length)
    rirs = [item['rir'] for item in batch]
    has_rir = torch.tensor([item['has_rir'] for item in batch], dtype=torch.bool)

    # Find samples with RIR
    rir_present = [r for r in rirs if r is not None]

    if rir_present:
        # Pad RIRs to max length
        max_rir_len = max(r.shape[-1] for r in rir_present)
        padded_rirs = torch.zeros(len(batch), max_rir_len)

        for i, rir in enumerate(rirs):
            if rir is not None:
                rir_len = rir.shape[-1]
                padded_rirs[i, :rir_len] = rir
    else:
        padded_rirs = None

    indices = torch.tensor([item['idx'] for item in batch])

    return {
        'clean': clean,
        'noise': noise,
        'rir': padded_rirs,
        'rir_mask': has_rir,
        'indices': indices
    }


def create_gpu_dataloader(
    speech_list: str,
    noise_list: str,
    rir_list: Optional[str],
    degradation_config: Dict[str, Any],
    batch_size: int = 128,
    num_workers: int = 4,
    seq_len: int = (512 - 1) * 256 + 512,
    sr: int = 16000,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create an optimized DataLoader for GPU degradation.

    Args:
        speech_list: Path to speech .scp file
        noise_list: Path to noise .scp file
        rir_list: Path to RIR .scp file or None
        degradation_config: Degradation configuration
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        seq_len: Sequence length
        sr: Sample rate
        shuffle: Whether to shuffle
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        Configured DataLoader
    """
    dataset = CachedAudioDegradationDataset(
        speech_list=speech_list,
        noise_list=noise_list,
        rir_list=rir_list,
        degradation_config=degradation_config,
        seq_len=seq_len,
        sr=sr,
        **dataset_kwargs
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=gpu_collate_fn,
        pin_memory=True,           # Faster CPU-to-GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Avoid worker restart
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
        drop_last=True
    )

    return dataloader


# Backward compatibility: alias for existing code
class GPUAudioDegradationDataset(CachedAudioDegradationDataset):
    """Alias for backward compatibility."""
    pass
