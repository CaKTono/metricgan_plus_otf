# MetricGAN+ On-The-Fly (GPU-Accelerated)

A GPU-accelerated implementation of MetricGAN+ for audio enhancement with on-the-fly audio degradation.

Based on the paper: **["MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement"](https://arxiv.org/abs/2104.03538)** (Interspeech 2021) by Szu-Wei Fu et al.

## Supported Audio Types

- **Speech Enhancement**: Standard datasets like VoiceBank-DEMAND, DNS Challenge, etc.
- **Singing Voice Enhancement**: Singing datasets like OpenCpop, ACESinger, OpenSinger, etc.
- **General Audio**: Any mono audio at 16kHz sample rate

## Features

- **GPU-Accelerated Degradation**: Faster than CPU-based processing
- **On-The-Fly Augmentation**: Audio degradation applied during training, not pre-computed
- **Multiple Degradation Effects**: Noise, reverb, clipping, bandwidth limitation, EQ, chorus, distortion, bitcrush, packet loss
- **Memory Caching**: Pre-loaded noise/RIR pools and LRU cache for audio files
- **Hybrid Mode**: Automatic CPU fallback if GPU processing fails
- **Multi-GPU Support**: DataParallel for distributed training

## Installation

```bash
# Clone the repository
git clone https://github.com/CaKTono/metricgan_plus_otf.git
cd metricgan_plus_otf

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.7.1
- torchaudio >= 0.7.2
- CUDA-capable GPU (recommended)

## Dataset Format

This implementation uses `.scp` (script) files to reference audio data. You need three types of files:

### SCP File Format

Each `.scp` file contains one audio file path per line with the format:
```
<utterance_id> <speaker_id> <full_path_to_audio>
```

Or simply:
```
<full_path_to_audio>
```

### Required SCP Files

| File | Description | Example |
|------|-------------|---------|
| `speech.scp` | Clean speech/singing audio paths | `utt001 spk01 /data/clean/audio001.wav` |
| `noise.scp` | Background noise audio paths | `noise001 env01 /data/noise/noise001.wav` |
| `rir.scp` | Room Impulse Response paths | `rir001 room01 /data/rir/rir001.wav` |

### Example SCP Files

**speech.scp:**
```
utt_0001 singer_01 /path/to/dataset/clean/singer_01/song_001.wav
utt_0002 singer_01 /path/to/dataset/clean/singer_01/song_002.wav
utt_0003 singer_02 /path/to/dataset/clean/singer_02/song_001.wav
```

**noise.scp:**
```
noise_0001 env_office /path/to/noise/office_ambience.wav
noise_0002 env_street /path/to/noise/street_traffic.wav
noise_0003 env_cafe /path/to/noise/cafe_chatter.wav
```

**rir.scp:**
```
rir_0001 room_small /path/to/rir/small_room.wav
rir_0002 room_large /path/to/rir/large_hall.wav
rir_0003 room_studio /path/to/rir/recording_studio.wav
```

### Directory Structure

```
your_data/
├── scp/
│   ├── speech_train.scp
│   ├── speech_test.scp
│   ├── noise.scp
│   └── rir.scp
├── clean/
│   ├── singer_01/
│   │   ├── song_001.wav
│   │   └── song_002.wav
│   └── singer_02/
│       └── song_001.wav
├── noise/
│   ├── office_ambience.wav
│   └── street_traffic.wav
└── rir/
    ├── small_room.wav
    └── large_hall.wav
```

### Audio Requirements

| Property | Requirement |
|----------|-------------|
| Sample Rate | 16 kHz (will be resampled if different) |
| Channels | Mono (will be converted if stereo) |
| Format | WAV (recommended), FLAC, MP3 |
| Bit Depth | 16-bit or 32-bit float |

## Usage

### Training with GPU Degradation (Recommended)

```bash
python main_on_the_fly.py \
    --exp_name my_experiment \
    --use_gpu_degradation True \
    --epochs 750 \
    --batch_size 128 \
    --lr 5e-4 \
    --target_metric pesq
```

### Training with CPU Degradation (Fallback)

```bash
python main_on_the_fly.py \
    --exp_name my_experiment \
    --use_gpu_degradation False \
    --epochs 750 \
    --batch_size 128
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp_name` | `singing_new` | Experiment name |
| `--use_gpu_degradation` | `True` | Use GPU-accelerated degradation |
| `--fallback_cpu` | `False` | Fallback to CPU on GPU errors |
| `--epochs` | `750` | Number of training epochs |
| `--batch_size` | `128` | Batch size |
| `--lr` | `5e-4` | Learning rate |
| `--target_metric` | `pesq` | Target metric (pesq/csig/cbak/covl) |
| `--num_workers` | `12` | DataLoader workers |
| `--cache_speech` | `True` | Cache speech files in RAM |
| `--preload_noise` | `True` | Preload noise pool |
| `--preload_rir` | `True` | Preload RIR pool |
| `--max_noise_pool` | `1000` | Max noise files to preload |
| `--max_rir_pool` | `500` | Max RIR files to preload |

### Inference

```bash
python inference.py \
    --weight_path results/my_experiment/model/ \
    --weight_file best_model.pth
```

## Degradation Configuration

The degradation pipeline applies the following effects with configurable probabilities:

```python
degradation_config = {
    # Noise addition
    "p_noise": 0.9,           # Probability of adding noise
    "snr_min": -5,            # Minimum SNR (dB)
    "snr_max": 20,            # Maximum SNR (dB)

    # Reverberation
    "p_reverb": 0.25,         # Probability of adding reverb

    # Clipping
    "p_clipping": 0.25,       # Probability of clipping
    "clipping_min_db": -20,   # Minimum clipping threshold (dB)
    "clipping_max_db": 0,     # Maximum clipping threshold (dB)

    # Bandwidth limitation
    "p_bandwidth_limitation": 0.5,
    "bandwidth_limitation_rates": [4000, 8000, 16000, 22050, 32000],

    # Bitcrush
    "p_bitcrush": 0.0,
    "bitcrush_min_bits": 3,
    "bitcrush_max_bits": 8,

    # Chorus
    "p_chorus": 0.0,

    # Distortion
    "p_distortion": 0.0,
    "distortion_min_db": 5,
    "distortion_max_db": 20,

    # EQ
    "p_eq": 0.0,

    # Packet loss
    "p_pl": 0.0,
    "pl_min_ratio": 0.05,
    "pl_max_ratio": 0.1,
}
```

## Project Structure

```
metricgan_plus_otf/
├── main_on_the_fly.py      # Main entry point
├── train.py                # CPU-based trainer
├── train_gpu.py            # GPU-accelerated trainer
├── model.py                # Generator and Discriminator models
├── dataloader.py           # DataLoader utilities
├── signal_processing.py    # STFT/iSTFT operations
├── inference.py            # Inference script
├── requirements.txt        # Python dependencies
├── audio_degradation/      # Degradation modules
│   ├── __init__.py
│   ├── gpu_audio_effects.py        # GPU effect implementations
│   ├── gpu_degradation_pipeline.py # Main GPU pipeline
│   ├── cached_dataset.py           # Cached dataset with preloading
│   ├── audio_degradation_pipeline.py # Original CPU pipeline
│   └── degradation_dataset.py      # Original CPU dataset
├── metric_functions/       # Audio quality metrics
│   ├── compute_metric.py
│   ├── get_metric_scores.py
│   └── metric_helper.py
├── configs/                # Configuration files
└── examples/               # Example scripts
    └── create_scp.py
```

## Creating SCP Files

Use the provided utility script to generate SCP files from your dataset:

```bash
python examples/create_scp.py \
    --audio_dir /path/to/audio/files \
    --output_scp /path/to/output.scp \
    --recursive
```

## Citation

If you use this code, please cite the original MetricGAN+ paper:

```bibtex
@inproceedings{fu2021metricgan+,
  title={MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement},
  author={Fu, Szu-Wei and Yu, Cheng and Hsieh, Tsun-An and Plantinga, Peter and Ravanelli, Mirco and Lu, Xugang and Tsao, Yu},
  booktitle={Proc. Interspeech},
  year={2021}
}
```

## Acknowledgments

- Original MetricGAN+ implementation by [SpeechBrain](https://github.com/speechbrain/speechbrain)
- PyTorch implementation by [wooseok-shin](https://github.com/wooseok-shin/MetricGAN-plus-pytorch)
- Audio degradation pipeline inspired by [AnyEnhance](https://github.com/viewfinder-annn/AnyEnhance-v1)

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.
