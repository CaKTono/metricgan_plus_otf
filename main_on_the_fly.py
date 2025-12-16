import random
import shutil
import argparse
import warnings
import numpy as np
import torch
import os
from pathlib import Path
from os.path import join as opj
from train import Trainer
from train_gpu import GPUTrainer  # GPU-enabled trainer
from metric_functions.get_metric_scores import get_pesq_parallel, get_csig_parallel, get_cbak_parallel, get_covl_parallel
from audio_degradation.degradation_dataset import AudioDegradationDataset
from audio_degradation.cached_dataset import CachedAudioDegradationDataset, gpu_collate_fn
from torch.utils.data import DataLoader, random_split
warnings.filterwarnings('ignore')

# Check if CUDA_VISIBLE_DEVICES is set
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    # Default to using the first GPU if CUDA_VISIBLE_DEVICES is not set
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("CUDA_VISIBLE_DEVICES not set, defaulting to GPU 0")
else:
    print(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible_devices}")

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--exp_name', type=str, default='singing_new', help='name of the experiment')
    parser.add_argument('--device', type=str, default='cuda', help='Gpu device')
    parser.add_argument('--output_path', type=str, default='/mnt/workspace/calvin/results_singing_new', help='Model and Log path')
    parser.add_argument('--base_path', type=str, default='/mnt/workspace/calvin/generated_dataset/singing_voice', help='Data base path')
    parser.add_argument('--target_metric', type=str, default='pesq', help='pesq or csig or cbak or covl')
    parser.add_argument('--epochs', type=int, default=750)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_of_sampling', type=int, default=1280)
    parser.add_argument('--num_of_val_sample', type=int, default=6903)
    parser.add_argument('--hist_portion', type=float, default=0.2, help='history portion of replay buffer')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval_per_epoch', type=int, default=1)
    parser.add_argument('--skip_val_epoch', type=int, default=100, help='skip early epoch evaluation')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--causal', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--val_speaker', type=str, nargs='*', default=[], help='select validation speaker (e.g., p226, p227, ..., etc)')

    # GPU degradation options
    parser.add_argument('--use_gpu_degradation', type=bool, default=True, help='Use GPU-accelerated degradation pipeline')
    parser.add_argument('--fallback_cpu', type=bool, default=False, help='Fallback to CPU if GPU degradation fails')
    parser.add_argument('--cache_speech', type=bool, default=True, help='Cache speech files in memory')
    parser.add_argument('--max_cache_size', type=int, default=10000, help='Maximum speech files to cache')
    parser.add_argument('--preload_noise', type=bool, default=True, help='Preload noise files into memory')
    parser.add_argument('--preload_rir', type=bool, default=True, help='Preload RIR files into memory')
    parser.add_argument('--max_noise_pool', type=int, default=1000, help='Maximum noise files to preload')
    parser.add_argument('--max_rir_pool', type=int, default=500, help='Maximum RIR files to preload')

    args = parser.parse_args()
    setup_seed(args.seed)

    # Define paths to combined .scp files
    speech_list = "/mnt/workspace/calvin/data/singing_train.scp"
    noise_list = "/mnt/workspace/calvin/data/noise_train.scp"
    rir_list = "/mnt/workspace/calvin/data/rir_train.scp"

    # Degradation configurations
    degradation_config = {
        "p_noise": 0.9,
        "snr_min": -5,
        "snr_max": 20,
        "p_reverb": 0.25,

        "p_clipping": 0.25,
        "clipping_min_db": -20,
        "clipping_max_db": 0,

        "p_bandwidth_limitation": 0.5,
        "bandwidth_limitation_rates": [
            4000,
            8000,
            16000,
            22050,
            32000
        ],
        "bandwidth_limitation_methods": [
            "kaiser_best",
            "kaiser_fast",
            "scipy",
            "polyphase"
        ],

        # Apply bitcrush
        "p_bitcrush": 0.0,
        "bitcrush_min_bits": 3,
        "bitcrush_max_bits": 8,

        # Add chorus
        "p_chorus": 0.0,
        "rate_hz": 1.0,
        "depth": 0.25,
        "centre_delay_ms": 7.0,
        "feedback": 0.0,
        "chorus_mix": 0.5,

        # Add distortion
        "p_distortion": 0.0,
        "distortion_min_db": 5,
        "distortion_max_db": 20,

        # EQ
        "p_eq": 0.0,
        "eq_min_times": 1,
        "eq_max_times": 3,
        "eq_min_length": 0.5,
        "eq_max_length": 1,

        # package loss
        "p_pl": 0.0,
        "pl_min_ratio": 0.05,
        "pl_max_ratio": 0.1,
        "pl_min_length": 0.05,
        "pl_max_length": 0.1
    }
    

    # Initialize Dataset based on GPU degradation flag
    if args.use_gpu_degradation:
        print("Using GPU-accelerated degradation pipeline")
        combined_dataset = CachedAudioDegradationDataset(
            speech_list=speech_list,
            noise_list=noise_list,
            rir_list=rir_list,
            degradation_config=degradation_config,
            seq_len=512*256,
            sr=16000,
            cache_speech=args.cache_speech,
            max_cache_size=args.max_cache_size,
            preload_noise=args.preload_noise,
            preload_rir=args.preload_rir,
            max_noise_pool=args.max_noise_pool,
            max_rir_pool=args.max_rir_pool,
        )
    else:
        print("Using CPU-based degradation pipeline")
        combined_dataset = AudioDegradationDataset(
            speech_list=speech_list,
            noise_list=noise_list,
            rir_list=rir_list,
            degradation_config=degradation_config,
            seq_len=512*256,
            sr=16000,
            device=args.device
        )

    # Debug: Check dataset size
    print(f"Total dataset size: {len(combined_dataset)}")

    # Set a fixed random seed for reproducibility
    torch.manual_seed(args.seed)

    # Define split ratios
    total_size = len(combined_dataset)
    train_size = int(0.95 * total_size)
    test_size = total_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

    # Debug: Check split sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    # Create DataLoaders with GPU-optimized settings
    if args.use_gpu_degradation:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=gpu_collate_fn,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=gpu_collate_fn,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=False
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )

    # Debug: Check a batch from the DataLoader
    for batch in train_dataloader:
        if args.use_gpu_degradation:
            print(f"Batch keys: {batch.keys()}")
            print(f"Clean shape: {batch['clean'].shape}, Noise shape: {batch['noise'].shape}")
        else:
            print(f"Batch shape: {[b.shape for b in batch]}")
        break

    model_output_path = Path(args.output_path, args.exp_name, 'model')
    log_output_path = Path(args.output_path, args.exp_name)

    model_output_path.mkdir(parents=True, exist_ok=True)
    log_output_path.mkdir(parents=True, exist_ok=True)

    data_paths = {'model_output': model_output_path, 'log_output': log_output_path}

    # Select trainer based on GPU degradation flag
    if args.use_gpu_degradation:
        trainer = GPUTrainer(
            args,
            data_paths,
            train_dataloader,
            test_dataloader,
            degradation_config=degradation_config
        )
    else:
        trainer = Trainer(args, data_paths, train_dataloader, test_dataloader)

    trainer.train()

    # Print the result path
    print(f"The result is saved at {args.output_path}/{args.exp_name}")

if __name__ == "__main__":
    main()
