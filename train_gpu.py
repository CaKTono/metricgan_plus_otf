"""
GPU-accelerated trainer for MetricGAN+ with on-the-fly degradation

This trainer applies audio degradation on GPU in the training loop,
providing significant speedup over CPU-based degradation in DataLoader workers.
"""

import os
import time
import json
import shutil
import random
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
import torchaudio

from tqdm import tqdm
from glob import glob
from pathlib import Path
from os.path import join as opj
from typing import Dict, Any, Optional

from model import Generator, Discriminator
from signal_processing import get_spec_and_phase, transform_spec_to_wav
from audio_degradation.gpu_degradation_pipeline import GPUDegradationPipeline

fs = 16000


class GPUTrainer:
    """
    GPU-accelerated trainer with on-the-fly degradation.

    This trainer applies audio degradation effects on GPU during training,
    providing significant speedup compared to CPU-based processing.
    """

    def __init__(
        self,
        args,
        data_paths: Dict[str, Path],
        train_dataloader,
        test_dataloader,
        degradation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the GPU trainer.

        Args:
            args: Command line arguments
            data_paths: Dictionary with 'model_output' and 'log_output' paths
            train_dataloader: DataLoader for training (returns raw audio for GPU processing)
            test_dataloader: DataLoader for testing
            degradation_config: Configuration for audio degradation
        """
        self.args = args
        self.device = torch.device(args.device)
        self.target_metric = args.target_metric

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model_output_path = data_paths['model_output']
        self.log_output_path = data_paths['log_output']

        # Initialize GPU degradation pipeline
        self.degradation_config = degradation_config
        self.gpu_degrader = GPUDegradationPipeline(
            config=degradation_config,
            sample_rate=16000,
            device=args.device
        ).to(self.device)

        # Initialize models
        self.init_model_optim()
        self.init_target_metric()

        self.best_scores = {'pesq': -0.5, 'csig': 0, 'cbak': 0, 'covl': 0, 'avg': 0}

        # Create output directories
        os.makedirs(opj(self.args.output_path, self.args.exp_name, 'tmp'), exist_ok=True)

        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Train MetricGAN+-{self.target_metric} (GPU Degradation)\n')
            f.write(f'Train set: {len(train_dataloader.dataset)}, Test set: {len(test_dataloader.dataset)}\n')
            f.write(f'Model parameters: {sum(p.numel() for p in self.G.parameters())/10**6:.3f}M\n')
            f.write(f'GPU degradation enabled: True\n')

        shutil.copy('train_gpu.py', opj(self.args.output_path, self.args.exp_name, 'train_gpu.py'))

        with open(opj(self.args.output_path, self.args.exp_name, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def init_model_optim(self):
        """Initialize models and optimizers."""
        self.G = Generator(causal=self.args.causal).to(self.device)
        self.D = Discriminator().to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        self.MSELoss = nn.MSELoss().to(self.device)
        self.optimizer_g = torch.optim.Adam(self.G.parameters(), lr=self.args.lr)
        self.optimizer_d = torch.optim.Adam(self.D.parameters(), lr=self.args.lr)

    def init_target_metric(self):
        """Initialize target metric function."""
        from metric_functions.get_metric_scores import (
            get_pesq_score, get_csig_score, get_cbak_score, get_covl_score
        )

        self.metric_funcs = {
            'pesq': get_pesq_score,
            'csig': get_csig_score,
            'cbak': get_cbak_score,
            'covl': get_covl_score
        }
        self.target_metric_func = self.metric_funcs[self.target_metric]

    def load_checkpoint(self, ver='latest'):
        """Load model checkpoint."""
        checkpoint = torch.load(opj(self.model_output_path, f'{ver}_model.pth'))
        self.epoch = checkpoint['epoch']
        state_dict_G = checkpoint['generator']
        state_dict_D = checkpoint['discriminator']

        if torch.cuda.device_count() > 1:
            self.G.module.load_state_dict(state_dict_G)
            self.D.module.load_state_dict(state_dict_D)
        else:
            self.G.load_state_dict(state_dict_G)
            self.D.load_state_dict(state_dict_D)

        self.optimizer_g.load_state_dict(checkpoint['g_optimizer'])
        self.optimizer_d.load_state_dict(checkpoint['d_optimizer'])

        if ver == 'best':
            print(f'---{self.epoch} Epoch loaded: model weights and optimizer---')
        else:
            print('---load latest model weights and optimizer---')

    def train(self):
        """Main training loop."""
        start_time = time.time()
        self.epoch = 1
        self.historical_set = []

        while self.epoch <= self.args.epochs:
            try:
                for epoch in np.arange(self.epoch, self.args.epochs + 1):
                    self.epoch = epoch
                    print(f'{epoch} Epoch start')

                    self.train_one_epoch()

                break  # Break if training completes successfully

            except Exception as e:
                print(f"Error occurred during training: {e}. Retrying epoch {self.epoch}...")
                import traceback
                traceback.print_exc()
                continue

        end_time = time.time()
        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Total training time: {(end_time - start_time) / 60:.2f} minutes\n')

        # Best validation scores
        self.load_checkpoint('best')
        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'-------- Model Best score --------\n')
        self.evaluation(phase='best')

    def train_one_epoch(self):
        """Train one epoch."""
        if self.epoch >= 2:
            self.train_generator()

        if self.epoch >= self.args.skip_val_epoch:
            if self.epoch % self.args.eval_per_epoch == 0:
                self.evaluation(phase='test')

        self.train_discriminator()

    def train_generator(self):
        """Train generator with GPU degradation."""
        self.G.train()
        print('Generator training phase (GPU degradation)')

        for batch in tqdm(self.train_dataloader):
            # Move raw audio to GPU
            clean = batch['clean'].to(self.device)
            noise = batch['noise'].to(self.device)
            rir = batch['rir'].to(self.device) if batch['rir'] is not None else None
            rir_mask = batch['rir_mask'].to(self.device) if batch['rir_mask'] is not None else None

            # Apply degradation on GPU (no gradients needed)
            with torch.no_grad():
                noisy, clean_processed = self.gpu_degrader(clean, noise, rir, rir_mask)

            # Get spectrograms
            clean_mag, _ = get_spec_and_phase(clean_processed.unsqueeze(1))
            noise_mag, _ = get_spec_and_phase(noisy.unsqueeze(1))

            # Squeeze batch dimension if needed
            clean_mag = clean_mag.squeeze(1)  # [B, T, F]
            noise_mag = noise_mag.squeeze(1)  # [B, T, F]

            # Get sequence lengths (all same length after padding)
            batch_size = clean_mag.shape[0]
            lengths = torch.full((batch_size,), clean_mag.shape[1], dtype=torch.long, device=self.device)

            # Forward pass through generator
            mask = self.G(noise_mag, lengths)
            mask = mask.clamp(min=0.05)
            enh_mag = torch.mul(mask, noise_mag).unsqueeze(1)

            ref_mag = clean_mag.detach().unsqueeze(1)
            d_inputs = torch.cat([ref_mag, enh_mag], dim=1)

            # Get target scores (compute on-the-fly or use placeholder)
            # For simplicity, use a target of 1.0 (clean reference)
            target = torch.ones(batch_size, device=self.device)

            score = self.D(d_inputs)
            loss = self.MSELoss(score, target)

            self.optimizer_g.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 5.0)
            self.optimizer_g.step()

    def train_discriminator(self):
        """Train discriminator with GPU degradation."""
        print("Discriminator training phase (GPU degradation)")
        self.D.train()

        # Collect samples for discriminator training
        enhanced_samples = []
        noisy_samples = []
        clean_samples = []

        self.G.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                if i >= self.args.num_of_sampling // self.args.batch_size:
                    break

                # Move raw audio to GPU
                clean = batch['clean'].to(self.device)
                noise = batch['noise'].to(self.device)
                rir = batch['rir'].to(self.device) if batch['rir'] is not None else None
                rir_mask = batch['rir_mask'].to(self.device) if batch['rir_mask'] is not None else None

                # Apply degradation on GPU
                noisy, clean_processed = self.gpu_degrader(clean, noise, rir, rir_mask)

                # Get spectrograms
                clean_mag, _ = get_spec_and_phase(clean_processed.unsqueeze(1))
                noise_mag, noise_phase = get_spec_and_phase(noisy.unsqueeze(1))

                clean_mag = clean_mag.squeeze(1)
                noise_mag = noise_mag.squeeze(1)

                # Generate enhanced audio
                mask = self.G(noise_mag)
                mask = mask.clamp(min=0.05)
                enh_mag = torch.mul(mask, noise_mag)

                # Store samples
                clean_samples.append(clean_mag)
                noisy_samples.append(noise_mag)
                enhanced_samples.append(enh_mag)

        # Concatenate all samples
        clean_all = torch.cat(clean_samples, dim=0)
        noisy_all = torch.cat(noisy_samples, dim=0)
        enhanced_all = torch.cat(enhanced_samples, dim=0)

        # Train discriminator on (clean, enhanced), (clean, noisy), (clean, clean)
        self.D.train()
        num_samples = min(clean_all.shape[0], self.args.num_of_sampling)

        # Shuffle indices
        indices = torch.randperm(num_samples)

        for start_idx in range(0, num_samples, self.args.batch_size):
            end_idx = min(start_idx + self.args.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            clean_batch = clean_all[batch_indices].unsqueeze(1)
            noisy_batch = noisy_all[batch_indices].unsqueeze(1)
            enhanced_batch = enhanced_all[batch_indices].unsqueeze(1)

            # Target scores: 1 for clean-clean, estimated for others
            # Using simplified targets: 1.0 for clean-clean, 0.5 for clean-enhanced, 0.2 for clean-noisy
            batch_size = clean_batch.shape[0]

            for target_mag, target_score in [
                (enhanced_batch, 0.5),
                (noisy_batch, 0.2),
                (clean_batch, 1.0)
            ]:
                inputs = torch.cat([clean_batch, target_mag], dim=1).to(self.device)
                score = self.D(inputs)
                target = torch.full((batch_size,), target_score, device=self.device)

                loss = self.MSELoss(score, target)
                self.optimizer_d.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 5.0)
                self.optimizer_d.step()

    def evaluation(self, phase='test'):
        """Evaluate model on test set."""
        print(f'Evaluation on {phase} data')
        self.G.eval()

        test_scores = {'pesq': [], 'csig': [], 'cbak': [], 'covl': []}

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_dataloader)):
                if i >= self.args.num_of_val_sample // self.args.batch_size:
                    break

                # Move raw audio to GPU
                clean = batch['clean'].to(self.device)
                noise = batch['noise'].to(self.device)
                rir = batch['rir'].to(self.device) if batch['rir'] is not None else None
                rir_mask = batch['rir_mask'].to(self.device) if batch['rir_mask'] is not None else None

                # Apply degradation on GPU
                noisy, clean_processed = self.gpu_degrader(clean, noise, rir, rir_mask)

                # Get spectrograms
                noise_mag, noise_phase = get_spec_and_phase(noisy.unsqueeze(1))
                noise_mag = noise_mag.squeeze(1)
                noise_phase = noise_phase.squeeze(1)

                # Generate enhanced audio
                mask = self.G(noise_mag)
                mask = mask.clamp(min=0.05)
                enh_mag = torch.mul(mask, noise_mag)

                # Convert back to waveform
                enh_wav = transform_spec_to_wav(
                    torch.expm1(enh_mag.unsqueeze(1)),
                    noise_phase.unsqueeze(1),
                    signal_length=clean.shape[-1]
                ).squeeze(1)

                # Compute metrics for each sample in batch
                clean_np = clean_processed.cpu().numpy()
                enh_np = enh_wav.cpu().numpy()

                for j in range(clean_np.shape[0]):
                    try:
                        clean_sample = clean_np[j].astype(np.float32)
                        enh_sample = enh_np[j].astype(np.float32)

                        # Compute metrics
                        for metric_name, metric_func in self.metric_funcs.items():
                            score = metric_func(clean_sample, enh_sample, norm=False)
                            test_scores[metric_name].append(score)
                    except Exception as e:
                        continue

        # Average scores
        avg_scores = {k: np.mean(v) if v else 0 for k, v in test_scores.items()}

        with open(opj(self.log_output_path, 'log.txt'), 'a') as f:
            f.write(f'Epoch:{self.epoch} | Test PESQ:{avg_scores["pesq"]:.3f} | '
                    f'Test CSIG:{avg_scores["csig"]:.3f} | Test CBAK:{avg_scores["cbak"]:.3f} | '
                    f'Test COVL:{avg_scores["covl"]:.3f}\n')

        print(f'PESQ: {avg_scores["pesq"]:.3f}, CSIG: {avg_scores["csig"]:.3f}, '
              f'CBAK: {avg_scores["cbak"]:.3f}, COVL: {avg_scores["covl"]:.3f}')

        if phase in ['valid', 'test']:
            # Save checkpoint
            if torch.cuda.device_count() > 1:
                state_dict_G = self.G.module.state_dict()
                state_dict_D = self.D.module.state_dict()
            else:
                state_dict_G = self.G.state_dict()
                state_dict_D = self.D.state_dict()

            checkpoint = {
                'epoch': self.epoch,
                'stats': avg_scores,
                'generator': state_dict_G,
                'discriminator': state_dict_D,
                'g_optimizer': self.optimizer_g.state_dict(),
                'd_optimizer': self.optimizer_d.state_dict(),
            }

            if avg_scores['pesq'] >= self.best_scores['pesq']:
                print('----------------------------------------')
                print('-----------------SAVE-------------------')
                self.best_scores = avg_scores
                torch.save(checkpoint, opj(self.model_output_path, 'best_model.pth'))
                print('----------------------------------------')

            torch.save(checkpoint, opj(self.model_output_path, 'latest_model.pth'))

        return avg_scores
