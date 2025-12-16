# audio_degradation/degradation_dataset.py

import sys
import os
# Add the current directory to the sys.path to import modules correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from tqdm import tqdm
from audio_degradation_pipeline import process_from_audio_path  # Corrected import
from metric_functions.get_metric_scores import (
    get_pesq_parallel,
    get_csig_parallel,
    get_cbak_parallel,
    get_covl_parallel,
)
from joblib import Parallel, delayed

def silence_ratio(wav: np.ndarray, thres: float = -40.) -> float:
    """
    Calculate the silence ratio of an audio waveform.

    Args:
        wav (np.ndarray): Audio waveform.
        thres (float): Threshold in dB below which audio is considered silent.

    Returns:
        float: Ratio of silent frames in the audio.
    """
    threshold = 10 ** (thres / 20.)
    return np.sum(np.abs(wav) < threshold) / len(wav)

class AudioDegradationDataset(Dataset):
    def __init__(self, speech_list, noise_list, rir_list, degradation_config, 
                 seq_len=(512 - 1) * 256 + 512, sr=16000, device='cpu', batch_size=1000, n_jobs=16,
                 silence_threshold=-40.0, silence_ratio_threshold=0.5):
        """
        Initializes the AudioDegradationDataset.

        Args:
            speech_list (str): Path to the speech .scp file.
            noise_list (str): Path to the noise .scp file.
            rir_list (str or None): Path to the RIR .scp file or None.
            degradation_config (dict): Configuration for audio degradation.
            seq_len (int): Sequence length for audio samples.
            sr (int): Sampling rate.
            device (str): Device to use ('cpu' or 'cuda').
            batch_size (int): Number of samples per batch.
            n_jobs (int): Number of parallel jobs.
            silence_threshold (float): dB threshold below which audio is considered silent.
            silence_ratio_threshold (float): Maximum allowable ratio of silence in audio.
        """
        # Directly use the paths from the .scp files
        self.speech_list, self.noise_list, self.rir_list = self.get_list(speech_list, noise_list, rir_list)
        self.degradation_config = degradation_config
        # Calculate new sequence length for the given sampling rate
        new_seq_len = int(seq_len * (sr / 44100))
        self.seq_len = new_seq_len
        
        self.sr = sr
        self.device = device
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.silence_threshold = silence_threshold  # dB threshold for silence
        self.silence_ratio_threshold = silence_ratio_threshold  # Ratio threshold to consider as silent

    def get_list(self, speech_list, noise_list, rir_list):
        """
        Parses the .scp files to retrieve audio file paths.

        Args:
            speech_list (str): Path to the speech .scp file.
            noise_list (str): Path to the noise .scp file.
            rir_list (str or None): Path to the RIR .scp file or None.

        Returns:
            tuple: Lists of speech paths, noise paths, and RIR paths.
        """
        # Use a context manager to handle optional logging
        with open(os.devnull, 'w') as log_file:
            def parse_file(file_path):
                list_paths = []
                with open(file_path, 'r') as file:
                    for line in file:
                        # Split the line by space and extract the third element as the path
                        parts = line.strip().split(maxsplit=2)
                        if len(parts) < 3:
                            log_file.write(f"Invalid line format: {line.strip()}\n")
                            continue
                        full_path = parts[2]  # Assuming the path is the third element
                        list_paths.append(full_path)
                        log_file.write(f"Parsed path: {full_path}\n")  # Write each parsed path to the log file
                return list_paths

            speech_paths = parse_file(speech_list)
            noise_paths = parse_file(noise_list)
            rir_paths = parse_file(rir_list) if rir_list else None

        return speech_paths, noise_paths, rir_paths

    def __len__(self):
        return len(self.speech_list)
    
    def check_and_filter_scores(self, clean_samples, noisy_samples, sample_ids, sample_paths):
        """
        Compute silence ratio and metrics in parallel for a batch of clean and noisy samples.
        Identify valid and invalid samples based on silence ratio and metric thresholds.

        Args:
            clean_samples (list of np.ndarray): List of clean audio samples.
            noisy_samples (list of np.ndarray): List of noisy audio samples.
            sample_ids (list of int): List of sample indices.
            sample_paths (list of tuple): List of tuples containing (speech_path, noise_path).

        Returns:
            tuple:
                valid_indices (list of int): Indices of samples that meet silence and metric thresholds.
                invalid_indices (list of int): Indices of samples that fail any threshold.
        """
        total_samples = len(clean_samples)
        print(f"Total samples before filtering: {total_samples}")

        # Initialize lists to store indices
        valid_indices = []
        invalid_indices = []

        # Compute silence ratio for all noisy samples in parallel
        silence_ratios = Parallel(n_jobs=self.n_jobs)(
            delayed(silence_ratio)(sample, self.silence_threshold) for sample in noisy_samples
        )

        # First pass: Filter out silent samples
        non_silent_indices = []
        for idx, ratio in enumerate(silence_ratios):
            if ratio < self.silence_ratio_threshold:
                non_silent_indices.append(idx)
            else:
                invalid_indices.append(idx)
                speech_path, noise_path = sample_paths[idx]
                print(f"Audio is silent, skipping file! Path: {speech_path}")

        print(f"Silence Filtering: {len(non_silent_indices)} valid samples, {len(invalid_indices)} silent samples skipped.")

        if not non_silent_indices:
            print("No non-silent samples to process.")
            return valid_indices, invalid_indices

        # Extract the non-silent samples for metric computation
        filtered_clean_samples = [clean_samples[i] for i in non_silent_indices]
        filtered_noisy_samples = [noisy_samples[i] for i in non_silent_indices]
        filtered_sample_ids = [sample_ids[i] for i in non_silent_indices]
        filtered_sample_paths = [sample_paths[i] for i in non_silent_indices]

        # Compute all metrics in parallel for non-silent samples
        pesq_scores = get_pesq_parallel(filtered_clean_samples, filtered_noisy_samples, norm=False, n_jobs=self.n_jobs)
        csig_scores = get_csig_parallel(filtered_clean_samples, filtered_noisy_samples, norm=False, n_jobs=self.n_jobs)
        cbak_scores = get_cbak_parallel(filtered_clean_samples, filtered_noisy_samples, norm=False, n_jobs=self.n_jobs)
        covl_scores = get_covl_parallel(filtered_clean_samples, filtered_noisy_samples, norm=False, n_jobs=self.n_jobs)

        # Define metric thresholds
        pesq_threshold = -0.5
        csig_threshold = 1.0
        cbak_threshold = 1.0
        covl_threshold = 1.0

        # Second pass: Filter based on metric scores
        for idx, sample_idx in enumerate(filtered_sample_ids):
            if (pesq_scores[idx] >= pesq_threshold and
                csig_scores[idx] >= csig_threshold and
                cbak_scores[idx] >= cbak_threshold and
                covl_scores[idx] >= covl_threshold):
                valid_indices.append(sample_idx)
                # print(f"Sample {sample_idx}: pesq_score={pesq_scores[idx]}, csig_score={csig_scores[idx]}, cbak_score={cbak_scores[idx]}, covl_score={covl_scores[idx]}")
            else:
                invalid_indices.append(sample_idx)
                speech_path, noise_path = filtered_sample_paths[idx]
                # print(f"Sample {sample_idx}: Invalid scores. Path: {speech_path}, pesq_score={pesq_scores[idx]}, csig_score={csig_scores[idx]}, cbak_score={cbak_scores[idx]}, covl_score={covl_scores[idx]}")

        print(f"Metric Filtering: {len(valid_indices)} valid samples, {len(invalid_indices)} invalid samples.")

        return valid_indices, invalid_indices


    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (clean_sample, noisy_sample) as torch tensors.
        """
        speech_path = self.speech_list[idx]
        noise_path = random.choice(self.noise_list)
        rir_path = random.choice(self.rir_list) if self.rir_list else None
        
        # Process the audio file with randomly selected noise and RIR
        speech_sample, noise_sample, noisy_speech, fs = process_from_audio_path(
            vocal_path=speech_path, 
            noise_path=noise_path, 
            rir_path=rir_path, 
            fs=self.sr, 
            force_1ch=True,
            degradation_config=self.degradation_config,
            length=self.seq_len,
        )
        # Ensure audio is mono by squeezing any 2D array (in case it's stereo)
        if isinstance(speech_sample, list):
            speech_sample = np.array(speech_sample)
        if isinstance(noisy_speech, list):
            noisy_speech = np.array(noisy_speech)

        speech_sample = np.squeeze(speech_sample)
        noisy_speech = np.squeeze(noisy_speech)

        # Convert numpy arrays to torch tensors
        speech_sample = torch.from_numpy(speech_sample).float()
        noisy_speech = torch.from_numpy(noisy_speech).float()

        # Pad or truncate to sequence length
        speech_sample, noisy_speech = self.pad_or_truncate(speech_sample, noisy_speech)

        return speech_sample, noisy_speech

    def pad_or_truncate(self, clean, noisy):
        """
        Pads or truncates the audio samples to the specified sequence length.

        Args:
            clean (torch.Tensor): Clean audio sample.
            noisy (torch.Tensor): Noisy audio sample.

        Returns:
            tuple: (padded/truncated clean sample, padded/truncated noisy sample)
        """
        if clean.size(-1) < self.seq_len:
            repeat_times = self.seq_len // clean.size(-1) + 1
            clean = clean.repeat(1, repeat_times)
            noisy = noisy.repeat(1, repeat_times)
            clean = clean[:, :self.seq_len]
            noisy = noisy[:, :self.seq_len]
        elif clean.size(-1) > self.seq_len:
            offset = np.random.randint(0, clean.size(-1) - self.seq_len)
            clean = clean[..., offset:offset+self.seq_len]
            noisy = noisy[..., offset:offset+self.seq_len]
        return clean, noisy

    def generate_dataset(self, output_dir, train_percentage=0.8):
        """
        Generate a dataset with degraded audio and save only valid samples in the specified directory structure.
        Metrics are computed in parallel to improve generation time.

        Args:
            output_dir (str): The root directory where the dataset will be saved.
            train_percentage (float): The percentage of data to be used for training.
        """
        # Print the initial amount of dataset
        initial_count = len(self.speech_list)
        print(f"Initial amount of dataset: {initial_count}")

        # Create directories
        train_clean_dir = os.path.join(output_dir, 'singing_voice/train/clean')
        train_noisy_dir = os.path.join(output_dir, 'singing_voice/train/noisy')
        test_clean_dir = os.path.join(output_dir, 'singing_voice/test/clean')
        test_noisy_dir = os.path.join(output_dir, 'singing_voice/test/noisy')

        for dir_path in [train_clean_dir, train_noisy_dir, test_clean_dir, test_noisy_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Shuffle and split the dataset
        indices = list(range(len(self.speech_list)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_percentage)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Define a helper function to process a batch of samples
        def process_batch(batch_indices, phase):
            clean_samples = []
            noisy_samples = []
            valid_indices_in_batch = []
            invalid_indices_in_batch = []
            sample_ids = []
            sample_paths = []

            # Process each sample in the batch
            for idx in batch_indices:
                speech_path = self.speech_list[idx]
                noise_path = random.choice(self.noise_list)
                rir_path = random.choice(self.rir_list) if self.rir_list else None

                try:
                    speech_sample, noise_sample, noisy_speech, fs = process_from_audio_path(
                        vocal_path=speech_path,
                        noise_path=noise_path,
                        rir_path=rir_path,
                        fs=self.sr,
                        force_1ch=True,
                        degradation_config=self.degradation_config,
                        length=self.seq_len,
                        check_scores_before_return=False  # We'll handle metrics separately
                    )
                    # Ensure audio is mono by converting lists to numpy arrays if necessary
                    if isinstance(speech_sample, list):
                        speech_sample = np.array(speech_sample)
                    if isinstance(noisy_speech, list):
                        noisy_speech = np.array(noisy_speech)

                    speech_sample = np.squeeze(speech_sample)
                    noisy_speech = np.squeeze(noisy_speech)

                    clean_samples.append(speech_sample)
                    noisy_samples.append(noisy_speech)
                    sample_ids.append(idx)
                    sample_paths.append((speech_path, noise_path))
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue  # Skip this sample

            if not clean_samples:
                return  # No samples to process in this batch

            # Compute metrics and filter samples
            valid_indices, invalid_indices = self.check_and_filter_scores(clean_samples, noisy_samples, sample_ids, sample_paths)

            # Save valid samples
            for i in valid_indices:
                idx = i
                # Find the position of idx in sample_ids
                try:
                    position = sample_ids.index(idx)
                except ValueError:
                    print(f"Sample ID {idx} not found in sample_ids list.")
                    continue
                speech_path, noise_path = sample_paths[position]
                clean_filename = f"{phase}_sample_{idx}.wav"
                noisy_filename = f"{phase}_sample_{idx}.wav"
                # print(f"Noisy filename: {noisy_filename}, Clean filename: {clean_filename}")

                clean_path = os.path.join(output_dir, f'singing_voice/{phase}/clean', clean_filename)
                noisy_path = os.path.join(output_dir, f'singing_voice/{phase}/noisy', noisy_filename)

                try:
                    sf.write(clean_path, clean_samples[position], samplerate=16000)
                    sf.write(noisy_path, noisy_samples[position], samplerate=16000)
                except Exception as e:
                    print(f"Error saving sample {idx}: {e}")
                    continue  # Skip saving this sample

            # Optionally, log or handle invalid samples
            for i in invalid_indices:
                idx = i
                # Find the position of idx in sample_ids
                try:
                    position = sample_ids.index(idx)
                    speech_path, noise_path = sample_paths[position]
                except ValueError:
                    speech_path = "Unknown"
                print(f"Eliminated sample {idx} with path {speech_path} due to invalid metrics or silence.")

        # Process training and testing sets
        for phase, batch_indices in zip(['train', 'test'], [train_indices, test_indices]):
            print(f"Processing {phase} set with {len(batch_indices)} samples...")
            # Split into batches
            for batch_start in tqdm(range(0, len(batch_indices), self.batch_size)):
                batch_indices_subset = batch_indices[batch_start:batch_start + self.batch_size]
                process_batch(batch_indices_subset, phase)  # Sequentially process batches

        # After processing, count the number of generated datasets in the output path
        print("\nDataset generation completed. Counting generated samples...")
        for phase in ['train', 'test']:
            clean_dir = os.path.join(output_dir, f'singing_voice/{phase}/clean')
            noisy_dir = os.path.join(output_dir, f'singing_voice/{phase}/noisy')
            
            num_clean = len([f for f in os.listdir(clean_dir) if os.path.isfile(os.path.join(clean_dir, f))])
            num_noisy = len([f for f in os.listdir(noisy_dir) if os.path.isfile(os.path.join(noisy_dir, f))])
            
            print(f"Phase: {phase} - Clean samples: {num_clean}, Noisy samples: {num_noisy}")

        # Optionally, print total generated samples
        total_clean = 0
        total_noisy = 0
        for phase in ['train', 'test']:
            clean_dir = os.path.join(output_dir, f'singing_voice/{phase}/clean')
            noisy_dir = os.path.join(output_dir, f'singing_voice/{phase}/noisy')
            
            num_clean = len([f for f in os.listdir(clean_dir) if os.path.isfile(os.path.join(clean_dir, f))])
            num_noisy = len([f for f in os.listdir(noisy_dir) if os.path.isfile(os.path.join(noisy_dir, f))])
            
            total_clean += num_clean
            total_noisy += num_noisy

        print(f"\nTotal generated dataset after filtering: Clean samples = {total_clean}, Noisy samples = {total_noisy}")

def main():
    speech_list = "/mnt/workspace/calvin/data/singing_train.scp"
    noise_list = "/mnt/workspace/calvin/data/noise_train.scp"
    rir_list = "/mnt/workspace/calvin/data/rir_train.scp"
    
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

        # Package loss
        "p_pl": 0.0,
        "pl_min_ratio": 0.05,
        "pl_max_ratio": 0.1,
        "pl_min_length": 0.05,
        "pl_max_length": 0.1
    }
    
    dataset = AudioDegradationDataset(
        speech_list=speech_list, 
        noise_list=noise_list, 
        rir_list=rir_list, 
        degradation_config=degradation_config,
        batch_size=100,  # Adjust based on your memory and performance
        n_jobs=32,         # Number of parallel jobs; set to the number of CPU cores
        silence_threshold=-40.0,          # dB threshold for silence
        silence_ratio_threshold=0.5        # Ratio threshold to consider as silent
    )
    dataset.generate_dataset(output_dir='/mnt/workspace/calvin/generated_dataset', train_percentage=0.95)

if __name__ == "__main__":
    main()
