# get_metric_scores.py
import os 
import librosa
import numpy as np
from joblib import Parallel, delayed, wrap_non_picklable_objects
from .compute_metric import compute_pesq, compute_csig, compute_cbak, compute_covl

fs = 16000  # Sampling frequency

# Helper function to load waveforms
def load_wavs(clean_input, enhanced_input):
    """
    Load clean and enhanced waveforms from file paths or use in-memory arrays.

    Args:
        clean_input (str or np.ndarray): Path to clean waveform or in-memory array.
        enhanced_input (str or np.ndarray): Path to enhanced waveform or in-memory array.

    Returns:
        tuple: (clean_wav, enhanced_wav)
    """
    if isinstance(clean_input, (str, os.PathLike)) and isinstance(enhanced_input, (str, os.PathLike)):
        # Load from file paths
        name = os.path.basename(enhanced_input)
        print("clean_input: ", clean_input)
        print("enhanced_input: ", enhanced_input)
        wave_name = name.split('#')[0] + '.wav' if '#' in name else name
        
        clean_wav, sr = librosa.load(os.path.join(clean_input, wave_name), sr=fs)
        enhanced_wav, _ = librosa.load(enhanced_input, sr=fs)
    elif isinstance(clean_input, np.ndarray) and isinstance(enhanced_input, np.ndarray):
        # Use in-memory arrays
        clean_wav = clean_input
        enhanced_wav = enhanced_input
        sr = fs  # Assuming consistent sampling rate
    else:
        raise ValueError("Input must be either file paths or numpy arrays.")
    
    # Ensure both waveforms have the same length
    min_length = min(len(clean_wav), len(enhanced_wav))
    clean_wav = clean_wav[:min_length]
    enhanced_wav = enhanced_wav[:min_length]
    
    return clean_wav, enhanced_wav

# PESQ
@wrap_non_picklable_objects
def get_pesq_score(clean_wav, enhanced_wav, norm):
    """
    Compute PESQ score between clean and enhanced waveforms.

    Args:
        clean_wav (str or np.ndarray): Clean waveform path or array.
        enhanced_wav (str or np.ndarray): Enhanced waveform path or array.
        norm (bool): Whether to normalize the PESQ score.

    Returns:
        float: PESQ score.
    """
    clean_wav, enhanced_wav = load_wavs(clean_wav, enhanced_wav)

    # Ensure 1D arrays
    if clean_wav.ndim > 1:
        clean_wav = clean_wav.flatten()
        print(f"Squeezed clean_wav to shape: {clean_wav.shape}")
    if enhanced_wav.ndim > 1:
        enhanced_wav = enhanced_wav.flatten()
        print(f"Squeezed enhanced_wav to shape: {enhanced_wav.shape}")

    # Ensure both waveforms have the same length
    min_length = min(len(clean_wav), len(enhanced_wav))
    clean_wav = clean_wav[:min_length]
    enhanced_wav = enhanced_wav[:min_length]

    # Debugging: Print shapes
    print(f"PESQ - clean_wav shape: {clean_wav.shape}, enhanced_wav shape: {enhanced_wav.shape}")

    # Compute PESQ
    score = compute_pesq(clean_wav, enhanced_wav, fs, norm)
    return score

def get_pesq_parallel(clean_list, enhanced_list, norm=True, n_jobs=16):
    """
    Compute PESQ scores in parallel for lists of waveforms.

    Args:
        clean_list (list): List of clean waveforms (paths or arrays).
        enhanced_list (list): List of enhanced waveforms (paths or arrays).
        norm (bool): Whether to normalize the PESQ scores.
        n_jobs (int): Number of parallel jobs.

    Returns:
        list: List of PESQ scores.
    """
    score = Parallel(n_jobs=n_jobs)(
        delayed(get_pesq_score)(clean_wav, enhanced_wav, norm) 
        for clean_wav, enhanced_wav in zip(clean_list, enhanced_list)
    )
    return score


# CSIG
@wrap_non_picklable_objects
def get_csig_score(clean_wav, enhanced_wav, norm):
    """
    Compute CSIG score between clean and enhanced waveforms.

    Args:
        clean_wav (str or np.ndarray): Clean waveform path or array.
        enhanced_wav (str or np.ndarray): Enhanced waveform path or array.
        norm (bool): Whether to normalize the CSIG score.

    Returns:
        float: CSIG score.
    """
    clean_wav, enhanced_wav = load_wavs(clean_wav, enhanced_wav)

    # Ensure 1D arrays
    if clean_wav.ndim > 1:
        clean_wav = clean_wav.flatten()
        print(f"Squeezed clean_wav to shape: {clean_wav.shape}")
    if enhanced_wav.ndim > 1:
        enhanced_wav = enhanced_wav.flatten()
        print(f"Squeezed enhanced_wav to shape: {enhanced_wav.shape}")

    # Ensure both waveforms have the same length
    min_length = min(len(clean_wav), len(enhanced_wav))
    clean_wav = clean_wav[:min_length]
    enhanced_wav = enhanced_wav[:min_length]

    # Debugging: Print shapes
    print(f"CSIG - clean_wav shape: {clean_wav.shape}, enhanced_wav shape: {enhanced_wav.shape}")

    # Compute CSIG
    score = compute_csig(clean_wav, enhanced_wav, fs, norm)
    return score

def get_csig_parallel(clean_list, enhanced_list, norm=True, n_jobs=16):
    """
    Compute CSIG scores in parallel for lists of waveforms.

    Args:
        clean_list (list): List of clean waveforms (paths or arrays).
        enhanced_list (list): List of enhanced waveforms (paths or arrays).
        norm (bool): Whether to normalize the CSIG scores.
        n_jobs (int): Number of parallel jobs.

    Returns:
        list: List of CSIG scores.
    """
    score = Parallel(n_jobs=n_jobs)(
        delayed(get_csig_score)(clean_wav, enhanced_wav, norm) 
        for clean_wav, enhanced_wav in zip(clean_list, enhanced_list)
    )
    return score


# CBAK
@wrap_non_picklable_objects
def get_cbak_score(clean_wav, enhanced_wav, norm):
    """
    Compute CBAK score between clean and enhanced waveforms.

    Args:
        clean_wav (str or np.ndarray): Clean waveform path or array.
        enhanced_wav (str or np.ndarray): Enhanced waveform path or array.
        norm (bool): Whether to normalize the CBAK score.

    Returns:
        float: CBAK score.
    """
    clean_wav, enhanced_wav = load_wavs(clean_wav, enhanced_wav)

    # Ensure 1D arrays
    if clean_wav.ndim > 1:
        clean_wav = clean_wav.flatten()
        print(f"Squeezed clean_wav to shape: {clean_wav.shape}")
    if enhanced_wav.ndim > 1:
        enhanced_wav = enhanced_wav.flatten()
        print(f"Squeezed enhanced_wav to shape: {enhanced_wav.shape}")

    # Ensure both waveforms have the same length
    min_length = min(len(clean_wav), len(enhanced_wav))
    clean_wav = clean_wav[:min_length]
    enhanced_wav = enhanced_wav[:min_length]

    # Debugging: Print shapes
    print(f"CBAK - clean_wav shape: {clean_wav.shape}, enhanced_wav shape: {enhanced_wav.shape}")

    # Compute CBAK
    score = compute_cbak(clean_wav, enhanced_wav, fs, norm)
    return score

def get_cbak_parallel(clean_list, enhanced_list, norm=True, n_jobs=16):
    """
    Compute CBAK scores in parallel for lists of waveforms.

    Args:
        clean_list (list): List of clean waveforms (paths or arrays).
        enhanced_list (list): List of enhanced waveforms (paths or arrays).
        norm (bool): Whether to normalize the CBAK scores.
        n_jobs (int): Number of parallel jobs.

    Returns:
        list: List of CBAK scores.
    """
    score = Parallel(n_jobs=n_jobs)(
        delayed(get_cbak_score)(clean_wav, enhanced_wav, norm) 
        for clean_wav, enhanced_wav in zip(clean_list, enhanced_list)
    )
    return score


# COVL
@wrap_non_picklable_objects
def get_covl_score(clean_wav, enhanced_wav, norm):
    """
    Compute COVL score between clean and enhanced waveforms.

    Args:
        clean_wav (str or np.ndarray): Clean waveform path or array.
        enhanced_wav (str or np.ndarray): Enhanced waveform path or array.
        norm (bool): Whether to normalize the COVL score.

    Returns:
        float: COVL score.
    """
    clean_wav, enhanced_wav = load_wavs(clean_wav, enhanced_wav)

    # Ensure 1D arrays
    if clean_wav.ndim > 1:
        clean_wav = clean_wav.flatten()
        print(f"Squeezed clean_wav to shape: {clean_wav.shape}")
    if enhanced_wav.ndim > 1:
        enhanced_wav = enhanced_wav.flatten()
        print(f"Squeezed enhanced_wav to shape: {enhanced_wav.shape}")

    # Ensure both waveforms have the same length
    min_length = min(len(clean_wav), len(enhanced_wav))
    clean_wav = clean_wav[:min_length]
    enhanced_wav = enhanced_wav[:min_length]

    # Debugging: Print shapes
    print(f"COVL - clean_wav shape: {clean_wav.shape}, enhanced_wav shape: {enhanced_wav.shape}")

    # Compute COVL
    score = compute_covl(clean_wav, enhanced_wav, fs, norm)
    return score

def get_covl_parallel(clean_list, enhanced_list, norm=True, n_jobs=16):
    """
    Compute COVL scores in parallel for lists of waveforms.

    Args:
        clean_list (list): List of clean waveforms (paths or arrays).
        enhanced_list (list): List of enhanced waveforms (paths or arrays).
        norm (bool): Whether to normalize the COVL scores.
        n_jobs (int): Number of parallel jobs.

    Returns:
        list: List of COVL scores.
    """
    score = Parallel(n_jobs=n_jobs)(
        delayed(get_covl_score)(clean_wav, enhanced_wav, norm) 
        for clean_wav, enhanced_wav in zip(clean_list, enhanced_list)
    )
    return score
