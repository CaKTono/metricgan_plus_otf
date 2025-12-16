#!/usr/bin/env python3
"""
Utility script to create .scp files from audio directories.

Usage:
    python create_scp.py --audio_dir /path/to/audio --output_scp output.scp --recursive
"""

import argparse
import os
from pathlib import Path
from typing import List


def find_audio_files(
    directory: str,
    extensions: List[str] = ['.wav', '.flac', '.mp3', '.ogg'],
    recursive: bool = True
) -> List[str]:
    """
    Find all audio files in a directory.

    Args:
        directory: Root directory to search
        extensions: List of audio file extensions to include
        recursive: Whether to search subdirectories

    Returns:
        List of absolute paths to audio files
    """
    audio_files = []
    directory = Path(directory)

    if recursive:
        for ext in extensions:
            audio_files.extend(directory.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            audio_files.extend(directory.glob(f'*{ext}'))

    return sorted([str(f.absolute()) for f in audio_files])


def create_scp_file(
    audio_files: List[str],
    output_path: str,
    format_type: str = 'full'
):
    """
    Create an .scp file from a list of audio files.

    Args:
        audio_files: List of audio file paths
        output_path: Output .scp file path
        format_type: Format type ('full', 'simple', or 'kaldi')
            - 'full': <utt_id> <speaker_id> <path>
            - 'simple': <path>
            - 'kaldi': <utt_id> <path>
    """
    with open(output_path, 'w') as f:
        for i, audio_path in enumerate(audio_files):
            path = Path(audio_path)

            # Extract utterance ID from filename
            utt_id = path.stem

            # Try to extract speaker ID from parent directory
            speaker_id = path.parent.name

            if format_type == 'full':
                f.write(f'{utt_id} {speaker_id} {audio_path}\n')
            elif format_type == 'simple':
                f.write(f'{audio_path}\n')
            elif format_type == 'kaldi':
                f.write(f'{utt_id} {audio_path}\n')
            else:
                raise ValueError(f'Unknown format type: {format_type}')

    print(f'Created {output_path} with {len(audio_files)} entries')


def main():
    parser = argparse.ArgumentParser(
        description='Create .scp files from audio directories'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        required=True,
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--output_scp',
        type=str,
        required=True,
        help='Output .scp file path'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search subdirectories recursively'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='full',
        choices=['full', 'simple', 'kaldi'],
        help='SCP file format (default: full)'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.wav', '.flac', '.mp3'],
        help='Audio file extensions to include'
    )

    args = parser.parse_args()

    # Find audio files
    audio_files = find_audio_files(
        args.audio_dir,
        extensions=args.extensions,
        recursive=args.recursive
    )

    if not audio_files:
        print(f'No audio files found in {args.audio_dir}')
        return

    print(f'Found {len(audio_files)} audio files')

    # Create SCP file
    create_scp_file(audio_files, args.output_scp, args.format)


if __name__ == '__main__':
    main()
