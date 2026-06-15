"""
Audio Preprocessing Module for Deepfake Detection

Extracts audio from video files and computes MFCC (Mel-Frequency Cepstral Coefficient)
features for use in audio classification.
"""

import os
import numpy as np
import librosa
import tempfile
from pathlib import Path


class AudioPreprocessor:
    """
    Handles audio extraction from video and MFCC feature computation.
    """
    
    def __init__(self, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512):
        """
        Initialize the AudioPreprocessor.
        
        Args:
            sr (int): Sample rate for audio processing (default: 22050 Hz)
            n_mfcc (int): Number of MFCC coefficients to extract (default: 40)
            n_fft (int): FFT window size (default: 2048)
            hop_length (int): Number of samples between successive frames (default: 512)
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_audio_from_video(self, video_path, output_wav_path=None):
        """
        Extract audio from video file and save as WAV.
        
        Args:
            video_path (str): Path to video file
            output_wav_path (str, optional): Path to save extracted audio.
                                             If None, uses temporary file.
        
        Returns:
            str: Path to extracted WAV file, or None if no audio track found
        
        Raises:
            FileNotFoundError: If video file doesn't exist
            Exception: If moviepy fails to extract audio
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            raise ImportError("moviepy not installed. Install with: pip install moviepy")
        
        try:
            # Load video
            video = VideoFileClip(video_path)
            
            # Check if audio exists
            if video.audio is None:
                video.close()
                return None
            
            # Set output path
            if output_wav_path is None:
                temp_dir = tempfile.gettempdir()
                output_wav_path = os.path.join(temp_dir, "extracted_audio.wav")
            
            # Write audio to file
            video.audio.write_audiofile(
                output_wav_path,
                verbose=False,
                logger=None  # Suppress output
            )
            
            video.close()
            
            return output_wav_path
        
        except Exception as e:
            raise Exception(f"Error extracting audio from video: {str(e)}")
    
    def extract_mfcc_features(self, audio_path):
        """
        Extract MFCC features from audio file.
        
        Args:
            audio_path (str): Path to audio file (WAV, MP3, etc.)
        
        Returns:
            np.ndarray: Mean MFCC features of shape (n_mfcc,).
                        Returns zeros if audio file is invalid.
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Handle edge case: empty audio
            if len(y) == 0:
                return np.zeros(self.n_mfcc)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Compute mean across time axis
            mfcc_mean = np.mean(mfcc, axis=1)
            
            return mfcc_mean
        
        except Exception as e:
            print(f"Warning: Error extracting MFCC features: {str(e)}")
            return np.zeros(self.n_mfcc)
    
    def process_video(self, video_path, cleanup_audio=True):
        """
        Complete pipeline: extract audio from video and compute MFCC features.
        
        Args:
            video_path (str): Path to video file
            cleanup_audio (bool): Whether to delete temporary audio file after processing
        
        Returns:
            dict: Dictionary containing:
                - 'mfcc_features': np.ndarray of MFCC features (n_mfcc,)
                - 'has_audio': bool indicating if audio was found
                - 'audio_path': str path to extracted audio (or None)
        
        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        # Extract audio
        audio_path = self.extract_audio_from_video(video_path)
        
        if audio_path is None:
            # No audio found - return zero features
            return {
                'mfcc_features': np.zeros(self.n_mfcc),
                'has_audio': False,
                'audio_path': None
            }
        
        # Extract MFCC features
        mfcc_features = self.extract_mfcc_features(audio_path)
        
        # Cleanup temporary audio file if requested
        if cleanup_audio and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass  # Silently ignore cleanup errors
        
        return {
            'mfcc_features': mfcc_features,
            'has_audio': True,
            'audio_path': audio_path if not cleanup_audio else None
        }
    
    @staticmethod
    def get_mfcc_spectrogram(audio_path, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512):
        """
        Get full MFCC spectrogram (not averaged) for visualization.
        
        Args:
            audio_path (str): Path to audio file
            sr (int): Sample rate
            n_mfcc (int): Number of MFCC coefficients
            n_fft (int): FFT window size
            hop_length (int): Hop length
        
        Returns:
            np.ndarray: MFCC spectrogram of shape (n_mfcc, time_steps)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            
            if len(y) == 0:
                return np.zeros((n_mfcc, 1))
            
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            return mfcc
        except Exception as e:
            print(f"Warning: Error extracting MFCC spectrogram: {str(e)}")
            return np.zeros((n_mfcc, 1))
    
    @staticmethod
    def get_waveform(audio_path, sr=22050):
        """
        Get audio waveform for visualization.
        
        Args:
            audio_path (str): Path to audio file
            sr (int): Sample rate
        
        Returns:
            tuple: (waveform array, sample rate)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            return y, sr
        except Exception as e:
            print(f"Warning: Error loading waveform: {str(e)}")
            return np.array([]), sr
