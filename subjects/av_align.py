"""Audio-Video Alignment evaluation using AV-Align metric.

This module evaluates the alignment between audio and video modalities using 
the AV-Align metric from TempoTokens. The metric assesses synchronization by 
detecting audio and video peaks and calculating their Intersection over Union (IoU).
A higher IoU score indicates better alignment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import librosa
import numpy as np
from tqdm import tqdm

from video import VideoData


SUBJECT_NAME = "av_align"
DEFAULT_DOWNSAMPLE_FACTOR = 1  # Process every Nth frame to speed up computation


def extract_frames_optimized(video_path: str, downsample: int = 1) -> tuple[np.ndarray, float, int]:
    """Extract frames from a video file (optimized version).

    Args:
        video_path: Path to the input video file.
        downsample: Factor to downsample frames (e.g., 2 = every other frame).

    Returns:
        A tuple of (frames, frame_rate, num_frames) where frames is a numpy array
        of shape (N, H, W) containing grayscale frames, frame_rate is the effective FPS
        after downsampling, and num_frames is the total number of frames extracted.
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open the video file: {video_path}")

    # Calculate downsampled frame count
    expected_frames = (total_frames + downsample - 1) // downsample
    
    # Pre-allocate array for grayscale frames
    first_ret, first_frame = cap.read()
    if not first_ret:
        cap.release()
        raise ValueError(f"Error: Cannot read first frame from {video_path}")
    
    h, w = first_frame.shape[:2]
    frames_gray = np.zeros((expected_frames, h, w), dtype=np.uint8)
    
    # Convert first frame to grayscale
    frames_gray[0] = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Read and convert frames with downsampling
    frame_idx = 1
    actual_frame_count = 1
    
    while actual_frame_count < expected_frames:
        # Skip frames according to downsample factor
        for _ in range(downsample):
            ret, frame = cap.read()
            frame_idx += 1
            if not ret:
                break
        
        if not ret:
            break
            
        frames_gray[actual_frame_count] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        actual_frame_count += 1
    
    cap.release()
    
    # Trim array if we read fewer frames than expected
    if actual_frame_count < expected_frames:
        frames_gray = frames_gray[:actual_frame_count]
    
    # Adjust FPS based on downsampling
    effective_fps = original_fps / downsample
    
    return frames_gray, effective_fps, actual_frame_count


def detect_audio_peaks(audio_path: str) -> np.ndarray:
    """Detect audio peaks using the Onset Detection algorithm.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Array of times (in seconds) where audio peaks occur.
    """
    y, sr = librosa.load(audio_path)
    # Calculate the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


def find_local_max_indexes(arr: np.ndarray, fps: float) -> np.ndarray:
    """Find local maxima in an array (vectorized version).

    Note:
        Local maxima with an optical flow magnitude less than 0.1 are ignored
        to prevent static scenes from being incorrectly calculated as peaks.

    Args:
        arr: NumPy array of values to find local maxima in.
        fps: Frames per second, used to convert indexes to time.

    Returns:
        NumPy array of times (in seconds) where local maxima occur.
    """
    n = len(arr)
    if n < 3:
        return np.array([])
    
    # Vectorized comparison: find where arr[i-1] < arr[i] > arr[i+1]
    is_local_max = (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]) & (arr[1:-1] >= 0.1)
    
    # Get indices (add 1 because we're comparing arr[1:-1])
    local_max_indices = np.where(is_local_max)[0] + 1
    
    # Convert to time
    return local_max_indices / fps


def compute_optical_flow_batch(frames_gray: np.ndarray) -> np.ndarray:
    """Compute optical flow magnitudes for consecutive frame pairs (vectorized).

    Args:
        frames_gray: NumPy array of grayscale frames with shape (N, H, W).

    Returns:
        NumPy array of average optical flow magnitudes for each frame pair.
        Length is N (first element is computed from frames 0->1).
    """
    n_frames = len(frames_gray)
    if n_frames < 2:
        return np.array([0.0])
    
    flow_magnitudes = np.zeros(n_frames, dtype=np.float32)
    
    # Compute optical flow for each consecutive pair
    for i in range(n_frames - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i], frames_gray[i + 1], None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_magnitudes[i] = magnitude.mean()
    
    # Last frame uses the same value as second-to-last transition
    flow_magnitudes[-1] = flow_magnitudes[-2] if n_frames > 1 else 0.0
    
    return flow_magnitudes


def detect_video_peaks(frames_gray: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Detect video peaks using Optical Flow (optimized version).

    Args:
        frames_gray: NumPy array of grayscale frames with shape (N, H, W).
        fps: Frame rate of the video.

    Returns:
        A tuple of (flow_trajectory, video_peaks) where flow_trajectory is a 
        NumPy array of optical flow magnitudes for each frame and video_peaks 
        is a NumPy array of times (in seconds) where video peaks occur.
    """
    flow_trajectory = compute_optical_flow_batch(frames_gray)
    video_peaks = find_local_max_indexes(flow_trajectory, fps)
    return flow_trajectory, video_peaks


def calculate_iou(audio_peaks: np.ndarray, video_peaks: np.ndarray, fps: float) -> float:
    """Calculate Intersection over Union (IoU) between audio and video peaks (optimized).

    Note:
        A video peak is matched to at most one audio peak to ensure that a 
        single video peak does not correspond to multiple audio peaks.

    Args:
        audio_peaks: Array of audio peak times (in seconds).
        video_peaks: Array of video peak times (in seconds).
        fps: Frame rate of the video.

    Returns:
        Intersection over Union score.
    """
    if len(audio_peaks) == 0 or len(video_peaks) == 0:
        return 0.0
    
    window = 1.0 / fps
    intersection_length = 0
    used_video_peaks = np.zeros(len(video_peaks), dtype=bool)
    
    # For each audio peak, find the closest unused video peak within window
    for audio_peak in audio_peaks:
        # Vectorized distance calculation
        distances = np.abs(video_peaks - audio_peak)
        
        # Find candidates within window that haven't been used
        valid_mask = (distances < window) & (~used_video_peaks)
        
        if np.any(valid_mask):
            # Get the closest valid peak
            valid_indices = np.where(valid_mask)[0]
            closest_idx = valid_indices[np.argmin(distances[valid_indices])]
            
            intersection_length += 1
            used_video_peaks[closest_idx] = True
    
    union_length = len(audio_peaks) + len(video_peaks) - intersection_length
    if union_length == 0:
        return 0.0
    
    return intersection_length / union_length


def _extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to the video file.
        audio_path: Path where the extracted audio should be saved.
    """
    import subprocess
    
    subprocess.run([
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '44100', '-ac', '2', audio_path, '-y'
    ], check=True, capture_output=True)


def evaluate_single_video(
    video_path: str,
    audio_path: str | None = None,
    temp_dir: Path | None = None,
    downsample: int = 1,
) -> Dict:
    """Evaluate AV-Align score for a single video.

    Args:
        video_path: Path to the video file.
        audio_path: Path to the audio file. If None, audio will be extracted 
                   from the video.
        temp_dir: Temporary directory for audio extraction. Required if 
                 audio_path is None.
        downsample: Factor to downsample video frames (e.g., 2 = every other frame).
                   Higher values = faster but less accurate.

    Returns:
        Dictionary containing the evaluation results with keys:
        - iou_score: The AV-Align IoU score
        - num_audio_peaks: Number of detected audio peaks
        - num_video_peaks: Number of detected video peaks
        - fps: Frame rate of the video (after downsampling)
        - downsample_factor: The downsampling factor used
    """
    # Extract audio if not provided
    need_cleanup = False
    if audio_path is None or not Path(audio_path).exists():
        if temp_dir is None:
            raise ValueError("temp_dir must be provided when audio_path is None")
        audio_path = str(temp_dir / "extracted_audio.wav")
        _extract_audio_from_video(video_path, audio_path)
        need_cleanup = True
    
    try:
        # Extract frames from video (optimized: grayscale only, with downsampling)
        frames_gray, fps, num_frames = extract_frames_optimized(video_path, downsample=downsample)
        
        # Detect audio peaks
        audio_peaks = detect_audio_peaks(audio_path)
        
        # Detect video peaks
        flow_trajectory, video_peaks = detect_video_peaks(frames_gray, fps)
        
        # Calculate IoU score
        iou_score = calculate_iou(audio_peaks, video_peaks, fps)
        
        result = {
            "iou_score": float(iou_score),
            "num_audio_peaks": int(len(audio_peaks)),
            "num_video_peaks": int(len(video_peaks)),
            "fps": float(fps),
            "downsample_factor": int(downsample),
        }
    except Exception as exc:
        result = {"error": str(exc)}
    finally:
        # Clean up extracted audio if needed
        if need_cleanup and Path(audio_path).exists():
            Path(audio_path).unlink()
    
    return result


def evaluate(
    data_list: Iterable[VideoData],
    device: str = "cuda",
    batch_size: int | None = None,
    model_args: Dict | None = None,
    sampling: int | None = None,
) -> List[VideoData]:
    """Evaluate AV-Align score for each :class:`VideoData`.

    Args:
        data_list: Iterable of VideoData objects to evaluate.
        device: Device to use (not used by this metric).
        batch_size: Batch size (not used by this metric).
        model_args: Additional model arguments (optional).
                   - subject_name: Custom result key name (default: "av_align")
                   - downsample: Frame downsampling factor (default: 2)
                                Higher = faster but less accurate
                                1 = no downsampling (slowest, most accurate)
                                2 = every other frame (2x faster)
                                4 = every 4th frame (4x faster)
        sampling: Sampling parameter (not used by this metric).

    Returns:
        List of VideoData objects with AV-Align results registered.
    """
    del device, batch_size, sampling  # These parameters are not used by this metric.
    model_args = model_args or {}
    
    subject_name = model_args.get("subject_name", SUBJECT_NAME)
    downsample = int(model_args.get("downsample", DEFAULT_DOWNSAMPLE_FACTOR))
    
    # Validate downsample factor
    if downsample < 1:
        downsample = 1
    
    for video_data in tqdm(data_list, desc=f"Evaluating AV-Align (downsample={downsample}x)"):
        with tempfile.TemporaryDirectory(prefix="av_align_eval_") as tmpdir:
            temp_dir = Path(tmpdir)
            
            result = evaluate_single_video(
                video_path=video_data.video_path,
                audio_path=video_data.audio_path,
                temp_dir=temp_dir,
                downsample=downsample,
            )
            
            video_data.register_result(subject_name, result)
    
    return list(data_list)
