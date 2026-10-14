"""Bounded raw-video decoding with explicit timestamps and missing-stream status.

No network access and no learned prediction occurs in this module. Video times
refer to FFmpeg's resampled output timeline; audio time is sample_index / 16000.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np


def ffmpeg_binary() -> str:
    located = shutil.which('ffmpeg')
    if located:
        return located
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError('Install ffmpeg on PATH or the imageio-ffmpeg dependency.') from exc


def decode_video(path: str | Path, *, fps: float = 2.0, size: int = 224,
                 max_seconds: float = 120.0, timeout_seconds: float = 180.0,
                 audio_path: str | Path | None = None, allow_missing_video: bool = False) -> dict:
    path = Path(path).resolve()
    video_present = path.is_file()
    if not video_present and not (allow_missing_video and audio_path is not None):
        raise FileNotFoundError(path)
    audio_source = Path(audio_path).resolve() if audio_path is not None else path
    if not audio_source.is_file():
        raise FileNotFoundError(audio_source)
    if audio_path is not None and audio_source.stem != path.stem:
        raise ValueError('Separate audio and video must have the same exact clip ID stem')
    if not 0 < fps <= 30 or not 16 <= size <= 1024 or not 0 < max_seconds <= 3600:
        raise ValueError('Invalid bounded decoding settings')
    started = time.perf_counter()
    binary = ffmpeg_binary()
    # -nostdin prevents an unattended process waiting for terminal input.
    common = [binary, '-nostdin', '-hide_banner', '-i', str(path), '-t', str(max_seconds)]
    audio_common = [binary, '-nostdin', '-hide_banner', '-i', str(audio_source), '-t', str(max_seconds)]
    audio = subprocess.run(audio_common + ['-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
                                    '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=timeout_seconds)
    audio_stderr = audio.stderr.decode('utf-8', errors='replace')
    if audio.returncode:
        waveform = np.empty(0, dtype=np.float32)
        audio_status = 'missing_stream' if ('matches no streams' in audio_stderr or 'does not contain any stream' in audio_stderr) else 'decode_error'
    else:
        waveform = np.frombuffer(audio.stdout, dtype='<f4').copy()
        if not np.isfinite(waveform).all():
            raise ValueError('Decoded audio contains nonfinite samples')
        audio_status = 'present' if waveform.size else 'empty_stream'
    # Aspect ratio is preserved by scale+pad. This is not face tracking.
    filters = (f'fps={fps}:start_time=0,scale={size}:{size}:force_original_aspect_ratio=decrease,'
               f'pad={size}:{size}:(ow-iw)/2:(oh-ih)/2,showinfo')
    if video_present:
        video = subprocess.run(common + ['-map', '0:v:0', '-an', '-vf', filters, '-pix_fmt', 'rgb24',
                                        '-f', 'rawvideo', 'pipe:1'], capture_output=True, timeout=timeout_seconds)
        video_stderr = video.stderr.decode('utf-8', errors='replace')
    else:
        video = None
        video_stderr = 'video file missing from the official media inventory'
    if video is None or video.returncode:
        frames = np.empty((0, size, size, 3), dtype=np.uint8)
        timestamps = np.empty(0, dtype=np.float64)
        video_status = 'missing_file' if video is None else ('missing_stream' if 'matches no streams' in video_stderr else 'decode_error')
    else:
        raw = np.frombuffer(video.stdout, dtype=np.uint8)
        frame_bytes = size * size * 3
        if raw.size % frame_bytes:
            raise ValueError('Incomplete decoded RGB frame')
        frames = raw.reshape((-1, size, size, 3)).copy()
        timestamps = np.array([float(v) for v in re.findall(r'\bpts_time:\s*([-+0-9.eE]+)', video_stderr)], dtype=np.float64)
        if len(timestamps) != len(frames):
            raise ValueError('FFmpeg timestamps do not match decoded frame count')
        video_status = 'present' if len(frames) else 'empty_stream'
    match = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', audio_stderr + video_stderr)
    duration = (int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])) if match else None
    if video_present:
        with path.open('rb') as handle:
            source_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    else:
        source_sha256 = None
    if audio_source == path:
        audio_source_sha256 = source_sha256
    else:
        with audio_source.open('rb') as handle:
            audio_source_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'audio': waveform, 'frames': frames, 'frame_times': timestamps,
            'metadata': {'path': str(path), 'source_sha256': source_sha256,
                'audio_source_path': str(audio_source), 'audio_source_sha256': audio_source_sha256,
                'separate_audio_source': audio_source != path,
                'source_duration_seconds': duration, 'audio_sample_rate': 16000,
                'audio_samples': waveform.size, 'frame_count': len(frames),
                'frame_timestamps_kind': 'ffmpeg_resampled_output_timeline',
                'fps': fps, 'frame_size': size, 'audio_status': audio_status,
                'video_status': video_status, 'max_seconds': max_seconds,
                'truncated': duration > max_seconds if duration is not None else None,
                'decode_wall_seconds': time.perf_counter() - started,
                'audio_error': audio_stderr[-2000:] if audio.returncode else None,
                'video_error': video_stderr[-2000:] if video is None or video.returncode else None}}
