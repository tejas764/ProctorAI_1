from __future__ import annotations

from typing import Any

import moviepy.editor as moviepy

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = Any


def extract_audio(video_path: str, audio_output: str = "temp_audio.wav") -> tuple[str, int]:
    clip = moviepy.VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_output, codec="pcm_s16le")
    return audio_output, int(clip.duration)


def get_vad_segments(pipeline: Pipeline, audio_path: str) -> list[tuple[float, float]]:
    vad = pipeline(audio_path)
    return [(seg.start, seg.end) for seg in vad.get_timeline().support()]


def is_vad_speaking(time_s: float, vad_segments: list[tuple[float, float]]) -> bool:
    return any(start <= time_s <= end for start, end in vad_segments)

