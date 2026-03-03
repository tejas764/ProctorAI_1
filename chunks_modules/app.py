from __future__ import annotations

import asyncio
import os
from typing import Any

import cv2
import pandas as pd

from chunks_modules.batch_processing import process_chunk_async
from chunks_modules.config import CONFIG_PATH, LIVE_MODE, NUM_CORES, ONLY_VAD, VIDEO_PATH
from chunks_modules.live_overlay import run_live_voice_overlay
from chunks_modules.media import Pipeline, extract_audio, get_vad_segments, is_vad_speaking


async def main() -> None:
    if LIVE_MODE:
        run_live_voice_overlay()
        return
    if Pipeline is Any:
        raise ImportError(
            "pyannote.audio is required when live_mode is False. "
            "Use Python 3.10/3.11 and install pyannote.audio, or set live_mode: True."
        )

    audio, duration = extract_audio(VIDEO_PATH)
    pipeline = Pipeline.from_pretrained(CONFIG_PATH)
    vad_segments = get_vad_segments(pipeline, audio)

    if ONLY_VAD:
        results = []
        for t in range(duration + 1):
            status = is_vad_speaking(t, vad_segments)
            results.append({"Time (s)": t, "VAD Speaking": status})
        del results
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        chunk_sz = total_frames // NUM_CORES
        tasks = [
            process_chunk_async(
                i * chunk_sz,
                (i + 1) * chunk_sz if i < NUM_CORES - 1 else total_frames,
                VIDEO_PATH,
                audio,
                fps,
                vad_segments,
            )
            for i in range(NUM_CORES)
        ]
        chunk_results = await asyncio.gather(*tasks)
        flat = [item for sub in chunk_results for item in sub]
        df = pd.DataFrame(flat).rename(
            columns={
                "Time (s)": "timestamp",
                "VAD Speaking": "vad_status",
                "Mahalanobis Status": "mahalanobis",
                "Rule-Based": "rule_based",
                "Optical Flow": "optical_flow",
                "AV Correlation": "av_correlation",
                "MAR Mean": "mar_mean",
                "Flow Mean": "flow_mean",
                "Corr Score Mean": "corr_score_mean",
                "Ensemble": "ensemble",
            }
        )
        df.to_csv("raw_metrics.csv", index=False)
        print(
            "Saved raw_metrics.csv with timestamp, vad_status, mahalanobis, "
            "rule_based, optical_flow, av_correlation, mar_mean, flow_mean, corr_score_mean, ensemble columns."
        )

    if os.path.exists(audio):
        os.remove(audio)

