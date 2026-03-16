#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end audio dataset preparation pipeline:
1) Ingest mixed-format audio files
2) Optional standardization to mono 16k PCM WAV
3) VAD-based chunking (~60 sec target, can be longer if needed)
4) Gemini transcription + speaker diarization
5) Idempotent CSV writing with fsync crash-safety

Input manifest CSV columns:
- file_path
- original_file   (or same as file_path if unavailable)

Output files:
- metadata_chunks.csv        (chunk-level metadata)
- labeled_metadata.csv       (chunk-level labels with transcription)

Requirements:
pip install google-generativeai pydub webrtcvad

System dependency:
- ffmpeg must be installed in environment (Colab: apt install ffmpeg)
"""

import argparse
import csv
import math
import os
import random
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import google.generativeai as genai
from pydub import AudioSegment
import webrtcvad


SYSTEM_PROMPT = (
    "Sen Call Center audiolari bo'yicha mutaxassissan. Audioni eshit va har bir gapni "
    "kim gapirayotganini aniqlab, dialog ko'rinishida yoz. Format:\n"
    "Agent: [text]\n"
    "Mijoz: [text]\n"
    "Qoidalar: Raqamlarni so'z bilan yoz. Hech qanday qo'shimcha izoh yozma. "
    "Agar inson ovozi umuman bo'lmasa, qat'iy ravishda '[SILENCE]' deb javob ber."
)

CHUNKS_COLUMNS = ["file_path", "original_file"]
LABELED_COLUMNS = ["file_path", "original_file", "transcription"]


@dataclass
class VADConfig:
    mode: int = 2                 # 0..3 (3 is most aggressive)
    frame_ms: int = 30            # 10, 20, or 30 only for webrtcvad
    padding_ms: int = 300         # voiced region smoothing
    min_chunk_sec: float = 20.0   # avoid tiny chunks
    target_chunk_sec: float = 60.0
    max_chunk_sec: float = 95.0   # can be larger if speech continues
    min_speech_ratio: float = 0.06  # if too silent, mark as silence chunk


def parse_args():
    p = argparse.ArgumentParser(description="Prepare dataset from mixed audio -> VAD chunks -> Gemini labels")
    p.add_argument("--input-csv", required=True, help="CSV with columns: file_path, original_file")
    p.add_argument("--work-dir", default="work_audio", help="Working directory for standardized/chunked audio")
    p.add_argument("--metadata-chunks", default="metadata_chunks.csv", help="Output chunks metadata CSV")
    p.add_argument("--labeled-output", default="labeled_metadata.csv", help="Output labeled CSV")
    p.add_argument("--model", default="gemini-1.5-pro", help="Gemini model name")
    p.add_argument("--no-standardize", action="store_true", help="Skip explicit standardization step")
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--base-delay", type=float, default=1.5)
    p.add_argument("--max-delay", type=float, default=60.0)
    p.add_argument("--vad-mode", type=int, default=2)
    p.add_argument("--target-sec", type=float, default=60.0)
    p.add_argument("--max-sec", type=float, default=95.0)
    return p.parse_args()


def ensure_csv_header(path: str, columns: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            f.flush()
            os.fsync(f.fileno())


def append_row_fsync(path: str, columns: List[str], row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_processed_set(path: str, key_col: str = "file_path") -> Set[str]:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or key_col not in r.fieldnames:
            return done
        for row in r:
            v = (row.get(key_col) or "").strip()
            if v:
                done.add(v)
    return done


def standardize_to_wav(input_path: str, out_path: str, sample_rate: int = 16000) -> str:
    """
    Convert arbitrary audio to mono PCM16 WAV (best for VAD + ASR).
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)  # PCM16
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    audio.export(out_path, format="wav")
    return out_path


def read_wav_pcm16(path: str) -> Tuple[bytes, int]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        if channels != 1 or sample_width != 2:
            raise ValueError(f"WAV must be mono PCM16. Got channels={channels}, sample_width={sample_width}")
        pcm = wf.readframes(wf.getnframes())
    return pcm, rate


def frame_generator(frame_ms: int, pcm: bytes, sample_rate: int) -> Iterator[bytes]:
    n = int(sample_rate * (frame_ms / 1000.0) * 2)  # 2 bytes per sample (PCM16 mono)
    offset = 0
    while offset + n <= len(pcm):
        yield pcm[offset: offset + n]
        offset += n


def vad_segments_ms(pcm: bytes, sample_rate: int, cfg: VADConfig) -> List[Tuple[int, int]]:
    """
    Returns speech segments in milliseconds using WebRTC VAD + padding.
    """
    vad = webrtcvad.Vad(cfg.mode)
    frames = list(frame_generator(cfg.frame_ms, pcm, sample_rate))
    if not frames:
        return []

    is_speech = [vad.is_speech(fr, sample_rate) for fr in frames]
    frame_ms = cfg.frame_ms

    # Smooth with simple hangover/padding
    pad_frames = max(1, cfg.padding_ms // frame_ms)
    smoothed = is_speech[:]
    for i in range(len(is_speech)):
        left = max(0, i - pad_frames)
        right = min(len(is_speech), i + pad_frames + 1)
        if any(is_speech[left:right]):
            smoothed[i] = True

    segments = []
    start = None
    for i, flag in enumerate(smoothed):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            end = i
            segments.append((start * frame_ms, end * frame_ms))
            start = None
    if start is not None:
        segments.append((start * frame_ms, len(smoothed) * frame_ms))

    return segments


def merge_and_chunk_segments(segments: List[Tuple[int, int]], total_ms: int, cfg: VADConfig) -> List[Tuple[int, int]]:
    """
    Merge speech segments and split around target/max durations.
    """
    if not segments:
        # All silence fallback: one chunk around target or full audio if shorter
        return [(0, min(total_ms, int(cfg.target_chunk_sec * 1000)))]

    chunks = []
    cur_start, cur_end = segments[0]

    for s, e in segments[1:]:
        # join close segments (<= 800ms gap)
        if s - cur_end <= 800:
            cur_end = e
        else:
            chunks.extend(split_long_segment(cur_start, cur_end, cfg))
            cur_start, cur_end = s, e

    chunks.extend(split_long_segment(cur_start, cur_end, cfg))

    # ensure min length: merge too-short with previous
    merged = []
    min_ms = int(cfg.min_chunk_sec * 1000)
    for s, e in chunks:
        if merged and (e - s) < min_ms:
            ps, pe = merged[-1]
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    return merged


def split_long_segment(s: int, e: int, cfg: VADConfig) -> List[Tuple[int, int]]:
    out = []
    target_ms = int(cfg.target_chunk_sec * 1000)
    max_ms = int(cfg.max_chunk_sec * 1000)
    length = e - s
    if length <= max_ms:
        return [(s, e)]
    cur = s
    while cur < e:
        nxt = min(cur + target_ms, e)
        # allow expansion up to max
        if (e - nxt) < int(cfg.min_chunk_sec * 1000) and (e - cur) <= max_ms:
            nxt = e
        out.append((cur, nxt))
        cur = nxt
    return out


def build_chunks_for_audio(
    in_audio: str,
    original_file: str,
    out_dir: str,
    cfg: VADConfig,
    force_standardize: bool = True
) -> List[Tuple[str, str]]:
    """
    Returns list of (chunk_path, original_file)
    """
    base = Path(in_audio).stem
    std_path = str(Path(out_dir) / "standardized" / f"{base}.wav")

    if force_standardize:
        wav_path = standardize_to_wav(in_audio, std_path, sample_rate=16000)
    else:
        # assume already proper wav
        wav_path = in_audio

    pcm, sr = read_wav_pcm16(wav_path)
    total_ms = int((len(pcm) / 2) / sr * 1000)

    segments = vad_segments_ms(pcm, sr, cfg)
    chunks_ms = merge_and_chunk_segments(segments, total_ms, cfg)

    audio = AudioSegment.from_wav(wav_path)
    chunk_dir = Path(out_dir) / "chunks" / base
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_rows = []
    for i, (s, e) in enumerate(chunks_ms, start=1):
        cpath = chunk_dir / f"{base}_chunk_{i:04d}.wav"
        piece = audio[s:e]
        piece.export(cpath, format="wav")
        chunk_rows.append((str(cpath), original_file))

    return chunk_rows


def is_retryable_error(exc: Exception) -> bool:
    txt = str(exc).lower()
    retry_tokens = [
        "429", "rate limit", "quota", "resource exhausted", "too many requests",
        "temporarily unavailable", "service unavailable", "internal error", "deadline exceeded"
    ]
    return any(t in txt for t in retry_tokens)


def gemini_label_chunk(model_name: str, audio_path: str, max_retries: int, base_delay: float, max_delay: float) -> str:
    attempt = 0
    while True:
        uploaded = None
        try:
            uploaded = genai.upload_file(path=audio_path)
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content([SYSTEM_PROMPT, uploaded])
            text = (getattr(resp, "text", "") or "").strip()
            return text if text else "[EMPTY_RESPONSE]"
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not is_retryable_error(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, 0.25 * delay)
            print(f"[WARN] Retry {attempt}/{max_retries} for {audio_path}: {exc} -> sleep {delay:.2f}s", file=sys.stderr)
            time.sleep(delay)
        finally:
            if uploaded is not None:
                try:
                    genai.delete_file(uploaded.name if hasattr(uploaded, "name") else uploaded)
                except Exception as de:
                    print(f"[WARN] delete_file failed for {audio_path}: {de}", file=sys.stderr)


def main():
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")

    genai.configure(api_key=api_key)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    ensure_csv_header(args.metadata_chunks, CHUNKS_COLUMNS)
    ensure_csv_header(args.labeled_output, LABELED_COLUMNS)

    processed_chunks = read_processed_set(args.labeled_output, key_col="file_path")
    existing_metadata_chunks = read_processed_set(args.metadata_chunks, key_col="file_path")

    cfg = VADConfig(
        mode=args.vad_mode,
        target_chunk_sec=args.target_sec,
        max_chunk_sec=args.max_sec
    )

    # Stage 1: Build chunks metadata (idempotent)
    with open(args.input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "file_path" not in reader.fieldnames or "original_file" not in reader.fieldnames:
            raise ValueError("input CSV must have columns: file_path, original_file")

        for row in reader:
            src = (row.get("file_path") or "").strip()
            orig = (row.get("original_file") or "").strip() or src
            if not src:
                continue
            if not os.path.exists(src):
                print(f"[ERROR] Missing input audio: {src}", file=sys.stderr)
                continue

            try:
                chunk_rows = build_chunks_for_audio(
                    in_audio=src,
                    original_file=orig,
                    out_dir=str(work_dir),
                    cfg=cfg,
                    force_standardize=(not args.no_standardize)
                )
                for cpath, corig in chunk_rows:
                    if cpath not in existing_metadata_chunks:
                        append_row_fsync(args.metadata_chunks, CHUNKS_COLUMNS, {
                            "file_path": cpath,
                            "original_file": corig
                        })
                        existing_metadata_chunks.add(cpath)
                print(f"[OK] Chunked: {src} -> {len(chunk_rows)} chunk(s)", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Chunking failed for {src}: {e}", file=sys.stderr)

    # Stage 2: Label chunks with Gemini (idempotent)
    with open(args.metadata_chunks, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpath = (row.get("file_path") or "").strip()
            orig = (row.get("original_file") or "").strip()

            if not cpath or cpath in processed_chunks:
                continue
            if not os.path.exists(cpath):
                print(f"[ERROR] Missing chunk: {cpath}", file=sys.stderr)
                continue

            try:
                txt = gemini_label_chunk(
                    model_name=args.model,
                    audio_path=cpath,
                    max_retries=args.max_retries,
                    base_delay=args.base_delay,
                    max_delay=args.max_delay
                )
                append_row_fsync(args.labeled_output, LABELED_COLUMNS, {
                    "file_path": cpath,
                    "original_file": orig,
                    "transcription": txt
                })
                processed_chunks.add(cpath)
                print(f"[OK] Labeled: {cpath}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Gemini failed for {cpath}: {e}", file=sys.stderr)

    print("[DONE] Pipeline finished.", file=sys.stderr)


if __name__ == "__main__":
    main()
