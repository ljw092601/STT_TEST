"""
실시간 한국어 STT 테스트 (Whisper Large v3 @ Elice Cloud) — 고정 청크 방식

마이크 입력을 N초 단위로 잘라 /v1/audio/transcriptions 로 전송하고
결과 텍스트를 순서대로 출력합니다. (VAD 방식은 realtime_stt.py 참고)

사용법:
  1. .env.example 을 .env 로 복사 후 STT_API_BASE_URL, STT_API_KEY 입력
  2. python realtime_stt_chunk.py            # 기본 5초 청크
     python realtime_stt_chunk.py --chunk 3  # 3초 청크
  3. Ctrl+C 로 종료
"""
import argparse
import io
import os
import queue
import sys
import threading
import time
import wave

import numpy as np
import requests
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper 권장 샘플레이트
CHANNELS = 1


def load_env():
    """간단한 .env 로더 (python-dotenv 없이)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def pcm_to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def transcribe(wav_bytes: bytes, base_url: str, api_key: str) -> str:
    url = f"https://{base_url}/v1/audio/transcriptions"
    files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
    data = {"model": "whisper-large-v3", "language": "korean", "return_timestamps": "true"}
    headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}
    r = requests.post(url, files=files, data=data, headers=headers, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    return (body.get("transcript") or {}).get("text", "").strip()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=float, default=5.0, help="청크 길이(초)")
    ap.add_argument("--silence", type=float, default=0.005, help="무음 판정 RMS 임계값 (0이면 항상 전송)")
    ap.add_argument("--device", type=int, default=None, help="입력 장치 번호 (생략 시 기본 마이크)")
    ap.add_argument("--list-devices", action="store_true", help="오디오 장치 목록 출력")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    base_url = os.environ.get("STT_API_BASE_URL", "").replace("http://", "").replace("https://", "").rstrip("/")
    api_key = os.environ.get("STT_API_KEY", "")
    if not base_url or not api_key:
        sys.exit("STT_API_BASE_URL / STT_API_KEY 가 설정되지 않았습니다. (.env 참고)")

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    stop = threading.Event()

    def worker():
        idx = 0
        while not stop.is_set() or not audio_q.empty():
            try:
                chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            idx += 1
            t0 = time.time()
            try:
                text = transcribe(pcm_to_wav_bytes(chunk), base_url, api_key)
                dt = time.time() - t0
                print(f"[{idx:03d}] ({dt:.1f}s) {text or '(인식된 텍스트 없음)'}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{idx:03d}] 오류: {e}", flush=True)

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    frames_per_chunk = int(SAMPLE_RATE * args.chunk)
    buffer = []
    buffered = 0

    def callback(indata, frames, time_info, status):
        nonlocal buffer, buffered
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        buffer.append(indata.copy())
        buffered += frames
        if buffered >= frames_per_chunk:
            chunk = np.concatenate(buffer)[:frames_per_chunk]
            buffer, buffered = [], 0
            rms = np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
            if rms >= args.silence:
                audio_q.put(chunk)
            else:
                print(f"      (무음 {args.chunk:.0f}s 건너뜀, rms={rms:.4f})", flush=True)

    print(f"녹음 시작 — {args.chunk:.0f}초 단위 전송, Ctrl+C 로 종료")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                            device=args.device, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료 중... 남은 청크 처리 대기")
        stop.set()
        th.join(timeout=70)


if __name__ == "__main__":
    main()
