"""
실시간 한국어 STT 테스트 (Whisper Large v3 @ Elice Cloud) — VAD 방식

마이크 입력을 VAD(음성 활동 감지)로 감시하다가, 말이 시작되면 녹음하고
말이 끝나면(일정 시간 무음) 그 구간만 /v1/audio/transcriptions 로 전송합니다.
고정 길이 청크와 달리 단어 중간이 잘리지 않고, 짧게 말하면 바로 전송됩니다.

사용법:
  1. .env.example 을 .env 로 복사 후 STT_API_BASE_URL, STT_API_KEY 입력
  2. python realtime_stt.py
     python realtime_stt.py --aggr 3 --silence-ms 500   # 민감도/끊는 무음 길이 조정
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
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import webrtcvad

SAMPLE_RATE = 16000       # Whisper 권장, webrtcvad 지원 샘플레이트
CHANNELS = 1
FRAME_MS = 30             # webrtcvad 는 10/20/30ms 프레임만 허용
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


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
    files = {"file": ("segment.wav", wav_bytes, "audio/wav")}
    data = {"model": "whisper-large-v3", "language": "korean", "return_timestamps": "true"}
    headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}
    r = requests.post(url, files=files, data=data, headers=headers, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    return (body.get("transcript") or {}).get("text", "").strip()


class VadSegmenter:
    """30ms 프레임을 받아 발화 구간(np.int16 배열)을 잘라낸다.

    - 말 시작: 최근 프레임 중 음성 비율이 높아지면 시작. 그 직전 pre_ms 만큼도 같이 포함(첫 음절 보존)
    - 말 끝:   silence_ms 동안 계속 무음이면 종료
    - 안전장치: max_s 를 넘으면 강제로 끊어서 전송
    """

    def __init__(self, aggressiveness: int, pre_ms: int, silence_ms: int, min_ms: int, max_s: float):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.pre_frames = max(1, pre_ms // FRAME_MS)
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.min_samples = SAMPLE_RATE * min_ms // 1000
        self.max_samples = int(SAMPLE_RATE * max_s)
        self.ring = deque(maxlen=self.pre_frames)   # 발화 전 프레임 보관
        self.in_speech = False
        self.buf = []
        self.silent_run = 0

    def feed(self, frame: np.ndarray):
        """frame: FRAME_SAMPLES 길이 int16. 발화가 끝나면 세그먼트 반환, 아니면 None."""
        is_speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)

        if not self.in_speech:
            self.ring.append(frame)
            if is_speech:
                # 링버퍼(발화 직전) + 현재 프레임으로 시작
                self.in_speech = True
                self.buf = list(self.ring)
                self.silent_run = 0
            return None

        self.buf.append(frame)
        self.silent_run = 0 if is_speech else self.silent_run + 1

        total = len(self.buf) * FRAME_SAMPLES
        if self.silent_run >= self.silence_frames or total >= self.max_samples:
            return self._flush(forced=total >= self.max_samples)
        return None

    def _flush(self, forced: bool):
        seg = np.concatenate(self.buf)
        self.in_speech = False
        self.buf = []
        self.ring.clear()
        if forced:
            return seg
        # 뒤에 붙은 무음 구간은 조금만 남기고 잘라냄
        keep_tail = FRAME_SAMPLES * min(self.silent_run, 10)
        seg = seg[: len(seg) - FRAME_SAMPLES * self.silent_run + keep_tail]
        return seg if len(seg) >= self.min_samples else None

    def finish(self):
        """종료 시 진행 중이던 발화가 있으면 반환."""
        if self.in_speech and self.buf:
            return self._flush(forced=True)
        return None


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--aggr", type=int, default=2, choices=[0, 1, 2, 3],
                    help="VAD 민감도 0(관대)~3(엄격). 시끄러우면 3, 조용하면 1~2")
    ap.add_argument("--silence-ms", type=int, default=700, help="이만큼 무음이면 발화 종료로 판단")
    ap.add_argument("--pre-ms", type=int, default=300, help="발화 시작 직전 포함할 길이")
    ap.add_argument("--min-ms", type=int, default=400, help="이보다 짧은 발화는 버림(잡음 방지)")
    ap.add_argument("--max-s", type=float, default=15.0, help="한 발화 최대 길이, 넘으면 강제 전송")
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

    seg_q: "queue.Queue[np.ndarray]" = queue.Queue()
    stop = threading.Event()

    def worker():
        idx = 0
        while not stop.is_set() or not seg_q.empty():
            try:
                seg = seg_q.get(timeout=0.2)
            except queue.Empty:
                continue
            idx += 1
            dur = len(seg) / SAMPLE_RATE
            t0 = time.time()
            try:
                text = transcribe(pcm_to_wav_bytes(seg), base_url, api_key)
                dt = time.time() - t0
                print(f"[{idx:03d}] 발화 {dur:.1f}s / 응답 {dt:.1f}s ▶ {text or '(인식된 텍스트 없음)'}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{idx:03d}] 오류: {e}", flush=True)

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    segmenter = VadSegmenter(args.aggr, args.pre_ms, args.silence_ms, args.min_ms, args.max_s)
    state = {"speaking": False}

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        frame = indata[:, 0].copy()
        seg = segmenter.feed(frame)
        if segmenter.in_speech != state["speaking"]:
            state["speaking"] = segmenter.in_speech
            print("  ● 말하는 중..." if state["speaking"] else "  ○ 발화 종료 → 전송", flush=True)
        if seg is not None:
            seg_q.put(seg)

    print(f"녹음 시작 — VAD 민감도 {args.aggr}, 무음 {args.silence_ms}ms 이면 전송. Ctrl+C 로 종료")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                            blocksize=FRAME_SAMPLES, device=args.device, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료 중... 남은 발화 처리 대기")
        last = segmenter.finish()
        if last is not None:
            seg_q.put(last)
        stop.set()
        th.join(timeout=70)


if __name__ == "__main__":
    main()
