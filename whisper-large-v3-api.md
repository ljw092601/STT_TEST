# Whisper Large v3 — 모델 라이브러리 정리

> 출처: Elice Cloud 모델 라이브러리 (KakaoTech Campus) — Whisper Large v3 상세 정보

| 항목 | 내용 |
|---|---|
| 모델명 | Whisper Large v3 (추천 모델) |
| 모델 ID | `openai/whisper-large-v3` |
| 모델 제공자 | OpenAI |
| 모델 종류 | Speech To Text |
| 사용 방식 | Serverless / Dedicated |
| 요금 (Serverless) | ₩6 / 60초 |
| 요금 (Dedicated) | 인스턴스 사용시간 기준 |
| API Rate Limit | 무제한 |

---

## 개요

BentoML로 패키징된 오픈소스 Whisper Large v3 모델을 호스팅하는 API 엔드포인트입니다.
OpenAI의 `/v1/audio/transcriptions` API 구조와 유사하지만, 오픈소스 모델 특성에 맞춰 일부 매개변수 규격이 최적화되어 있습니다.

- 고성능 자동 음성 인식(ASR) 및 번역 기능 제공
- REST API 방식으로 애플리케이션과 통합
- 타임스탬프 생성 시 `return_timestamps` 옵션 사용

## API 엔드포인트

| 기능 | 메서드 | 경로 |
|---|---|---|
| 음성 인식 | POST | `http://mlapi.run/abc-1234-xyz/v1/audio/transcriptions` |
| 음성 번역 | POST | `http://mlapi.run/abc-1234-xyz/v1/audio/translations` |

> `abc-1234-xyz` 부분은 예시 URL이며, 실제 발급받은 서비스 주소로 대체해야 합니다.

## 인증

Bearer Token 방식을 사용합니다. 요청 헤더에 API 키를 포함해야 합니다.

```
Authorization: Bearer <your-service-api-key>
```

---

## 사용법

### 설치

```bash
pip install requests
```

### 기본 설정

```python
import requests

# API 설정
API_BASE_URL = "mlapi.run/abc-1234-xyz"  # 예시 URL
API_KEY = "your-service-api-key"
```

---

## 예제 코드

### 1. 기본 음성 텍스트 변환 (Transcription)

오디오 파일을 전송하여 텍스트로 변환합니다.

```python
import requests

url = f"http://{API_BASE_URL}/v1/audio/transcriptions"
file_path = "speech.mp3"

files = {
    'file': (file_path, open(file_path, 'rb'), 'audio/mpeg')
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "accept": "application/json"
}

response = requests.post(url, files=files, headers=headers)
response.raise_for_status()

print(response.json())
```

### 2. 단어 단위 타임스탬프 및 언어 지정

단어 레벨의 타임스탬프가 필요한 경우 `return_timestamps`에 `"word"`를 **문자열**로 전달합니다.

```python
import requests

url = f"http://{API_BASE_URL}/v1/audio/transcriptions"
file_path = "meeting.mp3"

files = {
    'file': ('meeting.mp3', open(file_path, 'rb'), 'audio/mpeg')
}

# 스펙에 맞춘 데이터 구성
data = {
    'model': 'whisper-large-v3',
    'language': 'korean',          # 입력 언어 (선택 사항)
    'return_timestamps': 'word'    # "word" 입력 시 단어 단위 타임스탬프 반환
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "accept": "application/json"
}

# multipart/form-data 전송 (requests가 Boundary 자동 처리)
response = requests.post(url, files=files, data=data, headers=headers)
response.raise_for_status()
result = response.json()

# 결과 확인
print(result)
```

### 3. 음성 번역 (Translation)

영어가 아닌 오디오를 영어 텍스트로 번역하려면 `/v1/audio/translations` 엔드포인트를 사용합니다.

```python
url = f"http://{API_BASE_URL}/v1/audio/translations"
# ... 파일 및 헤더 설정 동일 ...

response = requests.post(url, files=files, headers=headers)
print(response.json())
```

---

## 응답 예시 (Response Example)

`/v1/audio/transcriptions` 요청 시 반환되는 JSON 구조입니다. 타임스탬프 정보(`chunks`)와 함께 변환된 텍스트가 제공됩니다.

```json
{
  "_result": {
    "status": "ok",
    "reason": null
  },
  "transcript": {
    "text": "안녕하십니까. 오늘의 주요 소식입니다. 전국 곳곳에 봄비가 내리는 가운데, 출근길 교통안전에 각별한 주의가 필요합니다.",
    "chunks": [
      {
        "timestamp": [
          0,
          10.74
        ],
        "text": "안녕하십니까. 오늘의 주요 소식입니다. 전국 곳곳에 봄비가 내리는 가운데, 출근길 교통안전에 각별한 주의가 필요합니다."
      }
    ]
  }
}
```

---

## 지원되는 매개변수

API 스펙(`openapi.json`)에 정의된 매개변수 목록입니다.

| 매개변수 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| `file` | formData | Binary | Yes | - | 변환할 오디오 파일 (Key 이름: `file`) |
| `model` | formData | String | No | `whisper-large-v3` | 사용할 모델 이름 |
| `language` | formData | String | No | `english` / `korean` | 오디오의 발화 언어 |
| `return_timestamps` | formData | Bool/Str | No | `true` | 타임스탬프 반환 옵션 — `true`: 문장 단위 / `"word"`: 단어 단위 |
