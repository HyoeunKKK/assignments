# KWS Assignment

이 프로젝트는 Google Speech Commands 계열 데이터를 사용해 12개 클래스를 분류하는 keyword spotting 학습 코드다. 입력 오디오는 log-mel spectrogram으로 변환되고, 모델은 depthwise separable convolution 기반의 `DSCNN`을 사용한다.

## 1. 문제 설정

분류 대상 클래스는 총 12개다.

- `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`
- `silence`
- `unknown`

`unknown`은 타겟 10개 단어가 아닌 나머지 단어들을 하나의 클래스로 묶은 것이고, `silence`는 실제 음성 파일 대신 0으로 채운 무음 파형을 사용한다.

## 2. 프로젝트 구조

```text
kws_assignment/
├── train.py                    # 학습/검증/테스트 전체 실행
├── configs/dscnn_xlarge.yaml   # 현재 사용 가능한 학습 설정
├── scripts/prepare_gsc.py      # GSC 데이터를 CSV split으로 정리
├── src/dataset.py              # CSV 기반 데이터셋 로더
├── src/augment.py              # waveform/spec augmentation
├── src/features.py             # log-mel feature 추출
└── src/models/dscnn.py         # DSCNN 모델 정의
```

참고로 `evaluate.py`, `infer.py`, `src/trainer.py`는 현재 비어 있고, 실제 학습 로직은 `train.py`에 직접 들어 있다.

## 3. 데이터 준비 방식

### 3.1 split 생성

`scripts/prepare_gsc.py`는 Google Speech Commands 루트 디렉터리를 읽어 `train.csv`, `valid.csv`, `test.csv`를 만든다.

동작 방식은 다음과 같다.

1. `validation_list.txt`, `testing_list.txt`를 읽어서 valid/test split을 결정한다.
2. 타겟 10개 단어는 원래 라벨 그대로 유지한다.
3. 타겟이 아닌 나머지 단어는 모두 `unknown`으로 합친다.
4. 각 split마다 `known` 샘플 수를 기준으로 `unknown_pct`, `silence_pct` 비율만큼 `unknown`, `silence` 샘플을 추가한다.
5. 결과를 `path,label` 형식 CSV로 저장한다.

`silence`는 CSV에 실제 파일 경로 대신 `__silence__`라는 특수 문자열로 기록된다.

예시:

```bash
python scripts/prepare_gsc.py --root /path/to/speech_commands_v0.02 --out_dir data/processed
```

기본 비율:

- `unknown_pct=10.0`
- `silence_pct=10.0`

## 4. 입력 전처리와 feature 추출

`src/dataset.py`의 `GSCDataset`이 CSV를 읽고 각 샘플을 다음 순서로 처리한다.

1. 오디오 로드
2. mono 변환
3. 필요하면 `torchaudio.functional.resample`로 16 kHz로 리샘플링
4. 1초(`16000` samples) 길이에 맞게 pad/trim
5. train split일 때 waveform augmentation 적용
6. log-mel spectrogram 추출
7. train split일 때 SpecAugment 적용
8. `[1, 80, T]` 형태 텐서와 클래스 인덱스 반환

### 4.1 waveform augmentation

`src/augment.py`에 다음 증강이 들어 있다.

- speed perturbation: `0.9`, `1.0`, `1.1` 중 선택, 확률 `0.5`
- random time shift: 최대 `±1600` samples, 즉 약 `±100 ms`
- random gain + Gaussian noise

단, `silence` 샘플에는 augmentation을 적용하지 않는다.

### 4.2 feature

`src/features.py`의 `LogMelExtractor`는 다음 설정으로 mel spectrogram을 만든다.

- sample rate: `16000`
- `n_fft=640`
- `win_length=640`
- `hop_length=320`
- `n_mels=80`

추출 후 `AmplitudeToDB`를 적용하고, 샘플별로 평균 0 / 표준편차 1이 되도록 정규화한다.

## 5. 모델 구조

모델은 `src/models/dscnn.py`의 `DSCNN`이다.

구조는 다음과 같다.

1. `stem`: `Conv2d(1 -> stem_out, 3x3, stride=2)` + BatchNorm + ReLU
2. 여러 개의 `DSConvBlock`
3. `AdaptiveAvgPool2d(1)`
4. Dropout
5. Linear classifier

`DSConvBlock`은 아래 두 층으로 구성된다.

- depthwise convolution (`groups=in_ch`)
- pointwise convolution (`1x1`)

즉, 일반 convolution 대신 depthwise separable convolution을 사용해서 파라미터 수와 연산량을 줄이는 구조다.

현재 설정 파일 `configs/dscnn_xlarge.yaml` 기준 모델 하이퍼파라미터는 아래와 같다.

```yaml
model:
  num_classes: 12
  dropout: 0.15
  channels: [256, 256, 384, 384, 640, 640, 768, 768]
  block_strides:
    - [1, 1]
    - [1, 1]
    - [2, 2]
    - [1, 1]
    - [2, 2]
    - [1, 1]
    - [1, 1]
```

학습 시작 시 `count_parameters()`로 trainable parameter 수를 계산하고, `2,500,000`개를 넘으면 assert로 중단한다.

## 6. 학습 루프

실제 학습은 `train.py`에서 수행된다.

### 6.1 전체 흐름

1. YAML config 로드
2. seed 고정
3. train/valid/test용 `GSCDataset`, `DataLoader` 생성
4. `DSCNN` 모델 생성
5. loss, optimizer, scheduler 설정
6. epoch마다 train 수행
7. 매 epoch 종료 후 valid 평가
8. 최고 valid accuracy 갱신 시 체크포인트 저장
9. 학습 종료 후 best checkpoint를 다시 로드해 test 평가

### 6.2 loss / optimizer / scheduler

- loss: `CrossEntropyLoss`
- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`

현재 config 기준 학습 설정:

```yaml
train:
  batch_size: 256
  num_workers: 4
  epochs: 120
  lr: 0.001
  weight_decay: 0.0001
  log_dir: runs/tb/dscnn_xlarge_120ep
  ckpt_path: runs/checkpoints/dscnn_xlarge_120ep_best.pt
```

### 6.3 metric과 기록

학습 중 기록하는 값은 다음과 같다.

- train loss / acc
- valid loss / acc
- learning rate
- gradient norm
- 최종 test loss / acc

로그는 두 군데에 저장된다.

- TensorBoard: `SummaryWriter`
- Weights & Biases: `wandb`

checkpoint에는 아래 정보가 저장된다.

- `model_state_dict`
- `config`
- `best_valid_acc`

## 7. 실행 방법

### 7.1 의존성 설치

프로젝트는 `pyproject.toml` 기준 Python `>=3.10`을 사용한다.

주요 의존성:

- `torch`
- `torchaudio`
- `numpy`
- `scipy`
- `pyyaml`
- `tensorboard`
- `wandb`

`uv.lock`이 있으므로 `uv`를 쓰는 경우 예시는 다음과 같다.

```bash
uv sync
```

### 7.2 데이터 split 생성

```bash
python scripts/prepare_gsc.py --root /path/to/speech_commands_v0.02 --out_dir data/processed
```

생성 결과:

- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/processed/test.csv`

### 7.3 학습 실행

현재 저장소에는 `configs/dscnn_xlarge.yaml`만 있으므로 config를 명시해서 실행하는 것이 맞다.

```bash
python train.py --config configs/dscnn_xlarge.yaml
```

주의:

- `train.py`의 기본값은 `configs/dscnn.yaml`인데, 현재 저장소에는 해당 파일이 없다.
- 따라서 실제 실행 시에는 `--config configs/dscnn_xlarge.yaml`를 반드시 주는 편이 안전하다.

## 8. 학습 중 생성되는 산출물

- TensorBoard 로그: `runs/tb/...`
- best checkpoint: `runs/checkpoints/...`
- W&B 로그: `wandb/`

## 9. 코드 기준 한 줄 요약

이 코드는 Google Speech Commands 데이터를 CSV split으로 정리한 뒤, 1초 음성 파형을 log-mel spectrogram으로 변환하고, waveform/spec augmentation을 적용해 DSCNN으로 학습하는 end-to-end keyword spotting 파이프라인이다.
