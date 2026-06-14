# Go2 Quadruped RL Locomotion

Unitree Go2 4족 보행 로봇의 보행(locomotion) 정책을 강화학습(PPO)으로 학습하는 프로젝트입니다.
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)의 PPO와 [MuJoCo](https://mujoco.org/) 물리 시뮬레이터를 사용합니다.

목표 속도 명령 `[vx, vy, wz]`를 따라 걷도록 학습하며, trot 보행 패턴과 발 높이(foot clearance) 등을 보상으로 유도합니다.

## Colab에서 실행

별도 설치 없이 브라우저에서 바로 실행할 수 있습니다. 아래 배지를 클릭하세요.
(런타임 → 런타임 유형 변경 → **T4 GPU** 권장)

| 노트북 | 설명 | 링크 |
|--------|------|------|
| `go2_locomotion_train.ipynb` | pretrained 모델 불러와 추가 학습 → 추론 → 영상 생성 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agbread/RL_tutorial/blob/main/notebooks/go2_locomotion_train.ipynb) |
| `go2_locomotion_reward_tuning.ipynb` | reward 3회 수정·재학습으로 보행 자세 개선 (라운드별 영상 비교) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agbread/RL_tutorial/blob/main/notebooks/go2_locomotion_reward_tuning.ipynb) |

> 노트북 첫 셀이 이 repo를 자동으로 clone하고 의존성을 설치합니다. Colab에서는 `/content` 기준으로 동작합니다.

## 로컬(오프라인) 설치 & 실행

검증 환경: **Python 3.10 / Ubuntu 22.04 / CUDA 11.7 GPU**

### 1) conda 환경 생성 + 패키지 설치

```bash
conda create -n go2 python=3.10 -y
conda activate go2
pip install -r requirements.txt
```

GPU(CUDA)를 쓰려면 `torch`는 환경에 맞는 빌드로 따로 설치하세요 (검증: `2.0.1+cu117`):

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
```

(CUDA 버전이 다르면 https://pytorch.org 에서 맞는 명령을 확인하세요.)

### 2) 학습 / 추론 (CLI)

```bash
conda activate go2
cd src
python train.py     # 학습 (옵션은 아래 "사용법" 참고)
python test.py      # 추론 + 영상 녹화
```

### 3) Jupyter 노트북 실행 (오프라인 버전)

```bash
conda activate go2
jupyter notebook    # 또는: jupyter lab
```

브라우저가 열리면 다음 노트북을 열어 위에서부터 실행하세요:

- `notebooks/go2_locomotion_train_offline.ipynb`
- `notebooks/go2_locomotion_reward_tuning_offline.ipynb`

로컬에서는 라이브 렌더링이 동작해 학습한 모델의 보행 영상이 직접 생성됩니다.
(Colab 버전 노트북은 렌더가 불안정해 사전 렌더링된 영상을 표시합니다.)

> 모델 `.zip`은 numpy 버전에 민감합니다. `requirements.txt` 의 numpy(1.26.4)를 그대로 사용하세요
> (다르면 `ModuleNotFoundError: No module named 'numpy.core_'` 등이 발생할 수 있습니다).

## 디렉터리 구조

```
RL_Tutorial/
├── src/
│   ├── train.py             # 학습 전용 엔트리포인트
│   ├── test.py              # 추론 전용: 학습된 모델 롤아웃 + 영상 녹화
│   ├── go2_mujoco_env.py    # Gymnasium 환경 (MuJoCo 기반 Go2)
│   ├── params.yaml          # 학습 하이퍼파라미터 (PPO, n_envs 등)
│   ├── envs.yaml            # 환경 설정 (보상 가중치, 명령 범위, 종료 조건 등)
│   ├── mdp/                 # 보상(reward) / 종료(termination) 로직
│   └── utils/               # 보상 로깅 콜백 등 유틸
├── unitree_go2/             # MuJoCo 모델 (scene XML + 메시 에셋)
├── models/                  # 학습된 체크포인트 (.zip)
├── logs/                    # TensorBoard 로그
├── notebooks/               # 분석 / 보상 ablation 노트북
└── recordings / *.mp4       # 롤아웃 영상
```

## 사용법

모든 명령은 `src/` 디렉터리에서 실행합니다.

```bash
cd src
```

학습은 `train.py`, 추론·영상 녹화는 `test.py`로 역할이 분리되어 있습니다.

### 학습 (train.py)

```bash
# 기본 학습 (모든 인자는 선택값, 미지정 시 params.yaml 사용)
python train.py

# 병렬 환경 수 / 총 timestep / 실행 이름 지정
python train.py --num_parallel_envs 12 --total_timesteps 2000000 --run_name my_run
```

- 모델은 `models/<날짜시각>` 또는 `models/<날짜시각>-<run_name>/` 에 저장됩니다 (`best_model.zip`, 체크포인트, `final_model.zip`).
- 학습 재개: `--model_path <기존모델.zip>` 또는 `params.yaml`의 `policy.use_pretrained: true`.
- TensorBoard: `tensorboard --logdir logs`

### 테스트 / 영상 녹화 (test.py)

학습된 모델을 롤아웃하고 mp4 영상을 저장합니다.

```bash
# 기본 모델(Go2_forth_test) 롤아웃 + 영상 저장
python test.py

# 모델 경로와 명령 속도를 지정해 롤아웃
python test.py --model_path models/<run_name>/best_model.zip --command 0.9 0.0 0.0

# 영상 없이 빠르게 확인
python test.py --model_path models/<run_name>/best_model.zip --no_video
```

주요 옵션: `--model_name`(기본 `Go2_forth_test`), `--max_time_step_s`(롤아웃 시간), `--control_hz`(기본 50), `--video_fps`, `--stochastic`(비결정적 추론), `--seed`.

## 설정 파일

- `src/params.yaml` — PPO 하이퍼파라미터(`learning_rate`, `n_steps`, `batch_size`, `gamma` 등), `n_envs`, `total_timestep`, `seed`, `eval_freq`.
- `src/envs.yaml` — 보상/패널티 가중치, 명령 속도 범위(`command.des_vel`), 보행 패턴(`gait`), 종료 조건(`termination`), 관측 스케일 등.

## 보상 설계 (요약)

- Positive: 선형/각속도 추종(`linear/angular_vel_tracking`), 생존(`healthy`), 기준 높이, 발 체공 시간(`feet_air_time`).
- Penalty: 토크, 수직/롤·피치 각속도, action rate, 관절 한계, trot 보행 강제(`gait_enforcement`), 발 높이(`foot_clearance`) 등.

자세한 값은 `src/envs.yaml`, 구현은 `src/mdp/reward.py` 참고.
