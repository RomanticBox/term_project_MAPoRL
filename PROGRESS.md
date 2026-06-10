# MAPoRL 구현 현황 및 작업 계획

> 최종 업데이트: 2026-06-10

---

## 1. 프로젝트 개요

**MAPoRL** (Multi-Agent Post-Co-Training for Collaborative LLMs with Reinforcement Learning)은 ACL 2025에 게재된 논문의 구현체입니다.  
기반 프레임워크는 HuggingFace TRL을 커스텀 확장한 버전(`trl/` 하위)이며, 학습 전체 파이프라인을 단일 Jupyter 노트북(`MAPoRL_tutorial.ipynb`)으로 설명·재현하는 것이 현재 목표입니다.

---

## 2. 기본 MAPoRL 구현 (논문 재현 범위)

### 2-1. 학습 파이프라인

| 구성 요소 | 파일 | 구현 상태 |
|---|---|---|
| 데이터셋 준비 (TinyGSM, ANLI R3) | `train_ppo_v2_multi_agent_multi_model.py` | ✅ 완료 |
| DataCollator (패딩 + open-ended 지원) | `trl/.../ppov2_trainer_multi_different_model.py:99` | ✅ 완료 |
| Multi-Adapter LoRA 모델 구조 | `utils/utils_model.py:ModelManager` | ✅ 완료 (GPU 필요) |
| PolicyAndValueWrapper | `ppov2_trainer_multi_different_model.py:132` | ✅ 완료 |
| 보상 시스템 (bonus_rule α, score_rule γ) | `utils_multi_unified.py` | ✅ 완료 |
| 멀티 에이전트 통신 프로토콜 | `utils_multi_unified_chat.py` | ✅ 완료 |
| PPO Rollout Phase | `ppov2_trainer_multi_different_model.py:754` | ✅ 완료 (GPU 필요) |
| PPO Update Phase (GAE + Clip Loss) | `ppov2_trainer_multi_different_model.py:1647` | ✅ 완료 |
| 컨센서스 조기 종료 | `ppov2_trainer_multi_different_model.py` | ✅ 완료 |
| 추론 파이프라인 | 노트북 Section 11 | ✅ 완료 (GPU 필요) |
| 평가 (라운드별 정확도, 전환 분석) | `utils/utils_cooperateLLM.py:stat_all()` | ✅ 완료 |

### 2-2. 학습 지원 스크립트

| 스크립트 | 역할 |
|---|---|
| `train_ppo_v2_multi_agent_multi_model.py` | 메인 학습 진입점 (accelerate + DeepSpeed) |
| `train_sft.py` | PPO 전 SFT 사전학습 |
| `reward_gen_data.py` | 보상 검증기 학습 데이터 생성 |
| `reward_data_balancing.py` | 검증기 데이터 클래스 불균형 보정 |
| `reward_server.py` | 보상 계산 전용 GPU 서버 (A24 분리 운용) |
| `train_reward.py` | 보상 검증기 파인튜닝 |
| `shell/training_turn2_3_gsm8k.sh` | 학습 실행 쉘 스크립트 |

### 2-3. 핵심 하이퍼파라미터 (GSM8K 기본 설정)

```
모델     : microsoft/Phi-3-mini-128k-instruct (4-bit 양자화)
에이전트 : 2개 (동일 모델, 서로 다른 LoRA 어댑터)
라운드   : 3 (Turn 0: policy 어댑터, Turn 1+: col 어댑터)
보상     : discounted_sum (γ=0.3) + α shaping [0.5, 0.5, 0.3, 0.3]
PPO      : clip ε=0.2, KL coef=0.002, epoch=4, vf_coef=0.1
학습 GPU : Titan RTX × 6 (DeepSpeed ZeRO-2)
```

---

## 3. 기본 구현을 넘어서 하려는 작업

### 3-1. 튜토리얼 노트북 (`MAPoRL_tutorial.ipynb`)

**목적:** 논문의 전체 파이프라인을 단일 `.ipynb`로 문서화 및 실험 비교.  
**분업:** 전체 구현 — 이 노트북 / 시각화·가독성 개선 — 나머지 팀원

| 섹션 | 상태 |
|---|---|
| 1. 논문 개요 | ✅ 완료 (개요 + α 보상 표) |
| 2. 환경 설정 | ✅ 완료 |
| 3. 데이터셋 준비 | ✅ 완료 |
| 4. DataCollator | ✅ 완료 |
| 5. 모델 아키텍처 | ✅ 완료 (Multi-Adapter LoRA 다이어그램) |
| 6. 보상 시스템 | ✅ 완료 |
| 7. 멀티 에이전트 통신 | ✅ 완료 |
| 8. PPO Rollout | ✅ 완료 (pseudocode) |
| 9. PPO Update (GAE + Loss) | ✅ 완료 |
| 10. 학습 설정 및 실행 | ✅ 완료 |
| 10.5. 실시간 대시보드 | ✅ 완료 (`MAPoRLDashboard` ipywidgets) |
| 11. 추론 파이프라인 | ✅ 완료 |
| 12. 평가 (stat_all) | ✅ 완료 (`compute_round_accuracy` 헬퍼) |
| **13. 실험 1 — 역할 분담 없음 비교** | ✅ 완료 (config 생성 + 평가 코드) |
| **14. 실험 2 — 비대칭 파라미터 크기** | ✅ 완료 (4가지 rank 조합 + 평가 코드) |

> **노트 (실행 가능 범위):**  
> - GPU 없이 실행 가능: 섹션 3, 4, 6, 7, 9, 10, 10.5, 12, 13(config 생성), 14(config 생성)  
> - GPU + 체크포인트 필요: 섹션 5(모델 초기화), 8(Rollout), 11(추론), 13/14(평가)

### 3-2. 실시간 학습 대시보드 (`MAPoRLDashboard`)

노트북 Section 10.5에 신규 추가된 기능.  
평가자(발표 심사 등)가 학습 진행 상황을 실시간으로 확인할 수 있도록 노트북 내 고정 패널 제공.

**표시 내용:**
- 현재 Phase 진행 바 (`초기화 → Rollout → Update → Evaluation → 완료`)
- Step / Turn / Agent / 경과 시간
- 라운드별 정확도 추이 (꺾은선 그래프)
- pg_loss / vf_loss 추이
- (Turn × Agent) 보상 그리드 (Base / Bonus α / Final, 정답 여부 색상 표시)
- 현재 배치의 에이전트 응답 및 추출 정답

**사용법 (실제 학습 루프 연동):**
```python
dash = MAPoRLDashboard(agent_num=2, round_num=3)

for turn in range(round_num):
    for agent in range(agent_num):
        dash.set_phase("Rollout", turn=turn, agent=agent, step=step)
        dash.log_agent_answer(turn, agent, q_idx, question, answer, correct, response)
        dash.set_rewards(turn, agent, base, bonus, final)
        dash.refresh()

dash.set_phase("Update", step=step)
dash.log_loss(pg, vf, kl)
dash.log_accuracy([acc_r0, acc_r1, acc_r2])
dash.refresh()
```

### 3-3. 인프라 / 환경 설정

| 항목 | 내용 | 상태 |
|---|---|---|
| 전용 conda 환경 | `/mnt/ssd/lsh/envs/maporl` (Python 3.10) | ✅ 완료 |
| Jupyter 커널 등록 | "Python (maporl)" | ✅ 완료 |
| 토큰 관리 | `~/.bashrc`에 `HF_TOKEN`, `WANDB_API_KEY` 저장 | ✅ 완료 |
| 캐시 경로 분리 | pip/HuggingFace/torch 캐시 → `/mnt/ssd` | ✅ 완료 |
| requirements.txt 수정 | 깨진 3개 패키지 제거 (`zmq==0.0.0` 등) | ✅ 완료 |

### 3-4. 향후 구현 예정 (README 및 논문 기반)

#### (a) VERL 통합

> README: *"We are also planning to use Verl to make this easy to use, especially for the inference time scaling"*

- 현재 구현은 TRL 기반 PPO
- VERL(Volcano Engine RL)로 백엔드를 교체하면 inference-time scaling 실험이 더 용이
- **작업 범위:** `trl/` 의존성을 VERL API로 대체, 동일한 멀티에이전트 로직 유지

#### (b) Answer Label 직접 사용 (Verifier-free 보상)

> README: *"I am currently trying to use answer label directly for the stable inference time scaling"*

- 현재: 별도 reward verifier 모델 또는 코드 실행으로 정답 확인
- 목표: 정답 레이블을 직접 보상으로 사용 → verifier 학습 단계 제거
- GSM8K처럼 정수 정답이 있는 태스크에서 특히 유효
- **작업 범위:** `no_reward_model=True` 경로 안정화, inference time에서 동일하게 적용

#### (c) Summarizer / Leader 모듈

> README: *"I think we should incorporate a summarizer for this case — some papers even propose a leader module (Estornell et al, 2025)"*

- 에이전트 토론 결과를 집약하는 **별도 리더 에이전트** 추가
- 여러 에이전트 응답을 요약 → 최종 답변 생성
- 참고: [Estornell et al., 2025](https://arxiv.org/abs/2507.08960)
- **작업 범위:** `construct_message_multi_agent()` 확장, 리더 어댑터 추가 또는 별도 모델

#### (d) FANTOM 데이터셋 지원

- 현재 노트북: GSM8K, ANLI 두 가지
- 코드베이스에는 FANTOM 분기가 이미 존재 (`train_ppo_v2_multi_agent_multi_model.py:134`)
- **작업 범위:** `DataCollator`의 `open_ended=True` 경로 노트북에 추가, 평가 지표 확장

#### (e) 에이전트별 다른 모델 사용

- 현재: 동일 모델 + 다른 LoRA 어댑터
- 목표: 에이전트마다 다른 기반 모델 (e.g., Phi-3 + LLaMA-3)
- `ModelManager`의 `different_models` 분기가 이미 구현됨
- **작업 범위:** 노트북 Section 5에 multi-model 초기화 흐름 추가

---

## 4. 환경 정보

```
OS      : Ubuntu (Linux 5.4.0)
GPU     : NVIDIA (CUDA 12.1)
디스크  : /dev/sda2 (468GB, 루트)  /dev/sdb (3.5TB, /mnt/ssd)
Python  : 3.10 (maporl conda env)
conda 환경 경로 : /mnt/ssd/lsh/envs/maporl
캐시 경로 : /mnt/ssd/lsh/.cache/huggingface
            /mnt/ssd/lsh/.cache/torch
            /mnt/ssd/lsh/.pip_cache
```

### 핵심 패키지 버전 (maporl 환경)

| 패키지 | 버전 |
|---|---|
| torch | 2.3.0+cu121 |
| transformers | 4.42.3 |
| peft | 0.11.1 |
| accelerate | 0.31.0 |
| deepspeed | 0.14.2 |
| bitsandbytes | 0.43.1 |
| datasets | 2.19.2 |
| trl | 0.8.7.dev0 (로컬) |

---

## 5. 빠른 시작

```bash
# 1. 환경 활성화
conda activate /mnt/ssd/lsh/envs/maporl
source ~/.bashrc

# 2. Jupyter 실행 (커널: "Python (maporl)" 선택)
cd /mnt/ssd/lsh/MAPoRL
jupyter lab

# 3. 학습 실행
accelerate launch \
    --config_file config/deepspeed_config/ds_config_zero2.yaml \
    --num_processes 6 \
    train_ppo_v2_multi_agent_multi_model.py \
    --config config/ppo_config/config_ppo2_multi_agent2_turn3_gsm8k_reloadF.py \
    --alpha "[0.5, 0.5, 0.3, 0.3]"
```
