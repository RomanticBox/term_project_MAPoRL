# MAPoRL Tutorial Notebook — Chapter Guide

> **Notebook**: `MAPoRL_tutorial.ipynb`  
> **Paper**: ACL 2025 · Multi-Agent Post-Co-Training for Collaborative LLMs with RL  
> **GPU requirements**: 5× GPU (excludes GPU 2 on 6× Titan RTX server)  
> Run the notebook from this directory (`term_project_MAPoRL/`).

---

## Table of Contents

| # | Chapter | Key Files |
|---|---------|-----------|
| 0 | [Source Code Setup](#0-source-code-setup) | `%%writefile` cells |
| 1 | [Paper Overview](#1-paper-overview) | — |
| 2 | [Environment Setup](#2-environment-setup) | `requirements.txt` |
| 3 | [Dataset Preparation](#3-dataset-preparation) | `utils/dataset.py` |
| 4 | [DataCollator](#4-datacollator) | `trl/.../ppov2_trainer_multi_different_model.py` |
| 5 | [Model Architecture](#5-model-architecture) | `utils/utils_model.py` |
| 6 | [Reward System](#6-reward-system) | `utils/utils_general.py` |
| 7 | [Multi-Agent Communication](#7-multi-agent-communication) | `utils/utils_general.py` |
| 8 | [PPO Rollout Phase](#8-ppo-rollout-phase) | `trl/.../ppov2_trainer_multi_different_model.py` |
| 9 | [PPO Update Phase](#9-ppo-update-phase) | `trl/.../ppov2_trainer_multi_different_model.py` |
| 10 | [Training Configuration & Execution](#10-training-configuration--execution) | `train_ppo_v2_multi_agent_multi_model.py` |
| 10.5 | [Live Training Dashboard](#105-live-training-dashboard) | inline class |
| 11 | [Inference Pipeline](#11-inference-pipeline) | checkpoint loading |
| 12 | [Evaluation](#12-evaluation) | `utils/utils_cooperateLLM.py` |
| 13 | [Experiment 1 — No Role Assignment](#13-experiment-1--no-role-assignment) | `config/ppo_config/` |
| 14 | [Experiment 2 — Asymmetric Parameter Size](#14-experiment-2--asymmetric-parameter-size) | `config/ppo_config/` |
| 15 | [Experiment 3 — Base Model Size Comparison](#15-experiment-3--base-model-size-comparison) | `config/ppo_config/` |

---

## Chapter Details

<details>
<summary><strong>0. Source Code Setup</strong> — %%writefile cells that write all project source files to disk</summary>

Run all cells in this section **once** to populate source files. Skip if already cloned from the repo.

| Cells | What is written |
|-------|----------------|
| Utility modules | `utils/cleansing.py`, `utils/dataset.py`, `utils/utils_general.py`, `utils/utils_model.py`, `utils/FANTOM_initial_setting.py` |
| Custom TRL trainer | `trl/trl/trainer/ppov2_trainer_multi_different_model.py`, `trl/trl/trainer/utils.py` |
| Training scripts | `train_ppo_v2_multi_agent_multi_model.py`, `train_reward.py`, `train_sft.py` |
| Install | `pip install -e trl/` to make the custom TRL importable |

After this section, all runtime dependencies are in place.

</details>

---

<details>
<summary><strong>1. Paper Overview</strong> — MAPoRL architecture and key concepts</summary>

Explains the overall MAPoRL framework through diagrams and pseudocode.

- **Architecture diagram**: Base LLM + multiple LoRA adapters shared across agents
- **Training loop overview**: Rollout → Reward → GAE → PPO update
- **Key insight**: Agents share one base model but hold separate adapters (`policy`, `col`, `value_*`, `ref`)
- **Role-split design**: Turn-0 uses `policy` adapter (domain knowledge); Turn-1+ uses `col` adapter (collaboration strategy)
- **Variants explained**: `collaboration_separation` flag controls whether roles are split

</details>

---

<details>
<summary><strong>2. Environment Setup</strong> — Package installation and GPU verification</summary>

- Installs all Python packages from `requirements.txt`
- Verifies CUDA availability and GPU memory
- Sets global logging suppression for clean notebook output
- **Important**: `CUDA_VISIBLE_DEVICES=0,1,3,4,5` (GPU 2 excluded — hardware instability)

</details>

---

<details>
<summary><strong>3. Dataset Preparation</strong> — TinyGSM and ANLI R3 loading & tokenization</summary>

MAPoRL supports two benchmarks:

| Dataset | Task | Answer format | Verifier |
|---------|------|--------------|---------|
| **TinyGSM** | Math word problems | Integer in `\boxed{}` | Python eval |
| **ANLI R3** | Natural language inference | entailment / neutral / contradiction | Label match |

Cells cover:
- `load_dataset()` for both datasets
- Chat-template formatting (`apply_chat_template`)
- Tokenization with left-padding and EOS handling
- Batch size / sequence length stats

</details>

---

<details>
<summary><strong>4. DataCollator</strong> — CustomDataCollator for multi-agent batches</summary>

Source: `trl/trl/trainer/ppov2_trainer_multi_different_model.py` lines 99–129

- Pads `input_ids` and `answer` tensors to the longest sequence in each batch
- FANTOM tasks additionally handle `wrong_answer` tensors
- Demonstrates collation with a toy 2-sample batch

</details>

---

<details>
<summary><strong>5. Model Architecture</strong> — Multi-Adapter LoRA with ModelManager</summary>

Source: `utils/utils_model.py` — `ModelManager` class

```
Base LLM (4-bit BnB, frozen)
  ├── adapter "policy"   ← Turn-0 (domain knowledge)
  ├── adapter "col"      ← Turn-1+ (collaboration)
  ├── adapter "value_0"  ← Turn-0 value head feature extractor
  ├── adapter "value_1"  ← Turn-1+ value head feature extractor
  └── adapter "ref"      ← Reference policy (weights=0, frozen, KL baseline)
Value Heads: nn.Linear(hidden_size → 1), one per round
```

Cells cover:
- `ModelManager` initialization for `diff_model_training=False` (shared base)
- `PolicyAndValueWrapper` for adapter-switching during rollout
- Ablation setup: `collaboration_separation=False` (single `policy` adapter for all turns)
- Asymmetric LoRA: passing `peft_config` as a list for per-agent rank differences

</details>

---

<details>
<summary><strong>6. Reward System</strong> — bonus_rule (α shaping) + score_rule (γ discounting)</summary>

Source: `utils/utils_general.py`

```
Per (turn, agent) response:
  base_reward  = verifier score ∈ {0, 1}
  bonus_rule(α) = debate quality shaping
                  (persuasion bonus, revision bonus, consistency penalty)
  score_rule(γ) = discounted_sum over rounds → final scalar reward
```

Cells cover:
- `extract_ans_from_response` — parse integer from `\boxed{}`
- `bonus_rule` logic with `α = [α_p, α_r, α_c, α_f]` parameters
- `score_rule` with `γ = 0.3` discount over 3 rounds
- Visualization of discounting effect across turn configurations

</details>

---

<details>
<summary><strong>7. Multi-Agent Communication Protocol</strong> — Message construction across rounds</summary>

Source: `utils/utils_general.py` — `construct_message_multi_agent`

```
Round 0: [user] {question} → [assistant] first answer
Round k: [user] {question}
          + other agents' Round-(k-1) responses
          + "What is your answer?"
          → [assistant] revised answer
```

Cells cover:
- `construct_message_multi_agent()` implementation walkthrough
- `show_dialog()` helper to pretty-print full conversation traces
- Example trace: 2 agents × 3 rounds on a GSM8K question

</details>

---

<details>
<summary><strong>8. PPO Rollout Phase</strong> — Trajectory generation (no gradients)</summary>

Source: `trl/trl/trainer/ppov2_trainer_multi_different_model.py` — `generate_rollouts()`

```
for turn in [0, 1, 2, ...]:
    switch adapter: turn=0 → "policy",  turn≥1 → "col"
    for agent in [0, 1]:
        construct_query (apply chat_template)
        model.generate() → response tokens
        compute ref_logprob  (KL penalty baseline)
        get_reward()         (value head estimate)
        postprocess          (truncate at max_output_length, add \boxed{})
        extract_answer       → base_reward ∈ {0, 1}
```

Cells cover:
- Rollout pseudocode with annotated adapter-switching
- Response postprocessing: forced `\boxed{}` insertion when missing
- Sequence length management (`max_input_length=200`, `max_output_length=80`)

</details>

---

<details>
<summary><strong>9. PPO Update Phase</strong> — GAE advantage estimation + clipped PPO loss</summary>

Source: `trl/trl/trainer/ppov2_trainer_multi_different_model.py` lines 1600–1870

```
1. score_rule + bonus_rule  → shaped reward per turn
2. KL penalty               : reward -= kl_coef × (log π − log π_ref)
3. GAE                      : Â_t = δ_t + γλ · Â_{t+1}
4. PPO clip loss             : L_pg = −min(r·Â, clip(r, 1−ε, 1+ε)·Â)
5. Value clip loss           : L_vf = max((V−R)², (clip(V, V_old±ε)−R)²)
6. Total loss                : L = L_pg + vf_coef × L_vf
7. accelerator.backward(loss)
8. AdamW step on LoRA params only
```

Cells cover:
- GAE implementation walkthrough with toy numbers
- PPO clip loss derivation and clipping visualization
- Full update-phase control flow diagram

</details>

---

<details>
<summary><strong>10. Training Configuration & Execution</strong> — Hyperparameters and multi-GPU launch</summary>

Source: `train_ppo_v2_multi_agent_multi_model.py`

Key hyperparameters used:

| Parameter | Value | Note |
|-----------|-------|------|
| `max_output_length` | 80 | Reduces peak memory ([B,H,L,L] attention) |
| `num_ppo_epochs` | 1 | Per-batch gradient steps |
| `num_mini_batches` | 2 | Mini-batch splits |
| `total_episodes` | 40 (smoke) / 200 (full) | 40 = 1 update for sanity check |
| `per_device_train_batch_size` | 1 | |
| `gradient_accumulation_steps` | 4 | |
| `kl_coef` | 0.002 | KL penalty weight |

Cells cover:
- Config file loading via `load_config_from_python()`
- DeepSpeed ZeRO-2 config (`config/deepspeed_config/ds_config_zero2.yaml`)
- Checkpoint save/reload structure
- Progress monitoring via `{output_dir}/progress.log`

</details>

---

<details>
<summary><strong>10.5. Live Training Dashboard</strong> — In-notebook training monitor</summary>

`MAPoRLDashboard` class renders a fixed panel inside the notebook showing:
- Current phase (Rollout / Update), update counter, ETA
- Per-agent response samples with scores
- Running accuracy and loss curves

Usage:
```python
dash = MAPoRLDashboard(agent_num=2, round_num=3)
dash.set_phase("Rollout", update=3)
dash.log_agent_answer(agent=0, turn=1, response="...", score=0.8)
dash.refresh()
```

A **simulation cell** demonstrates the dashboard without GPU.

</details>

---

<details>
<summary><strong>11. Inference Pipeline</strong> — Load checkpoint and run multi-agent debate</summary>

- Load base model + reload all adapters from a saved checkpoint directory
- Run `generate_rollouts()` in eval mode (no gradient, no value update)
- Collect per-agent, per-round responses
- Pretty-print debate traces with `show_dialog()`

Adapter directory structure expected:
```
{output_dir}/
  policy/          ← Turn-0 adapter weights
  col/             ← Turn-1+ adapter weights
  value_0/         ← Value adapter (round 0)
  value_1/         ← Value adapter (round 1)
  value_head_value_heads_0/model.safetensors
  value_head_value_heads_1/model.safetensors
```

</details>

---

<details>
<summary><strong>12. Evaluation</strong> — stat_all_simple: round-by-round accuracy + transition analysis</summary>

Source: `utils/utils_cooperateLLM.py` lines 287–422

`stat_all_simple(outputs)` computes:
- **Accuracy per round**: majority-vote over agent answers
- **Transition matrix**: wrong→right (↑) and right→wrong (↓) counts across rounds
- **Convergence rate**: fraction of debates that reach consensus

Cells cover:
- Simulation verification (2-agent, 3-round, 10 questions)
- Evaluation helper that wraps inference + stat_all_simple
- W&B log loader for post-training curve visualization

</details>

---

<details>
<summary><strong>13. Experiment 1 — No Role Assignment</strong> (collab_sep True vs False)</summary>

**Research question**: Does separating Task-turn and Collaboration-turn adapters improve performance?

| Config | `collaboration_separation` | Adapters used |
|--------|--------------------------|--------------|
| `colsep_true` | `True` | `policy` (turn 0) + `col` (turn 1+) |
| `colsep_false` | `False` | `policy` for all turns |

- Config files auto-generated into `config/ppo_config/`
- Training launched sequentially via `run_experiment()`
- `CUDA_VISIBLE_DEVICES=0,1,3,4,5`, `--num_processes 5`
- Results table: round-by-round accuracy comparison

</details>

---

<details>
<summary><strong>14. Experiment 2 — Asymmetric Parameter Size</strong></summary>

**Research question**: How does per-agent LoRA rank asymmetry affect collaborative debate?

| Config | Agent 0 rank | Agent 1 rank |
|--------|-------------|-------------|
| `sym_r4` | 4 | 4 |
| `sym_r16` | 16 | 16 |
| `asym_4_16` | 4 | 16 |
| `asym_4_32` | 4 | 32 |

- `peft_config` is passed as a list to `ModelManager` for per-agent ranks
- 4 configs trained sequentially
- Results: accuracy + parameter count comparison

</details>

---

<details>
<summary><strong>15. Experiment 3 — Base Model Size Comparison</strong></summary>

**Research question**: Does a larger base LLM lead to more effective multi-agent collaboration?

| Config | Base Model | Parameters |
|--------|-----------|-----------|
| `size_0.5B` | Qwen2.5-0.5B-Instruct | 0.5 B |
| `size_3B` | Phi-3-mini-128k-instruct | 3.4 B |
| `size_8B` | Llama-3-8B-Instruct | 8 B |

All other settings identical (collab_sep=True, r=8, GSM8K, 2 agents, 3 rounds).  
Evaluation compares accuracy gain from Round-0 → Round-2.

</details>

---

## Running the Experiments

```bash
# 1. Start Jupyter from this directory
cd /path/to/term_project_MAPoRL
jupyter notebook MAPoRL_tutorial.ipynb

# 2. Run Section 0 (%%writefile cells) once to populate source files
# 3. Run Sections 1–12 to understand the implementation
# 4. Run Section 13 Cell 74 to generate config files
# 5. Run Section 13 Cell 75 to start training (smoke test: total_episodes=40)

# Monitor training progress:
tail -f exp1_colsep_true/progress.log
```

> **Smoke test**: `total_episodes=40` → 1 PPO update → ~10 min on 5× Titan RTX  
> **Full run**: Change to `total_episodes=200` in Cell 74 `BASE_CFG`
