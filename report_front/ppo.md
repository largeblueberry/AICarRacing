## 2. 강화학습 알고리즘 — PPO 이론

본 절은 우리가 사용한 **PPO(Proximal Policy Optimization)** 의 이론적 배경을 정리하고, 각 이론 요소가 `src/ppo_agent_2.py`의 어느 구현에 대응하는지를 연결한다. 우리 과제의 행동공간은 연속(steering/throttle 류의 실수값)이므로, 정책은 **가우시안 연속행동정책(Gaussian continuous-action policy)** 으로 구현하였다.

### 2.1 Actor-Critic 구조

PPO는 정책(policy)과 가치함수(value function)를 동시에 학습하는 **Actor-Critic** 계열 알고리즘이다.

- **Actor(정책망)** 는 상태 `s`를 입력받아 행동분포 `π(a|s)`를 출력한다. 우리 구현에서는 CNN으로 추출한 특징 벡터를 받아 **가우시안 분포의 평균 `μ`와 로그표준편차 `log σ`** 를 출력한다(`Actor.forward`, `src/ppo_agent_2.py:67`). 평균은 `tanh`로 `[-1, 1]`에 제약하고(`:83`), 표준편차는 고정이 아니라 **학습 대상(learned std)** 으로 두었다(`fixed_std=False`, `:85`). 행동분포는 `torch.distributions.Normal(mean, std)`로 구성한다(`Actor.get_action_dist`, `:96`).
- **Critic(가치망)** 은 상태가치 `V(s)`를 스칼라로 추정하며, Advantage 계산과 가치손실의 타깃 역할을 한다(`Critic.forward`, `src/ppo_agent_2.py:156`).
- Actor와 Critic은 CNN 특징추출기를 공유하며, 세 모듈의 파라미터를 **하나의 Adam optimizer로 통합 최적화** 한다(`src/ppo_agent_2.py:232`).

행동 샘플링 시에는 분포에서 `sample()`로 행동을 뽑고 각 차원의 로그확률을 합산하여 `log π(a|s)`를 얻는다(`PPOAgent.act`, `src/ppo_agent_2.py:293`). 연속행동이므로 다차원 가우시안의 차원별 로그확률·엔트로피를 합산한다(`evaluate_actions`, `:126`).

### 2.2 정책 경사에서 PPO로 — 신뢰영역의 동기

순수 정책 경사(policy gradient, 예: REINFORCE/A2C)는

```
∇J(θ) = E[ ∇ log π_θ(a|s) · A(s,a) ]
```

형태의 추정량을 따라 정책을 갱신한다. 그러나 수집한 데이터(rollout)를 여러 epoch 재사용하면 정책이 데이터를 만든 정책 `π_old`에서 너무 멀리 이동해 **분포 이동(distribution shift)** 이 커지고, 한 번의 큰 갱신이 정책을 붕괴(collapse)시킬 수 있다.

이를 막기 위해 **TRPO** 는 갱신 전후 정책의 KL 발산을 신뢰영역(trust region)으로 제한한다. **PPO** 는 그 아이디어를 **2차 제약 대신 1차 클리핑(clipping)** 으로 근사하여, 확률비 `r_t`가 신뢰영역 `[1-ε, 1+ε]`를 벗어나면 목적함수의 기여를 잘라내는 방식으로 사실상 신뢰영역을 강제한다. 구현이 단순하면서도 안정적이라 본 과제에서 채택하였다.

### 2.3 Clipped Surrogate Objective

핵심 정책 목적함수는 다음과 같다.

```
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)

L^CLIP(θ) = E_t[ min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t ) ]
```

여기서 `A_t`는 Advantage 추정값, `ε`는 클리핑 폭(`clip_epsilon`)이다. 우리는 **ε = 0.15** 를 사용하였다(기본 0.2 대비 축소하여 신뢰영역을 좁혀 KL 폭발/붕괴를 억제, `config["clip_epsilon"] = 0.15`, `scripts/train_ppo_2action2.py:34`).

`min`과 클리핑의 의미: Advantage가 양수일 때는 `r_t`가 `1+ε`를 넘어도 이득이 더 커지지 않게 상한을 두고, 음수일 때는 `r_t`가 `1-ε` 아래로 내려가도 더 내려가지 않게 하한을 둔다. 즉 신뢰영역 밖으로의 과도한 갱신에 대한 인센티브를 제거한다.

구현은 수치적으로 `r_t = exp(log π - log π_old)`로 계산하며, 부호가 반대인 정책손실(최소화 대상)로 변환한다(`src/ppo_agent_2.py:501`, `:505`–`:507`):

```python
ratio = torch.exp(log_probs - old_log_probs_batch)
policy_loss_1 = advantages_batch * ratio
policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()   # = -L^CLIP
```

### 2.4 Advantage 추정 — GAE

Advantage는 **GAE(Generalized Advantage Estimation)** 로 추정하여 편향-분산을 절충한다. TD 오차

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done) - V(s_t)
```

를 정의하고, GAE는 이를 재귀적으로 누적한다.

```
A_t = δ_t + (γ·λ) · (1 - done) · A_{t+1}
R_t = A_t + V(s_t)          (가치함수 타깃, 즉 return)
```

우리는 **γ = 0.99, λ = 0.95** 를 사용한다(`config["gamma"]=0.99`, `config["gae_lambda"]=0.95`, `scripts/train_ppo_2action2.py:32`–`:33`). 구현은 rollout을 뒤에서 앞으로 순회하며 위 재귀식을 그대로 적용한다(`RolloutBuffer.compute_returns_and_advantages`, `src/rollout_buffer.py:142`, `:146`, `:151`). 또한 미니배치 추출 직전에 Advantage를 버퍼 전체에 대해 **전역 정규화(평균 0, 분산 1)** 하여 스케일을 안정화한다(`src/rollout_buffer.py:189`–`:191`).

### 2.5 손실 함수 — 정책 + 가치 + 엔트로피

총손실은 세 항의 가중합이다.

```
L_total = L_policy + vf_coef · L_value - ent_coef · H[π]
```

- **가치손실 `L_value`**: Critic 추정값과 GAE return의 **MSE(평균제곱오차)**.
  `value_loss = F.mse_loss(values, returns_batch)` (`src/ppo_agent_2.py:510`).
- **엔트로피 보너스 `H[π]`**: 정책 엔트로피를 키우는 방향으로 보상하여 조기 수렴/탐색 부족을 완화한다. 가우시안 정책의 차원별 엔트로피를 합산해 사용한다.
- **계수**: `vf_coef = 0.5`(표준값), `ent_coef = 0.01`(탐색 강박 완화) (`scripts/train_ppo_2action2.py:35`–`:36`).

총손실 조립은 `src/ppo_agent_2.py:516`에 있다.

```python
entropy_loss = -torch.mean(entropy)
loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
```

> 부호 주의: FP32 경로(`learn`)는 `entropy_loss = -mean(entropy)`로 정의한 뒤 `+ ent_coef·entropy_loss`를 더하고(`:513`, `:516`), 혼합정밀 경로(`learn_mixed_precision`)는 `entropy_loss = mean(entropy)`로 두고 `- ent_coef·entropy_loss`를 뺀다(`:395`, `:398`). 둘 다 **엔트로피를 최대화(보너스)** 하는 동일한 효과로, 요청한 `L = policy + vf_coef·value - ent_coef·entropy` 식과 일치한다.

학습 안정화를 위해 갱신마다 전체 파라미터에 대한 **gradient clipping** 을 적용한다(`max_grad_norm = 0.5`, `src/ppo_agent_2.py:524`).

### 2.6 KL 신뢰영역과 early-stop — per-minibatch 강제

클리핑만으로는 큰 버퍼에서 여러 epoch를 돌 때 정책이 신뢰영역을 누적적으로 벗어날 수 있다. PPO 표준 구현은 보통 **approx_kl** 을 모니터링해 `target_kl`을 넘으면 학습을 조기 종료한다. 우리는 안전성을 위해 이 검사를 **epoch 단위가 아니라 per-minibatch 단위** 로 수행한다(`src/ppo_agent_2.py:554`).

```python
if self.target_kl is not None and approx_kl > self.target_kl * 1.5:
    print(f"Early stop: ... minibatch KL {approx_kl:.4f} > {self.target_kl*1.5:.4f}")
    continue_training = False
    break
```

- **임계값**: `approx_kl > target_kl × 1.5`, 즉 **0.03 × 1.5 = 0.045** 를 초과하면 즉시 중단한다(`target_kl = 0.03`, `scripts/train_ppo_2action2.py:38`).
- **왜 per-minibatch가 더 안전한가**: 우리 버퍼는 32,768 샘플, 미니배치 2,048 → epoch당 약 **16번의 갱신** 이 일어난다. epoch 단위로만 KL을 검사하면 한 epoch 동안 16번 갱신이 모두 적용된 *뒤에야* KL이 측정되어, 그 사이 정책이 신뢰영역을 한참 지나쳐 KL 스파이크/collapse가 발생할 수 있다. per-minibatch 검사는 **매 갱신 직후** KL을 보고 임계 초과 시 그 즉시 멈추므로(현 minibatch와 epoch 루프를 모두 break, `:556`–`:560`), 신뢰영역 위반을 갱신 1회 수준으로 제한한다. 이는 본 파일 모듈 docstring에 명시된 2-action 라인 전용 안정화 수정이다(`src/ppo_agent_2.py:4`–`:6`). 동일한 per-minibatch 조기 종료가 혼합정밀 경로에도 적용된다(`learn_mixed_precision`, `:446`–`:452`).

approx_kl은 분산이 낮은 추정량 `E[(r-1) - log r]`(혼합정밀, `:430`) 또는 `0.5·E[(log π - log π_old)²]`(FP32, `:538`)로 계산한다. 함께 clip fraction(`|r-1| > ε`인 샘플 비율)도 로깅하여 클리핑 작동 정도를 모니터링한다(`:540`).

### 2.7 학습 파이프라인 하이퍼파라미터 요약

| 항목 | 값 | 출처 |
| --- | --- | --- |
| 병렬 환경 수 | 64 async envs | `scripts/train_ppo_2action2.py:22` |
| RolloutBuffer 크기 | 32,768 (= 64 envs × 512 steps) | `:29` |
| 미니배치 크기 | 2,048 | `:30` |
| PPO epochs | 6 | `:31` |
| 할인율 γ | 0.99 | `:32` |
| GAE λ | 0.95 | `:33` |
| 클리핑 ε (`clip_epsilon`) | 0.15 | `:34` |
| 가치 계수 `vf_coef` | 0.5 | `:35` |
| 엔트로피 계수 `ent_coef` | 0.01 | `:36` |
| `max_grad_norm` | 0.5 | `:37` |
| `target_kl` (per-minibatch, ×1.5 강제) | 0.03 | `:38` |
| 초기 행동 std | 0.5 (학습 대상) | `:42`, `:44` |
| 학습률 스케줄 | cosine 1e-4 → 1e-5 (warmup 후 코사인 감쇠) | `:28`, `:46` |
| 혼합정밀(mixed precision) | True (CUDA) | `:58` |
| 정책 종류 | 가우시안 연속행동정책 (learned std) | `src/ppo_agent_2.py:32`, `:96` |

학습 루프는 매 rollout마다 (1) 64개 비동기 환경에서 32,768 step을 수집하고, (2) GAE로 return/advantage를 계산한 뒤(`compute_returns_and_advantages`), (3) 2,048 미니배치로 최대 6 epoch 동안 PPO 갱신을 수행하되 per-minibatch KL early-stop으로 안전하게 종료하고, (4) **cosine learning-rate 스케줄(초기 1e-4 → 하한 1e-5)** 을 갱신한다(`update_learning_rate`, `src/ppo_agent_2.py:303`; 학습 루프 `scripts/train_ppo_2action2.py:278`–`:374`). CUDA에서는 `learn_mixed_precision`이 `autocast`/`GradScaler` 기반 혼합정밀로 동작하여 메모리/속도를 개선한다(`src/ppo_agent_2.py:343`, `:375`, `:405`).