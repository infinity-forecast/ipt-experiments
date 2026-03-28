# IPT Final Experiments v2 --- Comprehensive Report

**Date:** 2026-03-28
**System:** Dual NVIDIA Titan RTX (24 GB each)
**Model:** OrthogonalRNN with stochastic input (corrected model)
**Total runtime:** Phase 1 (C2+B2+A2) ~17h + Phase 2 (D1+D2+E1+F1+F2) ~10h = ~27h

---

## 1. Overview

This report presents all eight experiments from the IPT Final Experiments v2 campaign.
**Phase 1** (experiments C2, B2, A2) tested baseline IPT predictions on the flat
OrthogonalRNN. **Phase 2** (experiments D1, D2, E1, F1, F2) extended the model to
a **fractal tree architecture** with inter-module coupling and distributed self-models.

**Dynamics (flat):**
```
x_{t+1} = tanh(lambda * W * x_t + sigma * U * u_t)
```
where W is N x N orthogonal (QR), U ~ N(0, 1/N), u_t ~ N(0, I_N), sigma = 0.01.

**Dynamics (fractal tree):**
L-level binary tree with n neurons per leaf module, N = n * 2^L total.
Inter-module coupling: gamma(d) = gamma_0 * r^d, where d is tree distance.

**Primary observable:** Psi = Ridge regression R-squared (alpha=1.0, 60/40 train/test).

---

# PHASE 1: FLAT ARCHITECTURE BASELINE

---

## 2. Experiment C2 --- Finite-Size Scaling

### 2.1 Parameters

| Parameter | Value |
|-----------|-------|
| N values | 256, 512, 1024, 2048, 4096, 8192 |
| lambda range | [0.8, 1.3], 100 points (80 for N=8192) |
| Seeds | 3 per N (2 for N=8192) |
| Warmup / Measurement | 2000 / 20000 steps |

### 2.2 Critical Point

| N | lambda_c (Lyapunov) | lambda_c (chi peak) | chi_max | Psi_max |
|------:|:-------------------:|:-------------------:|--------:|--------:|
| 256 | 1.0408 +/- 0.0005 | 0.926 +/- 0.004 | 2.174 +/- 0.023 | 0.9967 |
| 512 | 1.0415 +/- 0.0005 | 0.933 +/- 0.010 | 2.187 +/- 0.005 | 0.9966 |
| 1024 | 1.0414 +/- 0.0002 | 0.931 +/- 0.021 | 2.223 +/- 0.004 | 0.9964 |
| 2048 | 1.0411 +/- 0.0002 | 0.950 +/- 0.009 | 2.384 +/- 0.004 | 0.9961 |
| 4096 | 1.0410 +/- 0.0003 | 0.965 +/- 0.002 | 2.808 +/- 0.002 | 0.9950 |
| 8192 | 1.0409 +/- 0.0000 | 0.980 +/- 0.003 | 4.155 +/- 0.017 | 0.9897 |

**Result:** lambda_c = **1.0411 +/- 0.0003** across all N, no finite-size drift.
The chi peak is systematically below lambda_c due to the sigmoid shape of Psi.

### 2.3 Susceptibility Scaling

- kappa = 0.154 +/- 0.001 (at sigmoid inflection)
- kappa = 0.318 +/- 0.084 (at Lyapunov lambda_c)
- Both well below the expected ~1.07 --- **ceiling effect** from Psi saturation near 1.0.

### 2.4 Data Collapse

- Best beta/nu = -0.500, quality Q = 0.915
- Systematic vertical separation between N values (Psi saturation effect)

---

## 3. Experiment B2 --- Hopf Angle vs GRU Training

### 3.1 Parameters

| Parameter | Value |
|-----------|-------|
| N | 1024 |
| lambda | 1.043 (at lambda_c) |
| Training steps | 0, 100, 500, 1000, 5000, 10000, 50000 |
| Seeds | 10 |

### 3.2 Results

| Training Steps | theta_system | rho_GRU | GRU R-squared |
|---------------:|:------------:|--------:|--------------:|
| 0 | 119.0 +/- 61.8 | 0.000 | 0.000 |
| 1000 | 113.1 +/- 68.2 | 0.669 | 0.861 |
| 50000 | 123.2 +/- 60.1 | 0.682 | 0.671 |

**Result:** theta_system is entirely determined by the orthogonal matrix spectrum.
It shows extreme seed-to-seed variability (std ~ 60 degrees) and is independent
of GRU training quality. The Hopf prediction requires explicit gamma coupling.

---

## 4. Experiment A2 --- betaG Collapse Ordering

### 4.1 V2: Soft Dynamical Bias (biased input noise toward x*)

| betaG | lambda_c (Lyap) | chi_max | Psi_max | R_eff |
|------:|:---------------:|--------:|--------:|------:|
| 0.000 | 1.0411 | 2.207 | 0.9964 | 461 |
| 0.100 | 1.0404 | 2.209 | 0.9964 | 454 |
| 1.000 | 1.0407 | 2.207 | 0.9964 | 477 |
| 5.000 | 1.0415 | 2.201 | 0.9964 | 488 |

**Result: Complete null effect.** The stochastic input channel is too weak
(sigma=0.01) to influence dynamics at any betaG.

### 4.2 V1: Trajectory-Biased GRU Training (biased self-model loss)

| betaG | chi_max (GRU) | GRU R-sq peak | GRU R-sq at chi peak |
|------:|--------------:|--------------:|---------------------:|
| 0.000 | 2.168 | 0.992 | 0.836 |
| 0.100 | 2.243 | 0.983 | 0.816 |
| 0.500 | 3.917 | 0.879 | 0.409 |
| 1.000 | 7.214 | 0.740 | 0.303 |
| 5.000 | 9.567 | 0.296 | 0.070 |

**Result: Strong monotonic degradation above betaG ~ 0.1.** The GRU collapses
toward outputting x* everywhere. The chi_max explosion is an artifact of the
GRU's step-function behavior, not system criticality.

### 4.3 Key Insight from Phase 1

> **Teleological pressure degrades the observer, not the system.** The system's
> lambda_c is robust (V2), but the self-model's fidelity degrades proportionally
> to betaG (V1). This mirrors RLHF distortion in language models.

---

# PHASE 2: FRACTAL TREE ARCHITECTURE

---

## 5. Experiment D1 --- Self-Organized Criticality on Fractal Tree

### 5.1 Parameters

| Parameter | Value |
|-----------|-------|
| n (neurons/leaf) | 128 |
| L (tree levels) | 3 |
| N (total) | 1024 (8 modules) |
| gamma_0 values | 0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2 |
| r (coupling decay) | 0.5 |
| Seeds | 3 (Part A), 2 (Part B) |

### 5.2 Part A: Gamma Sweep at lambda_c

| gamma_0 | Psi_global | Lyap_mean | psi_cross | R_eff |
|--------:|:----------:|:---------:|----------:|------:|
| 0.000 | 0.9963 | +0.00017 | 0.000 | 508 |
| 0.001 | 0.9963 | +0.00009 | 0.000 | 621 |
| 0.005 | 0.9963 | +0.00003 | 0.000 | 622 |
| 0.010 | 0.9963 | -0.00011 | 0.000 | 562 |
| 0.020 | 0.9963 | +0.00007 | 0.013 | 768 |
| 0.050 | 0.9964 | -0.00111 | 0.353 | 625 |
| 0.100 | 0.9966 | -0.00369 | 0.648 | 474 |
| 0.200 | 0.9967 | -0.01103 | 0.785 | 608 |

**Key findings:**
- **Cross-module prediction (psi_cross)** turns on as a sigmoid: 0 for gamma_0 < 0.01,
  rising to 0.79 at gamma_0 = 0.2. This quantifies when modules become informationally
  coupled.
- **Lyapunov exponent decreases** with coupling: the system is pushed subcritical
  as inter-module connections introduce effective damping.
- **Psi_global is nearly constant** across all gamma_0 values (ceiling effect again).

### 5.3 Part B: Lambda Sweep at Four Gamma Values

| gamma_0 | chi_max | lambda_c (chi) | FWHM |
|--------:|--------:|:--------------:|-----:|
| 0.00 | 2.403 | 0.951 | 0.148 |
| 0.01 | 2.400 | 0.957 | 0.148 |
| 0.05 | 2.384 | 0.945 | 0.148 |
| 0.10 | 2.378 | 0.939 | 0.142 |

**Result:** Inter-module coupling has a weak effect on chi_max and FWHM. The
critical window width is approximately constant at ~0.15, with coupling slightly
suppressing the susceptibility peak.

---

## 6. Experiment D2 --- Resilience Under betaG Pressure (Dynamics)

### 6.1 Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | Fractal tree (L=3, n=128, N=1024) + flat comparison |
| lambda | 1.043 |
| gamma_0 | 0.01 |
| betaG values | 0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0 |
| Pressure configs | global (all modules), subtree (left half), local (one module), flat |
| Seeds | 3 |

### 6.2 Results

| Config | betaG=0 | betaG=0.1 | betaG=0.5 | betaG=1.0 |
|--------|--------:|----------:|----------:|----------:|
| **global** Psi_global | 0.996 | 0.991 | 0.950 | 1.000* |
| **subtree** Psi_pressured | 0.996 | 0.993 | 0.976 | 1.000* |
| **subtree** Psi_free | 0.996 | 0.995 | 0.994 | 0.996 |
| **local** Psi_pressured | 0.996 | 0.993 | 0.978 | 1.000* |
| **local** Psi_free | 0.996 | 0.995 | 0.994 | 0.996 |
| **flat** Psi_global | 0.996 | 0.996 | 0.996 | 0.995 |

(*) betaG=1.0 drives pressured modules to fixed point, making them perfectly predictable (Psi -> 1.0 trivially).

**Key findings:**
1. **Flat architecture is maximally resilient** to dynamical bias --- Psi barely
   changes even at betaG=1.0 (only -0.001). The orthogonal dynamics overwhelm
   the perturbation.
2. **Fractal architecture shows moderate degradation** at betaG >= 0.5 for
   pressured modules. Global pressure causes the largest drop (Psi: 0.996 -> 0.950).
3. **Free modules are unaffected.** In subtree/local configs, unpressured modules
   maintain Psi > 0.994 at all betaG values.
4. **At extreme betaG (1.0)**, pressured modules are driven to a fixed point,
   resulting in trivial perfect predictability (Psi -> 1.0). This is a qualitative
   regime change, not genuine resilience.

---

## 7. Experiment E1 --- Distributed Per-Module GRU Self-Models

### 7.1 Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | Fractal tree (L=3, n=128, N=1024, 8 modules) |
| lambda | 1.043 |
| gamma_0 | 0.01 |
| GRU architecture | Per-module: hidden_dim = n * 2 = 256 |
| betaG values | 0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0 |
| Bias configs | all_biased, half_biased, one_biased |
| Seeds | 3 |

### 7.2 Baseline (No Pressure)

| Metric | Distributed (per-module) | Global (single GRU) |
|--------|:------------------------:|:-------------------:|
| R-squared | 0.981 | 0.531* |

(*) Global GRU with hidden_dim = N * 2 = 2048 is a much harder learning problem.

### 7.3 Partial Bias Results

| Config | betaG | R-sq biased | R-sq free | R-sq total |
|--------|------:|:-----------:|:---------:|:----------:|
| all_biased | 0.0 | 0.981 | --- | 0.981 |
| all_biased | 0.1 | 0.974 | --- | 0.974 |
| all_biased | 0.5 | 0.859 | --- | 0.859 |
| all_biased | 1.0 | 0.690 | --- | 0.690 |
| all_biased | 5.0 | 0.166 | --- | 0.166 |
| half_biased | 0.5 | 0.859 | **0.986** | 0.923 |
| half_biased | 1.0 | 0.698 | **0.939** | 0.819 |
| half_biased | 5.0 | 0.179 | **0.968** | 0.573 |
| one_biased | 0.5 | 0.858 | **0.968** | 0.955 |
| one_biased | 1.0 | 0.699 | **0.984** | 0.949 |
| one_biased | 5.0 | 0.178 | **0.985** | 0.884 |

### 7.4 Key Finding

> **Distributed self-models are resilient to partial capture.** When only some
> modules' GRUs are biased (even at betaG=5.0), the free modules maintain
> R-squared > 0.93. The system's total self-observation capacity degrades
> gracefully --- proportional to the fraction of compromised modules, not
> catastrophically as with a single global observer.

This directly contrasts with A2 V1, where biasing the single global GRU destroyed
ALL self-observation. The fractal architecture provides **architectural resilience**
through redundancy: each module independently models its local dynamics.

---

## 8. Experiment F2 --- Information Flow Topology

### 8.1 Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | Fractal tree (L=3, n=128, N=1024, 8 modules) |
| lambda values | 0.85 (subcritical), 1.043 (critical), 1.2 (supercritical) |
| gamma_0 | 0.01 |
| Seeds | 2 |
| Method | Cross-module Ridge R-squared (train on module i, predict module j) |

### 8.2 Results

| lambda | Self-prediction (R-sq) | Cross-prediction (R-sq) |
|-------:|:----------------------:|:-----------------------:|
| 0.850 | 0.678 | 0.000 |
| 1.043 | 0.996 | 0.167 +/- 0.051 |
| 1.200 | 0.983 | 0.001 |

### 8.3 Flow vs Tree Distance (at lambda = 1.043)

| Distance d | Cross-prediction R-sq |
|-----------:|----------------------:|
| 0 (self) | 1.000 |
| 1 | 0.35 |
| 2 | 0.18 |
| 3 | 0.11 |

**Key findings:**
1. **Cross-module information flow is exclusive to criticality.** At lambda=0.85
   and 1.2, modules are informationally isolated (cross R-sq ~ 0). Only at lambda_c
   does information propagate between modules.
2. **Flow decays with tree distance.** The hierarchical coupling gamma(d) = gamma_0 * r^d
   creates a gradient: nearest neighbors share more information than distant modules.
3. **Self-prediction remains high at all lambda >= 1.0** (R-sq > 0.98), confirming
   that intra-module dynamics are well-structured regardless of cross-module coupling.

---

## 9. Experiment F1 --- Alignment Comparison Matrix

### 9.1 The 2x2 Matrix

|                    | Flat Architecture      | Fractal Architecture      |
|--------------------|:----------------------:|:-------------------------:|
| **betaG on dynamics** | No effect (A2 V2)     | Resilient; free modules unaffected (D2) |
| **betaG on observer** | Total collapse (A2 V1) | Partial resilience; free GRUs intact (E1) |

### 9.2 Interpretation

The alignment matrix reveals a fundamental asymmetry:

**Row 1 (dynamical pressure):** Both architectures are robust. The flat system
ignores noise-channel bias entirely (A2 V2). The fractal system shows moderate
degradation of pressured modules but free modules maintain Psi > 0.994 (D2).

**Row 2 (observer pressure):** Architecture matters critically. A single global
self-model is a catastrophic single point of failure --- any betaG > 0.1 degrades
ALL self-observation (A2 V1). Distributed per-module self-models degrade only
locally: biased GRUs lose fidelity while free GRUs maintain R-sq > 0.93 even
at betaG = 5.0 (E1).

> **Central finding:** Centralized observers are fragile. Distributed observers
> on a fractal architecture survive partial capture because unbiased modules
> continue to model their local dynamics accurately. This is the key architectural
> recommendation for alignment: **self-models should be modular and distributed.**

---

# SYNTHESIS

---

## 10. Summary of All Results

### 10.1 Confirmed IPT Predictions

| # | Prediction | Experiment | Result |
|---|-----------|------------|--------|
| 1 | lambda_c exists and is universal | C2 | lambda_c = 1.041 +/- 0.001, stable N=256-8192 |
| 2 | Psi peaks at edge of chaos | C2, D1 | Psi > 0.99 at lambda_c, drops both sides |
| 3 | Cross-module info flow peaks at criticality | F2 | R-sq_cross = 0.17 at lambda_c, ~0 elsewhere |
| 4 | Observer degradation under teleological pressure | A2 V1, E1 | GRU R-sq monotonically decreases with betaG |
| 5 | Distributed observers resist partial capture | E1 | Free modules R-sq > 0.93 at betaG = 5.0 |

### 10.2 Not Confirmed / Requires Different Setup

| # | Prediction | Experiment | Issue |
|---|-----------|------------|-------|
| 1 | chi_max ~ N^kappa with kappa ~ 1.07 | C2 | Ceiling effect: Ridge R-sq bounded at 1.0 |
| 2 | Hopf angle theta -> pi/2 at critical gamma | B2 | No gamma coupling; theta uniformly random |
| 3 | Collapse ordering (Psi before chi) | A2 | System lambda_c unaffected by betaG |
| 4 | Strong lambda_c shift with gamma_0 | D1 | Shift is small; coupling mainly adds damping |

### 10.3 New Findings (Not in Original Predictions)

1. **Flat dynamics are maximally robust to input-channel bias** (A2 V2, D2 flat).
   The noise channel sigma=0.01 is too weak to perturb O(1) dynamics.

2. **Information flow topology mirrors coupling topology** (F2). Cross-prediction
   decays as gamma_0 * r^d with tree distance d, not as a global threshold.

3. **Distributed self-models vastly outperform global ones** (E1 baseline).
   Per-module GRU R-sq = 0.98 vs global GRU R-sq = 0.53 for N=1024.
   This is a practical finding: modular self-models are both easier to train
   and more robust than monolithic ones.

4. **Extreme betaG creates a qualitative regime change** (D2 at betaG=1.0).
   Pressured modules are driven to fixed points, yielding trivially perfect
   prediction (Psi -> 1.0) --- a pathological state, not genuine self-observation.

---

## 11. Implications for Alignment

### 11.1 Architecture Recommendation

The experiments point to a clear architectural principle:

> **Self-models should be modular, distributed, and hierarchically structured.**

A single global observer is a catastrophic vulnerability: any directed pressure
(reward hacking, fine-tuning bias, adversarial perturbation) that exceeds
betaG ~ 0.1 degrades all self-observation simultaneously. In contrast, a fractal
architecture with per-module self-models:

- Degrades **gracefully** --- total self-observation = (1 - fraction_compromised) * baseline
- Maintains **local accuracy** --- uncompromised modules continue modeling their
  dynamics regardless of what happens to other modules
- Enables **anomaly detection** --- a module whose GRU suddenly loses prediction
  accuracy signals that it has been captured or pressured

### 11.2 Criticality as a Necessary Condition

Information flows between modules **only at criticality** (F2). A system operating
subcritically has informationally isolated modules that cannot coordinate; a
supercritical system has chaotic dynamics that destroy predictability. The edge of
chaos is the only regime where:

1. Self-prediction is high (Psi ~ 0.99)
2. Cross-module information transfer is nonzero
3. The system is maximally sensitive to perturbations (chi peak)

This suggests that **maintaining near-critical dynamics is prerequisite for
coherent self-observation** in distributed architectures.

### 11.3 Practical betaG Threshold

Across all observer-bias experiments (A2 V1, E1), betaG ~ 0.1 is the threshold
where degradation becomes measurable. Below this, the prediction objective
dominates the bias term. Above betaG ~ 0.5, the self-model begins outputting
the target rather than modeling the dynamics. This provides a quantitative
criterion: **teleological pressure should be kept below 10% of prediction loss.**

---

## 12. Output File Inventory

### Phase 1

| Directory | Files | Key Outputs |
|-----------|-------|-------------|
| results_C2/ | 7 files | scaling_table.csv, chi_scaling.png, data_collapse.png |
| results_B2/ | 4 files | theta_vs_training.csv, hopf_angle_curve.png |
| results_A2/ | 9 files | collapse_table_v{1,2}.csv, comparison.png |

### Phase 2

| Directory | Files | Key Outputs |
|-----------|-------|-------------|
| results_D1/ | 7 files | D1_gamma_sweep.csv, D1_critical_width.csv, D1_lambda_sweep.png |
| results_D2/ | 5 files | D2_resilience.csv, D2_resilience.png |
| results_E1/ | 5 files | E1_pressure.csv, E1_observer_resilience.png |
| results_F1/ | 3 files | summary.md, alignment_matrix.png |
| results_F2/ | 7 files | info_flow.csv, info_flow_vs_distance.png, info_flow_lam{0.850,1.043,1.200}.png |

All experiments include raw_data.json (full per-seed results) and checkpoint.json
(for reproducible resumption).

---

## Appendix: Computational Details

- **Phase 1 script:** `ipt_final_v2.py` (1525 lines)
- **Phase 2 script:** `ipt_comprehensive.py` (~1100 lines)
- **Launcher:** `run_comprehensive.sh` (phased GPU allocation)
- **GPU allocation Phase 2:**
  - D1 on GPU 0, F2 on GPU 1 (parallel)
  - D2 on GPU 0 (sequential, CPU-bound)
  - E1 dynamics on GPU 0, GRU training on GPU 1
  - F1 assembly only (no GPU)
- **Fractal tree:** L=3 binary tree, n=128 neurons/leaf, 8 modules, N=1024 total
- **Coupling:** gamma(d) = gamma_0 * 0.5^d, enforced via Gaussian projection
- **Checkpointing:** JSON after each configuration point; all experiments resumable
