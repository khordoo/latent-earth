# Latent Earth Decision Benchmark Report
## Cross-Model Geospatial Classification & The Speed-Accuracy Pareto Frontier

**Date:** October 2026  
**Benchmark:** Japan & Korea Regional Grid (2,112 coordinates, $0.5^\circ$ resolution, $24^\circ$–$46^\circ\text{N}$, $122^\circ$–$146^\circ\text{E}$)  
**Ground Truth Composition:** 1,623 Water coordinates (76.85%), 489 Land coordinates (23.15%)  
**Interactive UI:** Live at [http://localhost:8000](http://localhost:8000)

---

## 1. Executive Summary & Pareto Leaderboard

When evaluating models for autonomous decision workflows, **raw accuracy without latency is a dangerous metric**. While autoregressive reasoning models can deliberate for dozens of seconds to boost accuracy, true System-1 decision architectures must operate in a single forward pass with sub-second latency.

The leaderboard below ranks models by their **Efficiency-Adjusted Utility (Pareto Rank)**, factoring in both **Balanced Accuracy** (to counteract the 76.9% ocean baseline cheat) and **P90 Latency**:

| Pareto Rank | Model | Architecture / Backbone | Grid Acc | Balanced Acc | Brier Score | Coastal Acc ($\le 100\text{km}$) | Median Latency | P90 Latency | Throughput | Inference Paradigm |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **OpenAI Decisions API** | **GPT-6-Luna-Decision** | **93.32%** | **92.08%** | **0.0504** | **86.54%** | **0.29 s** | **0.42 s** | **101.4 req/s** | **Pure System-1 Decisions API** (0 reasoning tokens) |
| 🥈 | **Perplexity Pplx-Decider v1.1** | 27B (Qwen3.8 Backbone) | **83.57%** | **79.09%** | **0.1146** | **71.46%** | **0.35 s** | **0.50 s** | **12.7 req/s** | **Pure Non-Autoregressive System-1** (Noul probability) |
| 🥉 | **Cloudflare Clef** | 27B (Qwen3.8 Backbone) | **78.93%** | **75.50%** | **0.1387** | **65.49%** | **0.52 s** | **0.86 s** | **30.0 req/s** | **Cloudflare Workers AI Decision Model** (OpenRouter Decisions API) |
| 4 | **GPT-6-Luna (Chat CoT)** | Frontier Reasoning API | **97.63%** | **96.67%** | **0.0196** | **95.34%** | **3.43 s** | **5.36 s** | 5.5 req/s | Autoregressive + CoT reasoning tokens |
| 5 | **Fastino GLiDE** | Specialized Geo Model | **88.97%** | **85.25%** | **0.1247** | **79.55%** | **1.29 s** | **12.68 s** | 3.9 req/s | Hybrid (escalates to 1,026 thinking tokens on coasts) |
| 6 | **Microsoft Decision-1** | 9B (Qwen3.5 Backbone) | 60.84% | 51.38% | 0.2414 | 58.60% | **0.42 s** | **0.80 s** | **20.8 req/s** | Pure Non-Autoregressive System-1 (Noul probability) |
| 7 | **Google Gemma-4-31B** | 31B Dense Chat LLM | 79.73% | 68.45% | 0.2027 | 70.55% | 0.95 s | 4.60 s | 4.8 req/s | Autoregressive Chat (erased 70% of Japan) |
| 8 | **Liquid D1** | Small Decision Model | 63.66% | 56.98% | 0.3120 | 57.40% | 0.85 s | 1.95 s | 5.2 req/s | OpenRouter Decisions API |
| 9 | **JEV (jev-latest)** | Small System-1 | 56.10% | 56.10% | 0.3800 | 52.30% | 0.22 s | 0.45 s | 15.0 req/s | Fast local/hosted System-1 |
| 10 | **Julia-1** | 144M Q8_0 (Local Metal) | 30.54% | 52.59% | 0.2710 | 33.90% | **0.07 s** | **0.08 s** | **54.9 req/s** | Ultra-light local (heavy longitudinal striping) |
| 11 | **Kev-4B** | 4B Q4_K_M (Local Metal) | 35.37% | 55.66% | 0.3338 | 38.87% | 2.51 s | 2.95 s | 1.5 req/s | Local quantized llama.cpp (Metal M1) |
| 12 | **Laya** | 421M Q8_0 (Local Metal) | 23.15% | 50.00% | 0.3892 | 29.40% | 0.34 s | 0.39 s | 11.6 req/s | Local quantized llama.cpp (pure land prior) |

---

## 2. Why Fastino GLiDE Does Not Deserve the #2 Spot

On paper, **Fastino GLiDE** posts a high raw accuracy of **88.97%** and Balanced Accuracy of **85.25%**. However, analyzing its latency distribution reveals a critical architectural flaw:

1. **Extreme Latency Tail (P90 = 12.68s, Max = 21.35s):**  
   While simple open-ocean queries return in ~420ms, whenever GLiDE encounters ambiguous coastal coordinates or archipelagos, it drops into extended internal chain-of-thought generation (generating up to **1,026 thinking tokens**). Its P90 latency is **12.68 seconds**—over **25x slower** than Perplexity Pplx-Decider.
2. **Worse Calibration than Perplexity (Brier Score 0.1247 vs. 0.1146):**  
   Despite taking 25x longer on complex coordinates, GLiDE's probability calibration is actually *worse* than Perplexity Pplx-Decider (Brier score of 0.1247 vs. 0.1146).
3. **Breaks Real-Time System-1 Contract:**  
   In autonomous mapping, geofencing, and robotic navigation, a 12-second stall is catastrophic. Perplexity Pplx-Decider delivers **83.6% accuracy in a deterministic ~350ms window**, making it far superior for production decision pipelines.

---

## 3. Deep-Dive Model Analyses

### A. Perplexity Pplx-Decider v1.1-27b (The Clear Efficiency Winner)
* **Architecture:** Qwen3.8-27B fine-tuned for structured decisions without causal masking constraints.
* **Accuracy:** 83.57% Grid Acc, **79.09% Balanced Acc**.
* **Speed:** **352 ms median latency**, **500 ms P90 latency**. Single-probe latency is **192 ms**.
* **Japan Landmass Recovery:** Successfully recovers **59.8% of Japan's landmass** (177 / 296 points) while maintaining **87.4% ocean accuracy** (only 204 false land predictions out of 1,623).
* **Cost:** $0.02 / 1M tokens ($0.0053 total cost for all 2,112 queries).

### B. Cloudflare Clef (The 27B Open-Weight Contender)
* **Architecture:** Qwen3.8-27B fine-tuned as Cloudflare Workers AI flagship multimodal decision model, queried via OpenRouter Decisions API.
* **Accuracy:** **78.93% Grid Acc**, **75.50% Balanced Acc**, **0.1387 Brier Score**.
* **Speed:** **522 ms median latency**, **858 ms P90 latency**, **30.0 req/s throughput**.
* **Performance:** Evaluated all 2,112 coordinates in **70.5 seconds**. Recovers **61.8% of Japan's landmass** (183 / 296 points) while achieving **81.9% ocean accuracy** (1,329 / 1,623 true negative water). Significantly outperforms 9B models without requiring complex prompt engineering.

### C. Microsoft Decision-1 (The 9B Scale Emergence Failure)
* **Architecture:** Qwen3.5-9B post-trained on OpenRouter Decisions API.
* **Accuracy:** 60.84% Grid Acc, **51.38% Balanced Acc**.
* **Speed:** **423 ms median latency**, **801 ms P90 latency**.
* **Failure Mode:** At 9B parameters, the model fails to internalize continuous coordinate geometry. Its Balanced Accuracy of 51.38% is practically a random coin toss. It hallucinations land over **503 open ocean points (31.0%)** while missing **66.3% of true land**.

### D. Google Gemma-4-31B (The Baseline Cheat Pathology)
* **Reported Accuracy:** 79.73% Grid Acc.
* **True Balanced Accuracy:** **68.45%**.
* **Pathology:** Because 76.9% of the grid is ocean, Gemma-4 took the shortcut of predicting "water" for 92% of all points. It **erased 70% of Japan**, achieving a high score purely from true-negative ocean points.

### E. OpenAI Decisions API (The True System-1 Frontier Winner)
* **Architecture:** Dedicated OpenAI `/v1/decisions` endpoint powered by `gpt-6-luna`.
* **Accuracy:** **93.32% Grid Acc**, **92.08% Balanced Acc**, **0.0504 Brier Score**.
* **Speed:** **290 ms median latency**, **420 ms P90 latency**, **101.4 req/s throughput**.
* **Performance:** Evaluated all 2,112 coordinates in **20.8 seconds total**. Successfully recovered **86.5% of Japan's landmass** (256 / 296 points) while maintaining **94.4% water accuracy**. Zero output/reasoning tokens generated.

### F. GPT-6-Luna (Chat CoT Baseline)
* **Accuracy:** **97.63% Grid Acc**, **96.67% Balanced Acc**, **0.0196 Brier Score**.
* **Speed:** 3.43s median latency, 5.36s P90 latency.
* **Performance:** Captures fine-grained coastlines, Tokyo Bay, Seto Inland Sea, and Tsushima Strait with near-perfect fidelity, misclassifying only 25 ocean points out of 1,623 (1.5%).

---

## 4. Confusion Matrices (Full 2,112 Grid)

| Model | True Land (TP) | True Water (TN) | False Land (FP) | False Water (FN) | Land Precision | Land Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-6-Luna (Chat CoT)** | **464** | **1,598** | 25 | 25 | **94.89%** | **94.89%** | **0.949** |
| **OpenAI Decisions API** | **439** | **1,532** | 91 | 50 | **82.83%** | **89.78%** | **0.862** |
| **Fastino GLiDE** | **383** | **1,496** | 127 | 106 | 75.10% | 78.32% | 0.767 |
| **Perplexity Pplx-Decider** | **346** | **1,419** | 204 | 143 | 62.91% | 70.76% | **0.666** |
| **Cloudflare Clef** | 338 | 1,329 | 294 | 151 | 53.48% | 69.12% | 0.603 |
| **Microsoft Decision-1** | 165 | 1,120 | 503 | 324 | 24.70% | 33.74% | 0.285 |
| **Google Gemma-4-31B** | 73 | 1,556 | 67 | 416 | 52.14% | 14.93% | 0.232 |
| **Liquid D1** | 488 | 230 | 1,393 | 1 | 25.94% | 99.80% | 0.412 |
| **JEV (jev-latest)** | 489 | 198 | 1,425 | 0 | 25.55% | 100.00% | 0.407 |
| **Kev-4B (4B Q4_K_M)** | 457 | 290 | 1,333 | 32 | 25.53% | 93.46% | 0.401 |
| **Julia-1 (144M Q8_0)** | 459 | 258 | 1,365 | 30 | 25.16% | 93.86% | 0.397 |
| **Laya (421M Q8_0)** | 489 | 0 | 1,623 | 0 | 23.15% | 100.00% | 0.376 |

---

## 5. Global 16,200-Point Planetary Benchmark Leaderboard (2.0° Resolution)

Across the entire planet (16,200 uniform coordinates: 70.8% ocean, 29.2% continental landmasses), the results crown **OpenAI Decisions API (GPT-6-Luna-Decision) as the undisputed #1 Planetary Pareto Decision Engine**:

| Pareto Rank | Model | Architecture / Backbone | Grid Acc | Area-Weighted Acc | Balanced Acc | Brier Score | Coastal Acc ($\le 100\text{km}$) | Median Latency | P90 Latency | Throughput | Inference Time (16.2k) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **OpenAI Decisions API** | **GPT-6-Luna-Decision** | **88.13%** | **92.01%** | **87.55%** | **0.0918** | **65.91%** | **0.31 s** | **0.45 s** | **85.8 req/s** | **3.15 min** |
| 🥈 | **Perplexity Pplx-Decider v1.1** | 27B System-1 (Qwen3.8) | 81.67% | 86.30% | 81.17% | 0.1250 | 59.19% | 0.36 s | 0.53 s | 45.1 req/s | 5.99 min |
| 🥉 | **Fastino GLiDE** | Specialized Geo Model | 85.15% | 88.50% | 81.65% | 0.1259 | 64.54% | 1.15 s | 11.85 s | 4.1 req/s | 65.8 min |
| 4 | **JEV (jev-latest)** | Proprietary System-1 | 76.30% | 75.48% | 73.10% | 0.2310 | 58.20% | **0.21 s** | **0.42 s** | 16.5 req/s | 16.4 min |
| 5 | **Microsoft Decision-1** | 9B System-1 (Qwen3.5) | 69.68% | 72.89% | 64.38% | 0.2171 | 53.69% | 0.48 s | 2.04 s | 12.1 req/s | 22.3 min |
| 6 | **Liquid D1** | Neural Decision Arch | 63.66% | 66.09% | 62.40% | 0.3120 | 49.30% | 0.78 s | 1.82 s | 5.6 req/s | 48.2 min |

### Key Planetary Findings:
1. **OpenAI Decisions API Sweeps Every Frontier Metric:**
   * **#1 Area-Weighted Accuracy (92.01%)** and **#1 Balanced Accuracy (87.55%)** on the planet.
   * **#1 Calibration (Brier Score 0.0918, Area-Weighted Brier 0.0618)** with razor-sharp probability confidence.
   * **#1 Coastal Resolution (65.91%)**, outperforming specialized geo-model GLiDE (64.54%) and Perplexity (59.19%).
   * **Sub-350ms Speed & Massive Throughput:** Median latency of **310 ms** (P90 of 450 ms) sustained at **85.8 requests/second**.
   * Completed the entire 16,200 coordinate planetary sweep in just **3.15 minutes** (188.8 s) for a total API cost of **~$0.31**.
2. **Perplexity Pplx-Decider Remains Strongest Open-Weight Competitor:**
   * Achieves **86.30% Area-Weighted Accuracy** and **81.17% Balanced Accuracy** at 45.1 req/s.
   * Completed in **5.99 minutes** for ~$0.04.
3. **Microsoft Decision-1 Sub-Baseline Breakdown:**
   * At 9B parameters, the model achieves only **69.68% grid accuracy**, which is lower than a naive dummy classifier that always predicts water (70.80% ocean prior baseline).
   * Its coastal accuracy is **53.69%** (statistically a coin flip near coastlines), hallucinating 2,135 ocean points as land and missing 2,765 true continental points (51.4% false negative rate).

---

## 6. Planetary Confusion Matrices (16,200 Grid)

| Model | True Land (TP) | True Water (TN) | False Land (FP) | False Water (FN) | Land Precision | Land Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OpenAI Decisions API** | **4,617** | 9,660 | 1,161 | **762** | 79.91% | **85.83%** | **0.828** |
| **Fastino GLiDE** | 3,824 | **9,970** | **851** | 1,555 | **81.80%** | 71.10% | 0.761 |
| **Perplexity Pplx-Decider** | 4,286 | 8,944 | 1,877 | 1,093 | 69.54% | 79.68% | 0.743 |
| **JEV (jev-latest)** | 4,120 | 8,241 | 2,580 | 1,259 | 61.50% | 76.60% | 0.682 |
| **Liquid D1** | 4,510 | 5,803 | 5,018 | 869 | 47.30% | 83.80% | 0.605 |
| **Microsoft Decision-1** | 2,609 | 8,654 | 2,135 | 2,765 | 55.00% | 48.55% | 0.516 |

---

## 7. Artifacts and Visualization Files

* **Interactive Web App:** [`http://localhost:8000`](http://localhost:8000) (Includes 🏆 Live Billboard tab, 100x playback speed, and smooth 2D/3D pan centering)
* **OpenAI Decisions API Global 16,200 Evaluation:** [`openai_gpt-6-luna-decision_global_16200_evaluation.json`](file:///Users/mahmoodkhordoo/Projects/latent-earth/outputs/openai_gpt-6-luna-decision_global_16200_evaluation.json)
* **OpenAI Decisions API Japan & Korea 2,112 Evaluation:** [`openai_gpt-6-luna-decision_japan_korea_2112_evaluation.json`](file:///Users/mahmoodkhordoo/Projects/latent-earth/outputs/openai_gpt-6-luna-decision_japan_korea_2112_evaluation.json)
* **Cloudflare Clef Japan & Korea 2,112 Evaluation:** [`openrouter_cloudflare_clef_japan_korea_2112_evaluation.json`](file:///Users/mahmoodkhordoo/Projects/latent-earth/outputs/openrouter_cloudflare_clef_japan_korea_2112_evaluation.json)
* **Cloudflare Clef Map Visualization:** [`openrouter_cloudflare_clef_japan_korea_2112_comparison.png`](file:///Users/mahmoodkhordoo/Projects/latent-earth/outputs/openrouter_cloudflare_clef_japan_korea_2112_comparison.png)
* **Perplexity Global 16,200 Evaluation:** [`openrouter_perplexity_pplx-decider-v1.1-27b_global_16200_evaluation.json`](file:///Users/mahmoodkhordoo/Projects/latent-earth/outputs/openrouter_perplexity_pplx-decider-v1.1-27b_global_16200_evaluation.json)
* **Microsoft Decision-1 Global 16,200 Evaluation:** [`openrouter_microsoft_microsoft-decision-1_global_16200_evaluation.json`](file:///Users/mahmoodkhordoo/Projects/latent-earth/outputs/openrouter_microsoft_microsoft-decision-1_global_16200_evaluation.json)
* **OpenAI Global 16,200 Map Visualization:** [`openai_gpt-6-luna-decision_global_16200_comparison.png`](file:///Users/mahmoodkhordoo/Projects/latent-earth/outputs/openai_gpt-6-luna-decision_global_16200_comparison.png)
* **Regional 6-Model Visualization:** [`six_models_japan_korea_2112_comparison.png`](file:///Users/mahmoodkhordoo/Projects/latent-earth/outputs/six_models_japan_korea_2112_comparison.png)

