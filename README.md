# Gender Performance Gaps in Multilingual ASR: The Role of Model Scale and Fine-Tuning

An extension project analyzing how **model scale** and **fairness-aware fine-tuning** affect gender-based performance disparities in multilingual Automatic Speech Recognition (ASR).

This work builds on Attanasio et al., *"Multilingual Speech Models for Automatic Speech Recognition Exhibit Gender Performance Gaps"*, extending it to study the role of model scale and parameter-efficient fine-tuning.

[![Data and Artifacts](https://img.shields.io/badge/Data%20and%20Artifacts-HF%20Dataset-yellow)](https://huggingface.co/datasets/g8a9/multilingual-asr-gender-gap)

**Authors:** Anushka Nayak, Arka Sharma, Serena Chang

---

## Overview

Prior work showed that multilingual ASR systems exhibit measurable gender-based gaps in Word Error Rate (WER). We reproduce and extend these findings to answer two research questions:

- **RQ1.** Do larger ASR models reduce the gender performance gap?
- **RQ2.** Can fairness-aware fine-tuning fix what scale alone cannot?

We evaluate the **Whisper** (tiny / medium / large-v3) and **SeamlessM4T** (medium / large-v2) model families on the English split of **FLEURS**, and apply parameter-efficient **LoRA** fine-tuning with three fairness conditions.

### Metrics
- **WER / CER** — overall and stratified by group (Majority = Male, Minority = Female)
- **Error Rate Gap (E score):** `E = 100 · (φ(r_female) − φ(r_male)) / φ(r_male)`, where `E = 0` indicates parity.

---

## Repository Structure

The repository is organized by project section. Each section contains its own `bash/`, `configs/`, and `src/` directories.

```
.
├── whisper/      # Whisper scaling experiments (tiny / medium / large-v3)
├── seamless/     # SeamlessM4T scaling experiments (medium / large-v2)
├── fine-tune/    # Fairness-aware LoRA fine-tuning and gender-gap evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

Within each section:
- `bash/` — shell scripts to launch experiments
- `configs/` — experiment and model configuration files
- `src/` — transcription, evaluation, and fine-tuning source code

---

## Getting Started

1. Create a new Python environment (Python ≥ 3.9 recommended).
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Navigate to the relevant section (`whisper/`, `seamless/`, or `fine-tune/`) and run the scripts in `bash/` or `src/`. Script names and parameters are mostly self-explanatory.

---

## Experiments

### 1. Scaling Whisper
Evaluated tiny, medium, and large-v3 on English FLEURS.

| Model | Overall WER | Majority (M) WER | Minority (F) WER | E score |
|---|---|---|---|---|
| Whisper-tiny | 0.1320 | 0.1689 | 0.1099 | −34.92 |
| Whisper-medium | 0.0461 | 0.0477 | 0.0452 | −5.36 |
| Whisper-large-v3 | 0.0413 | 0.0425 | 0.0405 | −4.58 |

**Finding:** All Whisper models perform better on female speakers (negative E). The gap shrinks as scale increases, with large-v3 giving the best accuracy *and* fairness.

### 2. Scaling SeamlessM4T
Evaluated medium and large-v2 on English FLEURS.

| Model | Overall WER | Majority (M) WER | Minority (F) WER | E score |
|---|---|---|---|---|
| SeamlessM4T-medium | 0.2609 | 0.2646 | 0.2588 | −2.18 |
| SeamlessM4T-large-v2 | 0.2556 | 0.2527 | 0.2574 | +1.85 |

**Finding:** Scaling up lowers WER/CER and reduces the absolute disparity, but the **bias direction flips** (medium favors female speakers; large-v2 slightly favors male speakers).

### 3. Fairness-Aware Fine-Tuning
LoRA fine-tuning of Whisper-tiny under three conditions, evaluated by E score.

**LoRA setup:** rank `r = 4`, layers 2–3, targets `q_proj` / `v_proj`, ~1% of parameters trained, 2,600 samples, 3 epochs.

| Condition | E score | Effect |
|---|---|---|
| No fine-tuning | −17.1% | baseline gap |
| Baseline FT (control) | −13.8% | gap reduced ✓ |
| Balanced FT (equal sampling) | −18.7% | gap widened ✗ |
| Weighted FT (2× female loss) | −20.8% | gap widened ✗ |

**Finding:** Standard fine-tuning reduced the gap most. Because Whisper-tiny disadvantages *male* speakers, the female-oriented fairness interventions (balanced sampling, weighted loss) pushed in the wrong direction and amplified the gap.

---

## Key Takeaways

- Larger models improve overall accuracy, robustness, and fairness.
- The **direction** of the gender gap is not fixed: it flips across model scales and families, so bias must be **measured before it is corrected**.
- Miscalibrated fairness interventions can amplify disparity rather than reduce it.

---

## Limitations & Future Work

- Scope limited to one language (English) and one dataset (FLEURS).
- No causal explanation for why gaps grow or shrink.
- Future directions: more recent ASR models, multiple languages, and higher LoRA ranks with multilingual training.

---

## License

Released under the Apache-2.0 License. See [`LICENSE`](LICENSE) for details.

## Acknowledgments

This project extends the work of Attanasio et al. We thank the authors for releasing their [code and dataset](https://huggingface.co/datasets/g8a9/multilingual-asr-gender-gap).
