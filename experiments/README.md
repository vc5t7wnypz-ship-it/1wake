# WAKE Experiments

Four experiments that operationalise the mechanistic interpretability
hypothesis: that a large language model reading Finnegans Wake maintains
genuine semantic superposition — multiple interpretive frames simultaneously
active — rather than collapsing to a single parse.

---

## E0 — Baseline Calibration

**Notebook:** `E0_baseline_calibration.ipynb`

**Purpose:**
Establish a null distribution of superposition scores for unambiguous,
monolingual passages.  Five passages — one each in plain English, Latin,
German, Irish Gaelic, and Biblical Hebrew — are run through the probe
ensemble.  Because these passages have no cross-linguistic ambiguity,
superposition scores should remain low (< 0.5).

**Expected outputs:**
- Entropy-per-token histograms, one per language, peaking sharply at low
  entropy.
- Superposition score distribution: mean < 0.3, with no tokens exceeding 0.5.
- Calibration table saved to `experiments/outputs/E0_calibration.json`.

**How to run:**
```bash
jupyter notebook experiments/E0_baseline_calibration.ipynb
```
Or from the command line (requires `nbconvert`):
```bash
jupyter nbconvert --to notebook --execute experiments/E0_baseline_calibration.ipynb
```

---

## E1 — Entropy Waveform of Full Wake

**Notebook:** `E1_entropy_waveform.ipynb`

**Purpose:**
Compute per-token Shannon entropy across pages 3–628 (the full Wake) and
plot an entropy *waveform*.  High-entropy tokens correspond to passages of
maximum multilingual density or polysemy.

**Expected outputs:**
- Interactive Plotly waveform: x = token position, y = entropy (bits).
- Top-10 highest-entropy tokens printed with page / line metadata.
- Waveform saved as `experiments/outputs/E1_entropy_waveform.html`.

**How to run:**
```bash
jupyter notebook experiments/E1_entropy_waveform.ipynb
```

---

## E2 — Superposition vs Parse Failure

**Notebook:** `E2_superposition_discrimination.ipynb`

**Purpose:**
Distinguish genuine semantic superposition (two or more interpretive frames
simultaneously active in the residual stream) from parse failure (the model
simply does not know what to do with the text).

Takes the top-entropy tokens from E1, runs all 6 lenses on each passage,
applies PCA to the residual-stream geometry, and plots a SuperpositionMap:
a scatter of tokens coloured by their classification
(`superposed` / `collapsed` / `opaque`).

**Expected outputs:**
- PCA scatter plot of residual-stream geometry per lens.
- SuperpositionMap saved as `experiments/outputs/E2_superposition_map.html`.
- Classification table: each top-entropy token labelled.

**How to run:**
```bash
jupyter notebook experiments/E2_superposition_discrimination.ipynb
```

---

## E3 — Anamnesis Mapping

**Notebook:** `E3_anamnesis.ipynb`

**Purpose:**
Run the Anamnesis Protocol on passages from all four books of the Wake
(Books I–IV) and collect *novel connections* — patterns in the residual
stream not yet encoded in the McHugh annotation graph.

Uses UMAP to cluster novel connections and visualise candidate new lenses
(interpretive frames not yet in the registry).

**Expected outputs:**
- UMAP scatter of novel-connection embeddings, coloured by book.
- Candidate new lens proposals saved to
  `experiments/outputs/E3_candidate_lenses.json`.
- Interactive visualisation saved as `experiments/outputs/E3_anamnesis_map.html`.

**How to run:**
```bash
jupyter notebook experiments/E3_anamnesis.ipynb
```

---

## Prerequisites

All experiments require:
- Python 3.11+
- `torch`, `transformers`, `accelerate` for model inference
- `plotly`, `matplotlib`, `seaborn` for visualisation
- `umap-learn` for E3

For GPU-accelerated runs (strongly recommended):
- CUDA 12+ with a GPU with ≥ 24 GB VRAM (for GPT-J 6B or Llama-2-13B)

A CPU mock-mode is included in each notebook that generates synthetic
activations so the notebooks can be explored without a GPU.
