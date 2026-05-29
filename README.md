# WAKE — Working Against Known Epistemics

Mechanistic interpretability of *Finnegans Wake* using LLM internals.

## Research Question

Does the model hold multiple valid parses in superposition, or does it collapse to one?

Standard NLP treats the Wake as a broken corpus. This project treats it as a correctly functioning system whose operational mode is unknown to current instruments.

## Architecture

```
corpus/         Raw text + McHugh annotations + scholarly sources
graph/          Neo4j knowledge graph (etymology, semantic fields, mythology)
lenses/         Query state definitions (Vico, Kabbalah, Freud, Irish, Norse, Bruno)
engine/         Core pipeline: tokenizer, model, multipass, anamnesis, superposition
interpretability/ Linear probes, attention analysis, residual stream geometry, circuits
api/            FastAPI backend
ui/             React frontend
experiments/    Jupyter notebooks for research runs
tests/          Pytest suite
```

For a detailed walkthrough of every component — what it does, why it was built that way, and how the parts connect — see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Quick Start

```bash
# Start infrastructure
docker compose up neo4j

# Install Python dependencies
poetry install

# Start API
uvicorn api.main:app --reload

# Start UI
cd ui && npm install && npm run dev
```

## Experiments

| ID | Name | Description |
|----|------|-------------|
| E0 | Baseline Calibration | Verify probes work on monolingual text |
| E1 | Entropy Waveform | Full entropy profile of 628 pages |
| E2 | Superposition Discrimination | Genuine multi-parse vs. parse failure |
| E3 | Anamnesis Mapping | Implicit lens discovery from activation fingerprints |
| E4 | Four-Headed Shin | PaRDeS hermeneutic depth test |

## What Success Looks Like

**Minimum**: Entropy waveform shows statistically significant structure across the Wake's 628 pages.

**Strong**: Superposition detection distinguishes genuine multi-parse holding from parse failure.

**Novel**: Anamnesis discovers 2-3 implicit lenses not in the initial set.

**PKD**: Residual stream geometry shows certain passages genuinely between lens clusters — the model holds Vico and Kabbalah simultaneously.

## Model

Designed for open-weights models via TransformerLens. Recommended: `meta-llama/Llama-3.1-70B`. Development/iteration: `meta-llama/Llama-3.1-8B`.

## License

Research use.
