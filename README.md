# AutoValue AI 🚗

**AI-powered used car price estimation** — ZHAW KI-Anwendungen FS2026

## Pipeline

```
Car Photo  ──► CV Block (CLIP zero-shot)  ──► condition_score
                                                    │
Structured data (make/year/km/fuel/gear) ────────── ┤
                                                    ▼
                                       ML Block (GradientBoosting)
                                                    │
                                            predicted_price
                                                    │
RAG: similar cars from dataset ─────────────────── ┤
                                                    ▼
                                       NLP Block (GPT-4o-mini)
                                                    │
                                        Explanation + Chatbot
```

## Setup

### 1. Install uv

```bash
# Windows
winget install astral-sh.uv

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
cd autovalue-ai
uv sync
```

### 3. Download dataset

Download the UK Used Cars Dataset from Kaggle:  
https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

Place all CSV files (`audi.csv`, `bmw.csv`, `cclass.csv`, etc.) into `data/`.

### 4. Train models

```bash
uv run python -c "from src.ml_block import train_and_save; train_and_save('data')"
```

This saves trained model artefacts to `models/`.

### 5. Run app

```bash
uv run python app.py
```

Open http://localhost:7860 in your browser.

## HuggingFace Spaces Deployment

1. Create a new Space (Gradio SDK)
2. Push this repository
3. Add `OPENAI_API_KEY` as a Space secret
4. Upload the `models/` folder (trained artefacts)

## Project Structure

```
autovalue-ai/
├── app.py                 # Gradio UI
├── src/
│   ├── cv_block.py        # CLIP zero-shot condition score
│   ├── ml_block.py        # GradientBoosting price prediction
│   └── nlp_block.py       # GPT-4o-mini + RAG + chatbot
├── data/                  # Dataset CSVs (not committed)
├── models/                # Trained model artefacts (not committed)
├── notebooks/             # Training notebook
├── pyproject.toml         # uv dependencies
├── requirements.txt       # HuggingFace Spaces
└── DOCUMENTATION.md       # Project documentation
```

## Blocks

| Block | Method |
|---|---|
| Computer Vision | CLIP zero-shot (HuggingFace) |
| ML Numeric Data | GradientBoosting + RandomizedSearchCV |
| NLP | GPT-4o-mini + Prompt Engineering + RAG |

## Ethics

AutoValue AI is classified as **Limited-Risk** under the EU AI Act (2024).  
The chatbot discloses its AI nature. All estimates are advisory — not binding.
