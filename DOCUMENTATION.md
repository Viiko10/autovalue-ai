# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title: AutoValue AI — KI-gestützte Gebrauchtwagenpreisbewertung
- Student: Viktor (ZHAW Wirtschaftsinformatik, Sem 6)
- GitHub repository URL: https://github.com/Viiko10/autovalue-ai
- Deployment URL: https://huggingface.co/spaces/Viiko10/autovalue-ai
- Submission date: 07.06.2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: ML Numeric Data
- Primary block 2: NLP

If a third block is selected, it is documented and graded separately as extra work.
> **Computer Vision is the third block (bonus):** CLIP zero-shot produces `condition_score` fed into the ML block — all three blocks are integrated into one end-to-end pipeline.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Used car buyers and sellers lack transparent, data-driven price references. Manual valuations are time-consuming and inconsistent.
- Goal: Build an AI pipeline that estimates the market price of a used car from structured data (make, year, mileage) and a photo, and explains the estimate in natural language.
- Success criteria: RMSE < 3,000 GBP on test set (aspirational target; achieved RMSE = 5,451 GBP — acceptable given the UK dataset's wide price range of £500–£200,000 and heterogeneous makes); RAG-grounded NLP explanation that references comparable market listings; working Gradio demo on HuggingFace Spaces.

### 1.2 Integration Logic
- How the selected blocks interact: The CV block analyses a user-uploaded car photo and outputs a `condition_score` ∈ [0,1]. This score is passed as an additional feature to the ML block alongside structured car metadata. The ML block produces a price estimate which, together with the `condition_score` and RAG-retrieved similar listings, is fed to the NLP block for a grounded explanation.
- Data and output flow between blocks:

```
User Photo  ──► [CV Block: CLIP zero-shot]  ──► condition_score (float)
                                                         │
Structured data (make/model/year/km/fuel/gear) ──────────┤
                                                         ▼
                                          [ML Block: GradientBoosting]
                                                         │
                                                  predicted_price
                                                         │
Similar cars from dataset (RAG) ──────────────────────── ┤
                                                         ▼
                                          [NLP Block: GPT-4o-mini]
                                                         │
                                              Natural language explanation
                                              + interactive chatbot
```

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [UK Used Cars Dataset (Kaggle)](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) | CSV (11 make-specific files) | ~100,000 rows | Training + test set for price prediction |
| 2 | Same dataset (kept in-memory) | Pandas DataFrame | ~100,000 rows | RAG knowledge base: similar car retrieval for NLP block |

#### 2A.2 Preprocessing, EDA and Features
- Cleaning steps: Remove rows with price < 500 or > 200,000 (outliers). Drop rows with missing `price`, `year`, or `mileage`. Combine 11 make-specific CSVs, add `make` column from filename. See [`src/ml_block.py`, lines 53–86](src/ml_block.py#L53-L86).
- Preprocessing steps: OrdinalEncoder for categorical features (`make`, `fuel_type`, `transmission`) with `handle_unknown='use_encoded_value'`. StandardScaler applied only for MLPRegressor. See [`src/ml_block.py`, lines 111–134](src/ml_block.py#L111-L134).
- EDA key findings (see [`notebooks/ml_training.ipynb`](notebooks/ml_training.ipynb) and `demo/eda.png`):
  - Price distribution is right-skewed; median ≈ £10,500, 75th percentile ≈ £18,000. Outlier removal is essential.
  - Strong negative correlation between mileage and price: cars with > 150,000 km rarely exceed £10,000.
  - Year is the strongest single predictor: post-2018 cars command significantly higher prices.
  - BMW, Mercedes-Benz, and Audi have median prices 2–3× higher than Ford or Vauxhall.
  - Diesel vehicles slightly more expensive on average, but the premium has narrowed (dataset spans 2000–2021).
  - Automatic transmission adds roughly £1,500–2,500 to the median price across makes.
- Feature engineering and selection:
  - `car_age = current_year - year` (age matters more than raw year; see [`src/ml_block.py`, line 93](src/ml_block.py#L93))
  - `km_per_year = mileage / (car_age + 0.1)` (usage intensity proxy; normalises mileage for age)
  - `is_luxury`: binary flag if make ∈ {BMW, Audi, Mercedes-Benz, Porsche, Tesla, …}
  - `is_sport`: binary flag if model name contains 'sport', 'gti', 'turbo', 'amg', …
  - `condition_score`: float from CV block [0.1–1.0] — bridges CV ↔ ML
  - **Excluded** `price_per_km`: would be data leakage (contains target information)
  - See [`src/ml_block.py`, lines 89–104](src/ml_block.py#L89-L104).

#### 2A.3 Model Selection
- Models tested: LinearRegression (baseline), RandomForestRegressor, GradientBoostingRegressor, MLPRegressor
- Why these models were chosen: LinearRegression establishes a baseline. RandomForest and GradientBoosting are ensemble methods that capture non-linear price interactions. MLPRegressor validates neural-network approach; requires StandardScaler. GradientBoosting was tuned with RandomizedSearchCV as it typically outperforms on tabular data.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish baseline | Raw features, no engineering | LinearRegression | RMSE 6,567 GBP, R²=0.554 | — |
| 2 | Improve with ensembles | Add car_age, km_per_year, is_luxury, is_sport | RandomForest, GradientBoosting | RMSE 5,597 GBP, R²=0.676 | −15% |
| 3 | Neural net + hyperparameter tuning | MLPRegressor + RandomizedSearchCV on GB | MLPRegressor (R²=0.685), GradientBoosting_tuned | RMSE 5,451 GBP, R²=0.693 | −3% |

See [`src/ml_block.py`, lines 145–232](src/ml_block.py#L145-L232) for full `train_and_save()` implementation.

#### 2A.5 Evaluation and Error Analysis
- Metrics used: RMSE, MAE, R², MAPE (all computed in `evaluate_model()`, [`src/ml_block.py`, lines 137–142](src/ml_block.py#L137-L142)); 5-fold cross-validation on GradientBoosting.
- Final results (GradientBoosting_tuned, best params: n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8): RMSE=5,451 GBP, MAE=3,199 GBP, R²=0.693, MAPE=18.4% — 5-fold CV RMSE=5,555 ± 42 GBP (stable)
- Feature importance (GradientBoosting): `year` and `mileage` are the two dominant features (combined ~55% importance), followed by `car_age` and `km_per_year`. Categorical features (`make`, `transmission`, `fuel_type`) contribute ~25% collectively. `condition_score` (from CV block) contributes ~3–5% — modest but consistent, as expected for a noisy zero-shot signal. See [`notebooks/ml_training.ipynb`](notebooks/ml_training.ipynb) (Feature Importance cell) and `demo/feature_importance.png`.
- Error patterns and likely causes: Higher errors on rare/exotic makes (low training samples). New electric vehicles (Tesla) systematically underpriced — UK dataset has fewer EV samples. Premium models with low mileage sometimes predicted below market value.

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s): `condition_score` (float, 0.1–1.0) from CV block — injected as an additional feature column during both training and inference.
- Outputs provided to other block(s): `predicted_price` (float, GBP) → NLP block for explanation; `train_df` (DataFrame) → NLP block for RAG retrieval.

---

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | OpenAI API (GPT-4o-mini) | External LLM API | — | Generate price explanations + chatbot responses |
| 2 | UK Used Cars Dataset (same as ML block) | Pandas DataFrame (in-memory) | ~100,000 rows | RAG knowledge base: retrieve 3 similar cars per query |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing: Car attributes are formatted into XML-tagged blocks (`<car_data>`, `<similar_listings>`) following Week 10 prompt injection pattern. Numeric values are formatted for readability (e.g., `{mileage:,.0f} km`).
- Prompt design or retrieval setup:
  - **RAG retrieval:** pandas filter on `make` (exact), `year` (±2), `mileage` (±40%) → top-3 similar cars. No vector DB needed for structured data — structured filter is equivalent to Hybrid Search (Week 12). See [`src/ml_block.py`, lines 293–316](src/ml_block.py#L293-L316).
  - **Prompt structure** (Week 10 — Context + Instructions + Description + Input):
    - System prompt: role ("AutoValue AI"), task ("explain price"), description ("input = structured car data"), format ("3-4 sentences")
    - User prompt: question first → grounding constraint → `<car_data>` XML → `<similar_listings>` XML (Week 12 pattern: information placed at start and end)
  - See [`src/nlp_block.py`, lines 21–55](src/nlp_block.py#L21-L55).

#### 2B.3 Approach Selection
- Approach used: Decoder-only LLM (GPT-4o-mini) + Prompt Engineering (Week 10) + lightweight dataset-based RAG (Week 11) + Chat History (Week 9)
- Alternatives considered:
  - **Fine-tuning**: rejected — requires labeled Q&A dataset, expensive, doesn't solve data freshness.
  - **Vector DB RAG (FAISS/Chroma)**: not needed — car data is structured; pandas filter (Hybrid Search) gives more precise results than semantic similarity for exact make/model matches.
  - **Larger model (GPT-4o)**: not needed — explanation task is well-defined and constrained; gpt-4o-mini sufficient and cheaper.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Basic explanation | Simple prompt, no RAG | GPT-4o-mini, minimal system prompt | Qualitative: generic, mentions car name | Baseline |
| 2 | Grounded explanation | Add `<car_data>` XML, explicit grounding constraint | Week 10 structured prompt | Qualitative: specific price factors cited | Less hallucination |
| 3 | Market-referenced explanation | Add RAG: `<similar_listings>` with 3 comparable cars | Full RAG-grounded prompt (Week 11/12) | Qualitative: comparable listings cited | Traceable, data-grounded |

See [`src/nlp_block.py`, lines 21–55](src/nlp_block.py#L21-L55).

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy: Qualitative grounding check — does the answer reference the `<similar_listings>` evidence? Does it stay within the XML context? (Week 12: Grounding criterion). Manual review of 10 sample outputs.
- Results: Explanations cite 2-3 price factors and reference at least one comparable listing when available. Hallucinations absent when evidence provided.
- Error patterns and likely causes: When no similar cars are found (rare make), the LLM can fall back on general knowledge despite the grounding constraint. Partially mitigated by the `get_similar_cars()` fallback chain (exact match → same make → random sample), ensuring at least 3 listings are always provided.

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s): `predicted_price` from ML block; `condition_score` + `condition_label` from CV block; `similar_cars` DataFrame retrieved from ML block's training data.
- Outputs provided to other block(s): Natural language explanation (string) → displayed in Gradio UI. Chat history (list of dicts) maintained across chatbot turns.

---

### 2C. Computer Vision (If selected)

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | User-uploaded car photo | Image (PIL) | Variable | Input for CLIP zero-shot condition assessment |
| 2 | [openai/clip-vit-base-patch32 (HuggingFace)](https://huggingface.co/openai/clip-vit-base-patch32) | Pre-trained model | 400M image-text pairs (training data) | Zero-shot image-text similarity scoring |

#### 2C.2 Preprocessing and Augmentation
- Image preprocessing: PIL Image passed directly to CLIPProcessor — handles resize to 224×224, normalization, and tokenization internally. No manual augmentation needed for zero-shot inference.
- Augmentation strategy: N/A for zero-shot inference. If transfer learning were used (EfficientNetB0), augmentation would include RandomFlip + RandomRotation (Week 5/6).

#### 2C.3 Model Selection
- Vision model(s) used: CLIP (Contrastive Language-Image Pre-Training, OpenAI) — `openai/clip-vit-base-patch32` via HuggingFace Transformers.
- Why these model(s) were chosen:
  - **Zero-shot**: No labeled car damage dataset required → faster deployment, no annotation cost.
  - **CLIP**: Trained on 400M (image, text) pairs — understands damage descriptions linguistically, making damage labels like "a car with severe damage" directly comparable to car images.
  - Directly applies Week 7 content (Zero-Shot classification with CLIP).
  - Alternative (Transfer Learning / EfficientNetB0): Would need a labeled car damage dataset and GPU training. Described in Week 6 content but not required for this zero-shot approach.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Binary classification | "damaged" vs "undamaged" | CLIP zero-shot (2 labels) | Qualitative: coarse | Baseline |
| 2 | 4-class condition score | 4 damage labels, weighted sum | CLIP zero-shot (4 labels) | condition_score ∈ [0.1, 1.0] | More granular, continuous score |
| 3 | Score as ML feature | condition_score added to ML feature matrix | CLIP → GradientBoosting | RMSE improvement when condition provided | Bridges CV ↔ ML |

See [`src/cv_block.py`, lines 30–46](src/cv_block.py#L30-L46).

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks: Manual visual check on 10 car photos (good/damaged). CLIP probabilities match human intuition for clearly damaged cars. Edge cases: cars with dirty windshields or unusual angles sometimes scored lower than expected.
- Final results: condition_score reliably distinguishes "perfect" (≥0.85) from "severely damaged" (≤0.3) cars. Middle range (fair/good) is less precise.
- Error patterns and limitations:
  - **Angle dependency**: side view vs. front view gives different scores for same car.
  - **Lighting**: dark/low-contrast images produce more uncertain probability distributions.
  - **Domain gap**: CLIP trained on web images — professional damage assessment photos may differ.
  - Mitigation: Default score 0.75 (Good) used when no image provided.

#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s): None — CV block is the first stage in the pipeline.
- Outputs provided to other block(s): `condition_score` (float) → ML block as additional feature; `condition_label` (str: Excellent/Good/Fair/Poor) → NLP block for human-readable context.

---

## 3. Deployment

- Deployment URL: https://huggingface.co/spaces/Viiko10/autovalue-ai
- Main user flow:
  1. User uploads car photo (optional) → CLIP scores condition
  2. User enters make/model/year/mileage/fuel/transmission → GradientBoosting predicts price
  3. GPT-4o-mini generates RAG-grounded explanation citing comparable listings
  4. User asks follow-up questions in chatbot tab; chat history maintained across turns
- Screenshots:
  - [`demo/screenshot_1_estimator.png`](demo/screenshot_1_estimator.png) — Price Estimator tab (input form)
  - [`demo/screenshot_2_estimator.png`](demo/screenshot_2_estimator.png) — Price Estimator tab with result (CV condition score, ML price, NLP explanation)
  - [`demo/screenshot_chat.png`](demo/screenshot_chat.png) — Chat with AutoValue AI (follow-up question + GPT-4o-mini response)
  - [`demo/screenshot_about.png`](demo/screenshot_about.png) — About tab (system overview, pipeline, metrics)

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- Environment setup:
  ```bash
  # Install uv (https://docs.astral.sh/uv/)
  # Windows: winget install astral-sh.uv
  
  cd autovalue-ai
  uv sync           # creates .venv and installs all dependencies
  ```

- Data setup:
  ```bash
  # Download UK Used Cars Dataset from Kaggle:
  # https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes
  # Place all CSV files (audi.csv, bmw.csv, etc.) into data/
  ```

- Training command(s):
  ```bash
  uv run python -c "from src.ml_block import train_and_save; train_and_save('data')"
  # Saves models/best_model.joblib, models/encoder.joblib, models/scaler.joblib
  # Also saves models/train_data.joblib (for RAG retrieval)
  ```

- Inference/run command(s):
  ```bash
  # Local Gradio app
  uv run python app.py
  
  # HuggingFace Spaces: push repo, set OPENAI_API_KEY as secret
  ```

- Reproducibility notes:
  - All random operations use `random_state=42`.
  - Train/test split: 80/20, stratified by price range is not applied (regression task).
  - Python ≥ 3.11 required. All dependency versions pinned in `pyproject.toml`.
  - Pre-trained CLIP model downloaded automatically via HuggingFace Hub on first run.

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [x] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [x] A core section is done exceptionally well
- [x] Extended evaluation
- [x] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

Evidence for selected bonus items:

**Third block (CV):** CLIP zero-shot implemented as first pipeline stage. `condition_score` bridges CV → ML (not just a display feature — it's a trained model input). Both CLIP (Week 7) and Transfer Learning (EfficientNetB0, Week 6) architectures described.

**Multiple data sources:** (1) UK Used Cars CSV — training data; (2) Same dataset as RAG knowledge base; (3) OpenAI CLIP model (HuggingFace, 400M pairs training data); (4) OpenAI GPT-4o-mini API.

**Extended evaluation:** 5-fold cross-validation + RandomizedSearchCV (10 iterations, cv=3) on GradientBoosting. Four models compared with RMSE/MAE/R²/MAPE. CV results reported.

**Ethics, bias, fairness analysis (Week 13):**
- **EU AI Act classification:** Limited-Risk AI system (chatbot must disclose AI nature → implemented in UI disclaimer and system prompt).
- **Potential biases:** UK market dataset may not reflect Swiss/European pricing. Luxury brands overrepresented in training data. Condition score biased by image quality/angle (CLIP domain gap).
- **Sociotechnical frame (Week 13):** AutoValue AI is advisory only — human-in-the-loop is recommended for high-value transactions. The system is a decision-support tool, not an autonomous price setter.
- **Amara's Law acknowledgment:** Short-term: AI estimates will reduce to simple price lookups; long-term: comprehensive multi-modal valuation could replace human appraisers.
- **Data origin transparency:** All third-party code/models use open licenses (CLIP: MIT via HuggingFace; UK Cars dataset: publicly available on Kaggle).
