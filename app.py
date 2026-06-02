from __future__ import annotations

import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from src.cv_block import get_condition_display
from src.ml_block import get_similar_cars, predict_price
from src.nlp_block import chat_with_autovalue, get_price_explanation


FUEL_OPTIONS = ["Petrol", "Diesel", "Hybrid", "Electric"]
TRANSMISSION_OPTIONS = ["Manual", "Automatic"]


_car_context: dict = {}
_chat_history: list[dict] = []


def run_pipeline(
    car_image,
    make: str,
    model_name: str,
    year: int,
    mileage_km: float,
    fuel_type: str,
    transmission: str,
) -> tuple[str, str, str]:
    global _car_context, _chat_history
    _chat_history = []

    if car_image is not None:
        img = Image.fromarray(car_image) if not isinstance(car_image, Image.Image) else car_image
    else:
        img = None

    condition_score, condition_label, probs = get_condition_display(img)

    condition_text = (
        f"**Condition Score:** {condition_score:.2f} / 1.0  \n"
        f"**Label:** {condition_label}  \n\n"
        f"| Category | Probability |\n|---|---|\n"
        f"| Perfect | {probs[0]:.1%} |\n"
        f"| Minor damage | {probs[1]:.1%} |\n"
        f"| Moderate damage | {probs[2]:.1%} |\n"
        f"| Severe damage | {probs[3]:.1%} |"
    )

    try:
        predicted_price = predict_price(
            make=make,
            model_name=model_name,
            year=int(year),
            mileage_km=float(mileage_km),
            fuel_type=fuel_type,
            transmission=transmission,
            condition_score=condition_score,
        )
        price_text = (
            f"## GBP {predicted_price:,.0f}\n\n"
            f"*Based on make, model, year, mileage, fuel type, transmission, "
            f"and CV condition score.*"
        )
    except Exception as e:
        predicted_price = 0.0
        price_text = f"Model not loaded. Run training first.\n\n`{e}`"

    explanation_text = ""
    if predicted_price > 0:
        try:
            similar_cars = get_similar_cars(make, model_name, int(year), float(mileage_km))
            explanation_text = get_price_explanation(
                make=make,
                model_name=model_name,
                year=int(year),
                mileage_km=float(mileage_km),
                fuel_type=fuel_type,
                transmission=transmission,
                condition_score=condition_score,
                condition_label=condition_label,
                predicted_price=predicted_price,
                similar_cars=similar_cars,
            )
        except Exception as e:
            explanation_text = f"NLP error: {e}"
    else:
        explanation_text = "Run model training first."

    _car_context = {
        "make": make,
        "model": model_name,
        "year": int(year),
        "mileage": float(mileage_km),
        "fuel_type": fuel_type,
        "transmission": transmission,
        "condition_score": condition_score,
        "condition_label": condition_label,
        "predicted_price": predicted_price,
    }

    return condition_text, price_text, explanation_text


def respond_to_chat(
    user_message: str,
    chat_display: list,
) -> tuple[list, str]:
    global _chat_history

    if not _car_context:
        reply = "Please run the price estimation first so I have context about the car."
    else:
        try:
            reply = chat_with_autovalue(
                user_message=user_message,
                car_context=_car_context,
                history=_chat_history,
            )
            _chat_history.append({"role": "user", "content": user_message})
            _chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            reply = f"Error: {e}"

    chat_display = chat_display or []
    chat_display.append({"role": "user", "content": user_message})
    chat_display.append({"role": "assistant", "content": reply})
    return chat_display, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="AutoValue AI — Used Car Price Estimator",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# AutoValue AI\n"
            "**AI-powered used car price estimation** | Computer Vision + ML + NLP"
        )

        with gr.Tab("Price Estimator"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Car Photo")
                    car_image = gr.Image(label="Upload car photo (optional)", type="numpy")

                    gr.Markdown("### Car Details")
                    make_in = gr.Textbox(label="Make", placeholder="e.g. BMW", value="BMW")
                    model_in = gr.Textbox(label="Model", placeholder="e.g. 3 Series", value="3 Series")
                    year_in = gr.Slider(label="Year", minimum=2000, maximum=datetime.now().year, step=1, value=2019)
                    mileage_in = gr.Number(label="Mileage (km)", value=45000)
                    fuel_in = gr.Dropdown(label="Fuel Type", choices=FUEL_OPTIONS, value="Diesel")
                    trans_in = gr.Dropdown(label="Transmission", choices=TRANSMISSION_OPTIONS, value="Automatic")

                    estimate_btn = gr.Button("Estimate Price", variant="primary")


                with gr.Column(scale=1):
                    gr.Markdown("### Results")
                    condition_out = gr.Markdown(label="CV Condition Score")
                    price_out = gr.Markdown(label="ML Price Prediction")
                    explanation_out = gr.Markdown(label="NLP Explanation")

            estimate_btn.click(
                fn=run_pipeline,
                inputs=[car_image, make_in, model_in, year_in, mileage_in, fuel_in, trans_in],
                outputs=[condition_out, price_out, explanation_out],
            )

        with gr.Tab("Chat with AutoValue AI"):
            gr.Markdown(
                "Ask follow-up questions about the estimated price.\n"
                "Run the price estimator first to provide car context."
            )
            chatbot = gr.Chatbot(label="AutoValue AI Chatbot", height=400)
            with gr.Row():
                chat_input = gr.Textbox(
                    label="Your question",
                    placeholder="Why is the mileage affecting the price so much?",
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

            send_btn.click(
                fn=respond_to_chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                fn=respond_to_chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )

        with gr.Tab("About"):
            gr.Markdown("""
## AutoValue AI — System Overview

**Pipeline:** Car Photo → CV Condition Score → ML Price Prediction → NLP Explanation

| Block | Technology |
|---|---|
| Computer Vision | CLIP zero-shot (openai/clip-vit-base-patch32) |
| ML Prediction | GradientBoosting + RandomizedSearchCV |
| NLP Explanation | GPT-4o-mini + RAG from dataset |

**Dataset:** UK Used Cars (Kaggle) — 108,535 rows, 11 makes
**ML Results:** RMSE = GBP 5,451 | R² = 0.693 | MAPE = 18.4% | 5-fold CV RMSE = 5,555 ± 42

**EU AI Act Classification:** Limited-Risk AI System
Chatbot discloses AI nature. All estimates are advisory — not binding.

**Ethical considerations:**
- Prices are estimates only, based on UK market data
- Model may reflect dataset biases (UK market, limited EV data)
- Condition score depends on image quality and angle (CLIP limitations)
- Human review recommended for high-value transactions

*ZHAW Wirtschaftsinformatik, Modul KI-Anwendungen, FS2026*
""")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)
