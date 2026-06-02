from __future__ import annotations

import os
from typing import Optional

import pandas as pd

SYSTEM_PROMPT = """You are AutoValue AI, an expert Swiss used car price analyst.

Your task is to explain the estimated price of a used car clearly and concisely to the user.
The input includes structured car data extracted by our AI pipeline and comparable market listings.

Guidelines:
- Base your explanation solely on the data provided.
- Highlight the top 2-3 price factors (age, mileage, condition, make).
- Keep the explanation to 3-4 sentences.
- Answer in the user's language (German or English).
- Never claim to be a human. You are an AI assistant (EU AI Act transparency requirement)."""


def build_explanation_prompt(
    make: str,
    model_name: str,
    year: int,
    mileage_km: float,
    fuel_type: str,
    transmission: str,
    condition_score: float,
    condition_label: str,
    predicted_price: float,
    similar_cars: pd.DataFrame,
) -> str:
    if len(similar_cars) > 0:
        evidence_lines = "\n".join(
            f"- {row['make']} {row['model']} ({int(row['year'])}, "
            f"{int(row['mileage']):,} km): GBP {int(row['price']):,}"
            for _, row in similar_cars.iterrows()
        )
    else:
        evidence_lines = "No comparable listings found."

    return f"""Answer the following question: Why is this {make} {model_name} ({year}) worth approximately GBP {predicted_price:,.0f}?

Base your answer solely on the information given:

<car_data>
Make: {make} | Model: {model_name} | Year: {year}
Mileage: {mileage_km:,.0f} km | Fuel: {fuel_type} | Transmission: {transmission}
Condition Score: {condition_score:.2f}/1.0 ({condition_label})
Estimated Price: GBP {predicted_price:,.0f}
</car_data>

<similar_listings>
{evidence_lines}
</similar_listings>"""


def get_price_explanation(
    make: str,
    model_name: str,
    year: int,
    mileage_km: float,
    fuel_type: str,
    transmission: str,
    condition_score: float,
    condition_label: str,
    predicted_price: float,
    similar_cars: pd.DataFrame,
    api_key: Optional[str] = None,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    user_prompt = build_explanation_prompt(
        make, model_name, year, mileage_km, fuel_type, transmission,
        condition_score, condition_label, predicted_price, similar_cars,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content


def chat_with_autovalue(
    user_message: str,
    car_context: dict,
    history: list[dict],
    api_key: Optional[str] = None,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    system = f"""{SYSTEM_PROMPT}

Current car under discussion:
<car_context>
{car_context.get('make', '?')} {car_context.get('model', '?')} ({car_context.get('year', '?')})
Mileage: {car_context.get('mileage', '?'):,} km | Fuel: {car_context.get('fuel_type', '?')}
Condition: {car_context.get('condition_label', '?')} ({car_context.get('condition_score', 0):.2f})
ML Predicted Price: GBP {car_context.get('predicted_price', 0):,.0f}
</car_context>"""

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400,
        temperature=0.5,
    )
    return response.choices[0].message.content
