import os
from datetime import date

import pandas as pd
import streamlit as st

from data_loader import format_location, load_weather_data, search_ukrainian_locations
from predict import predict_weather
from train_model import train_models


st.set_page_config(
    page_title="Rain Prediction Dashboard",
    page_icon="🌧️",
    layout="wide"
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(45, 212, 191, 0.16), transparent 34rem),
            linear-gradient(135deg, #0f172a 0%, #111827 48%, #18181b 100%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.4rem 0 0.4rem;
    }
    .hero h1 {
        font-size: 3rem;
        line-height: 1.05;
        margin-bottom: 0.35rem;
    }
    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 54rem;
    }
    .section-title {
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 750;
    }
    .hint {
        color: #94a3b8;
        font-size: 0.96rem;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 1rem 1rem 0.7rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(45, 212, 191, 0.45);
        background: linear-gradient(135deg, #14b8a6, #2563eb);
        color: white;
        font-weight: 700;
    }
    .stButton > button:hover {
        border-color: #67e8f9;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_percent(value):
    return f"{value * 100:.1f}%"


def show_weather_summary(df):
    rainy_days = int((df["precipitation_sum"] > 0).sum())
    total_days = len(df)
    avg_temp = ((df["temperature_max"] + df["temperature_min"]) / 2).mean()
    total_precipitation = df["precipitation_sum"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Днів у вибірці", total_days)
    col2.metric("Днів з опадами", rainy_days, f"{rainy_days / total_days * 100:.1f}%")
    col3.metric("Середня температура", f"{avg_temp:.1f} °C")
    col4.metric("Сума опадів", f"{total_precipitation:.1f} мм")


def show_weather_charts(df):
    chart_df = df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.set_index("date")
    chart_df["average_temperature"] = (
        chart_df["temperature_max"] + chart_df["temperature_min"]
    ) / 2

    temp_col, rain_col = st.columns(2)

    with temp_col:
        st.subheader("Температура")
        st.line_chart(
            chart_df[["temperature_max", "temperature_min", "average_temperature"]],
            use_container_width=True,
        )

    with rain_col:
        st.subheader("Опади")
        st.bar_chart(
            chart_df[["precipitation_sum"]],
            use_container_width=True,
        )


def show_model_results(results, feature_info=None):
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("f1", ascending=False)

    if feature_info is not None:
        st.subheader("Feature selection")
        st.info(
            "Фінальний набір ознак сформовано методом SelectKBest. "
            "Модель навчається тільки на ознаках з найвищими статистичними оцінками."
        )

        class_distribution = feature_info.get("class_distribution")
        if class_distribution is not None:
            st.subheader("Баланс класів")
            st.write(
                "Під час навчання використовується `class_weight='balanced'`, "
                "щоб зменшити вплив дисбалансу між днями з опадами і без опадів."
            )
            st.dataframe(
                class_distribution.rename("Частка, %").to_frame(),
                use_container_width=True,
            )

        imputation_values = feature_info.get("imputation_values")
        if imputation_values is not None:
            st.subheader("Попередня обробка даних")
            st.write(
                "Пропуски в ознаках заповнюються median-значеннями, "
                "розрахованими тільки на навчальній вибірці. "
                "`precipitation_sum` використовується лише для target і не входить до ознак."
            )
            st.dataframe(
                imputation_values.rename("Median для заповнення пропусків").to_frame(),
                use_container_width=True,
            )

        selected_features = feature_info["selected_features"]
        st.write("Відібрані ознаки:", ", ".join(selected_features))

        score_df = feature_info["feature_scores"].copy()
        score_df = score_df.rename(columns={
            "feature": "Ознака",
            "score": "Оцінка SelectKBest",
        })
        st.dataframe(
            score_df.style.format({"Оцінка SelectKBest": "{:.3f}"}),
            use_container_width=True,
        )

        chart_df = score_df.set_index("Ознака")
        st.bar_chart(chart_df[["Оцінка SelectKBest"]])

    st.dataframe(
        results_df.style.format({
            "accuracy": "{:.3f}",
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1": "{:.3f}",
        }).highlight_max(axis=0, color="#155e75"),
        use_container_width=True,
    )

    st.subheader("Порівняння моделей")
    st.bar_chart(results_df[["accuracy", "precision", "recall", "f1"]])

    best_name = results_df.index[0]
    best_f1 = results_df.iloc[0]["f1"]
    st.success(f"Найкраща модель за F1-score: {best_name} ({best_f1:.3f})")


def show_prediction(prediction_df):
    prediction_df = prediction_df.copy()
    prediction_df["date"] = pd.to_datetime(prediction_df["date"])
    prediction_df["result"] = prediction_df["prediction"].apply(
        lambda x: "Очікуються опади" if x == 1 else "Опадів не очікується"
    )
    prediction_df["probability_percent"] = (
        prediction_df["probability"] * 100
    ).round(1)

    rainy_forecast_days = int((prediction_df["prediction"] == 1).sum())
    max_probability = prediction_df["probability_percent"].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Днів прогнозу", len(prediction_df))
    col2.metric("Днів з імовірними опадами", rainy_forecast_days)
    col3.metric("Макс. ймовірність опадів", f"{max_probability:.1f}%")

    st.subheader("Прогноз на найближчі дні")
    table_df = prediction_df[[
        "date",
        "temperature_max",
        "temperature_min",
        "wind_speed",
        "result",
        "probability_percent",
    ]].rename(columns={
        "date": "Дата",
        "temperature_max": "Макс. температура",
        "temperature_min": "Мін. температура",
        "wind_speed": "Вітер",
        "result": "Результат",
        "probability_percent": "Ймовірність опадів, %",
    })

    st.dataframe(
        table_df.style.background_gradient(
            subset=["Ймовірність опадів, %"],
            cmap="Blues",
        ),
        use_container_width=True,
    )

    chart_df = prediction_df.set_index("date")[["probability_percent"]]
    st.subheader("Ймовірність опадів")
    st.line_chart(chart_df, use_container_width=True)


with st.sidebar:
    st.header("Параметри")
    city_query = st.text_input(
        "Пошук населеного пункту України",
        value="",
        placeholder="Почни вводити: Київ, Львів, Харків...",
    )
    try:
        location_options = search_ukrainian_locations(city_query)
    except Exception as error:
        location_options = []
        st.warning(f"Не вдалося виконати пошук: {error}")

    if location_options:
        selected_location = st.selectbox(
            "Оберіть населений пункт",
            location_options,
            format_func=format_location,
        )
        city = selected_location["name"]
        st.metric("Широта", f"{selected_location['latitude']:.4f}")
        st.metric("Довгота", f"{selected_location['longitude']:.4f}")
        if selected_location.get("admin_area"):
            st.caption(f"Область/регіон: {selected_location['admin_area']}")
    else:
        selected_location = None
        city = city_query.strip()
        st.info("Введи назву населеного пункту, щоб побачити варіанти.")

    start_date = st.date_input("Дата початку", value=date(2023, 1, 1))
    end_date = st.date_input("Дата кінця", value=date(2024, 12, 31))
    st.caption(
        "Можна вводити довільний населений пункт України. "
        "Для навчання краще брати історичний період від 6 місяців."
    )


st.markdown(
    """
    <div class="hero">
        <h1>Прогноз опадів</h1>
        <p>
            Інтерактивний ML-dashboard на основі погодних даних Open-Meteo:
            завантаження історії, навчання моделей, метрики якості та прогноз.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


if start_date >= end_date:
    st.warning("Дата початку має бути раніше за дату кінця.")

training_days = (end_date - start_date).days
if training_days < 180:
    st.warning(
        "Період даних менший за 180 днів. Для стабільнішого навчання моделей "
        "краще обрати хоча б 6-12 місяців історії."
    )


st.markdown('<div class="section-title">1. Дані</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hint">Завантаж історичні погодні дані для обраного міста.</div>',
    unsafe_allow_html=True,
)

if st.button("Завантажити дані", use_container_width=True):
    if start_date >= end_date:
        st.error("Вибери коректний період дат.")
    elif selected_location is None:
        st.error("Оберіть населений пункт зі списку.")
    else:
        try:
            with st.spinner("Завантажую дані з Open-Meteo..."):
                df = load_weather_data(
                    city,
                    str(start_date),
                    str(end_date),
                    location=selected_location,
                )
                st.session_state["weather_df"] = df
                st.session_state["city"] = city
                st.session_state["selected_location"] = selected_location
            location = df.attrs.get("location", {})
            location_label = ", ".join(
                item for item in [
                    location.get("name"),
                    location.get("admin_area"),
                    location.get("country"),
                ]
                if item
            )
            st.success(
                f"Дані успішно завантажені для: {location_label}. "
                "Файл збережено у weather_daily.csv"
            )
        except Exception as error:
            st.error(f"Не вдалося завантажити дані: {error}")


if "weather_df" in st.session_state:
    df = st.session_state["weather_df"]
elif os.path.exists("weather_daily.csv"):
    df = pd.read_csv("weather_daily.csv")
    st.session_state["weather_df"] = df
else:
    df = None


if df is not None:
    show_weather_summary(df)

    tab1, tab2 = st.tabs(["Графіки", "Таблиця"])
    with tab1:
        show_weather_charts(df)
    with tab2:
        st.dataframe(df, use_container_width=True)
else:
    st.info("Натисни кнопку завантаження, щоб отримати таблицю та графіки.")


st.markdown('<div class="section-title">2. Навчання моделей</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hint">Моделі прогнозують, чи будуть опади за температурою та вітром.</div>',
    unsafe_allow_html=True,
)

if st.button("Навчити моделі", use_container_width=True):
    if not os.path.exists("weather_daily.csv"):
        st.error("Спочатку завантаж дані.")
    else:
        try:
            with st.spinner("Навчаю Logistic Regression, Decision Tree та Random Forest..."):
                results, model, feature_info = train_models()
                st.session_state["model_results"] = results
                st.session_state["feature_info"] = feature_info
            st.success("Моделі навчені. Найкраща модель збережена у models/rain_model.pkl")
        except Exception as error:
            st.error(f"Не вдалося навчити моделі: {error}")


if "model_results" in st.session_state:
    show_model_results(
        st.session_state["model_results"],
        st.session_state.get("feature_info"),
    )


st.markdown('<div class="section-title">3. Прогноз</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hint">Отримай прогноз погоди на найближчі 5 днів і пропусти його через модель.</div>',
    unsafe_allow_html=True,
)

if st.button("Зробити прогноз", use_container_width=True):
    if not os.path.exists("models/rain_model.pkl"):
        st.error("Спочатку навчи моделі.")
    elif selected_location is None:
        st.error("Оберіть населений пункт зі списку.")
    else:
        try:
            with st.spinner("Розраховую прогноз..."):
                prediction_df = predict_weather(city, location=selected_location)
                st.session_state["prediction_df"] = prediction_df
            st.success("Прогноз готовий.")
        except Exception as error:
            st.error(f"Не вдалося зробити прогноз: {error}")


if "prediction_df" in st.session_state:
    show_prediction(st.session_state["prediction_df"])
