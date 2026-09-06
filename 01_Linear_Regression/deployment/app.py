import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Karachi Car Price Estimator",
    page_icon="🚗",
    layout="centered",
)

# ----------------------------------------------------------------------
# Load the trained model and everything Stage 4/5 computed, once per run
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    with open("meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta


model, scaler, meta = load_artifacts()

FEATURE_COLUMNS = meta["feature_columns"]
NUMERIC_COLS = meta["numeric_cols"]
KEEP_MANUFACTURERS = set(meta["keep_manufacturers"])
KEEP_VARIANTS = set(meta["keep_variants"])
MANUFACTURER_TO_VARIANTS = meta["manufacturer_to_variants"]
VARIANT_ENGINE_RANGE = meta.get("variant_engine_range", {})


def predict_price(year, mileage, engine, fuel_type, transmission, manufacturer, variant):
    # Guard against a manufacturer/variant pair that never appears together in
    # the real data (e.g. Mercedes + Corolla). The dropdowns already narrow
    # Variant down to a given Manufacturer's own list, so this should never
    # actually trigger through the UI, but it's kept as an explicit backend
    # check rather than trusting the frontend never to get out of sync.
    allowed_variants = MANUFACTURER_TO_VARIANTS.get(manufacturer, [])
    if variant not in allowed_variants:
        raise ValueError(
            f"'{manufacturer}' does not make a '{variant}'. Pick a variant that matches the manufacturer."
        )

    manufacturer_grouped = manufacturer if manufacturer in KEEP_MANUFACTURERS else "Other"
    variant_grouped = variant if variant in KEEP_VARIANTS else "Other"

    row = pd.DataFrame(
        [
            {
                "year": year,
                "mileage": mileage,
                "engine": engine,
                "fuelType": fuel_type,
                "transmission": transmission,
                "manufacturer_grouped": manufacturer_grouped,
                "variant_grouped": variant_grouped,
            }
        ]
    )

    row_encoded = pd.get_dummies(
        row, columns=["fuelType", "transmission", "manufacturer_grouped", "variant_grouped"]
    )
    row_encoded = row_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    row_encoded[NUMERIC_COLS] = scaler.transform(row_encoded[NUMERIC_COLS])

    log_price = model.predict(row_encoded)[0]
    return float(np.exp(log_price))


# ----------------------------------------------------------------------
# Styling: a plain, editorial listing-page look. No gradients, no glow.
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #F7F4EE;
        }

        .block-container {
            max-width: 720px;
            padding-top: 3rem;
            padding-bottom: 4rem;
        }

        #MainMenu, header, footer {
            visibility: hidden;
        }

        .hero-title {
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-size: 2.4rem;
            color: #23241F;
            margin-bottom: 0.2rem;
            letter-spacing: -0.01em;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #6B6A63;
            margin-bottom: 1.8rem;
            line-height: 1.5;
        }

        .stat-strip {
            display: flex;
            gap: 1.6rem;
            border-top: 1px solid #DEDACF;
            border-bottom: 1px solid #DEDACF;
            padding: 0.9rem 0;
            margin-bottom: 2rem;
        }

        .stat-item {
            font-size: 0.82rem;
            color: #6B6A63;
        }

        .stat-item b {
            color: #23241F;
            font-weight: 600;
        }

        .section-label {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8B6A4A;
            margin-top: 1.6rem;
            margin-bottom: 0.6rem;
        }

        div.st-key-car_form_card {
            background-color: #FFFFFF;
            border: 1px solid #E5E1D6 !important;
            border-radius: 10px;
            padding: 1.8rem 1.8rem 1.2rem 1.8rem;
        }

        .stSelectbox label, .stNumberInput label {
            font-size: 0.85rem !important;
            color: #4A4A44 !important;
            font-weight: 500 !important;
        }

        div[data-baseweb="select"] > div, .stNumberInput input {
            border-radius: 6px !important;
            border-color: #D8D3C6 !important;
        }

        .stButton button {
            background-color: #23241F;
            color: #F7F4EE;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1.4rem;
            font-weight: 500;
            width: 100%;
            transition: background-color 0.15s ease;
        }

        .stButton button:hover {
            background-color: #46473F;
            color: #F7F4EE;
        }

        .result-card {
            background-color: #23241F;
            border-radius: 10px;
            padding: 1.8rem 2rem;
            margin-top: 1.6rem;
            text-align: center;
        }

        .result-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #A8A69C;
            margin-bottom: 0.4rem;
        }

        .result-price {
            font-family: 'Fraunces', serif;
            font-size: 2.6rem;
            font-weight: 600;
            color: #F7F4EE;
        }

        .result-caption {
            font-size: 0.82rem;
            color: #A8A69C;
            margin-top: 0.5rem;
        }

        .footer-note {
            font-size: 0.78rem;
            color: #9A988E;
            text-align: center;
            margin-top: 2.5rem;
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="hero-title">Karachi Car Price Estimator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Enter a car\'s specs and get an estimated asking price, '
    "based on a multiple linear regression model trained on real Karachi marketplace listings.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="stat-strip">
        <div class="stat-item"><b>{meta['n_rows']:,}</b> listings trained on</div>
        <div class="stat-item"><b>{meta['test_r2']:.0%}</b> of price variance explained</div>
        <div class="stat-item"><b>{meta['year_min']}&ndash;{meta['year_max']}</b> model years covered</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Form
#
# Deliberately NOT using st.form here. Streamlit forms only re-run the
# script when the submit button is pressed, so a Manufacturer change
# inside a form would not refresh the Variant list until after the fact,
# by which point you could already have picked a variant that belongs to
# a different manufacturer (Mercedes + Corolla, for example). Using plain
# widgets means every change re-runs immediately, so Variant always
# reflects the manufacturer currently selected.
# ----------------------------------------------------------------------
with st.container(border=True, key="car_form_card"):
    st.markdown('<div class="section-label">Make &amp; Model</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        manufacturer = st.selectbox(
            "Manufacturer",
            meta["manufacturer_options"],
            index=meta["manufacturer_options"].index("Toyota") if "Toyota" in meta["manufacturer_options"] else 0,
            key="manufacturer",
        )
    with col2:
        variant_choices = MANUFACTURER_TO_VARIANTS.get(manufacturer, meta["variant_options"])
        default_variant_index = variant_choices.index("Corolla") if "Corolla" in variant_choices else 0
        variant = st.selectbox("Variant", variant_choices, index=default_variant_index, key="variant")

    st.markdown('<div class="section-label">Specifications</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        year = st.number_input(
            "Year", min_value=meta["year_min"], max_value=meta["year_max"], value=2019, step=1
        )

        engine_range = VARIANT_ENGINE_RANGE.get(variant)
        engine_default = engine_range["median"] if engine_range else meta["engine_median"]
        # Keying by variant means the default resets to that variant's own
        # typical engine size when you switch cars, rather than carrying
        # over whatever the previous variant's engine value happened to be.
        engine = st.number_input(
            "Engine size (cc)",
            min_value=600,
            max_value=6500,
            value=engine_default,
            step=100,
            key=f"engine_{variant}",
        )
        if engine_range:
            single_value = engine_range["min"] == engine_range["max"]
            if single_value:
                range_text = f"{engine_range['min']:,}cc"
                st.caption(f"Every {variant} in our data is {range_text}.")
            else:
                range_text = f"{engine_range['min']:,}–{engine_range['max']:,}cc"
                st.caption(f"Typical for {variant}: {range_text}")
            if not (engine_range["min"] <= engine <= engine_range["max"]):
                st.warning(
                    f"{engine:,}cc is outside what we've seen for {variant} "
                    f"({range_text} in our data). "
                    "The estimate below may be less reliable for this combination."
                )
    with col4:
        mileage = st.number_input(
            "Mileage (km)", min_value=0, max_value=500000, value=meta["mileage_median"], step=1000
        )
        petrol_index = meta["fueltype_options"].index("Petrol") if "Petrol" in meta["fueltype_options"] else 0
        fuel_type = st.selectbox("Fuel type", meta["fueltype_options"], index=petrol_index)

    transmission = st.selectbox("Transmission", meta["transmission_options"])

    submitted = st.button("Estimate price", use_container_width=True)

if submitted:
    try:
        price = predict_price(year, mileage, engine, fuel_type, transmission, manufacturer, variant)
    except ValueError as error:
        st.error(str(error))
    else:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated Listing Price</div>
                <div class="result-price">PKR {price:,.0f}</div>
                <div class="result-caption">Typical error on unseen listings: roughly &plusmn;PKR {int(price * 0.17):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer-note">
        Trained on real Karachi marketplace listings scraped December 2025. Estimates reflect
        seller asking prices, not confirmed sale prices, and should not be used as a substitute
        for an in person inspection.<br>
        Built by M S Umar &middot; github.com/msumar7426
    </div>
    """,
    unsafe_allow_html=True,
)
