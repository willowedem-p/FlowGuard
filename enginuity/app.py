"""
FlowGuard — Smart Water Quality Assessment and Treatment
Recommendation System
SDG 3 (Good Health) & SDG 6 (Clean Water)

The machine-learning model and engineering logic below (ml_prediction,
calculate_purity, classify_water, recommend_treatment, decision_fusion)
are taken directly from the project notebook and are UNCHANGED.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import io
import base64
import warnings
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="FlowGuard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "test_history.csv")
CSS_PATH = os.path.join(BASE_DIR, "style.css")
DATASET_PATH = os.path.join(BASE_DIR, "water_quality_dataset1.csv")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
BACKGROUND_IMAGE_PATH = os.path.join(ASSETS_DIR, "main_background.jpg")

HISTORY_COLUMNS = [
    "DateTime", "pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C",
    "ML_Prediction", "ML_Safe_Prob", "Purity_Index", "WQI_Class",
    "Final_Decision", "Confidence", "Treatment",
]

WHO_LIMITS = {
    "pH": (6.5, 8.5),
    "TDS_ppm": (0, 500),
    "Turbidity_NTU": (0, 5),
    "Temperature_C": (20, 30),
}

MODEL_LOAD_ERROR = None

# ============================================================================
# DATA LOADING AND CLEANING (notebook-aligned)
# ============================================================================
@st.cache_data
def load_cleaned_dataset():
    df = pd.read_csv(DATASET_PATH)
    df.drop_duplicates(inplace=True)

    if "Label" not in df.columns:
        raise ValueError("Dataset must contain a 'Label' column with Safe/Unsafe values.")

    df["Label"] = df["Label"].map({"Safe": 1, "Unsafe": 0})
    # NOTE: Turbidity_NTU is kept as raw float — the notebook does NOT binarise it
    # during training; binarisation only happens inside calculate_purity().

    safe = df[df["Label"] == 1]
    unsafe = df[df["Label"] == 0]
    safe_sample = safe.sample(n=3000, random_state=42)
    unsafe_sample = unsafe.sample(n=7000, random_state=42)

    cleaned = pd.concat([safe_sample, unsafe_sample])
    cleaned = cleaned.sample(frac=1, random_state=42).reset_index(drop=True)
    return cleaned


# ============================================================================
# LOAD MODEL & SCALER (cached)
# ============================================================================
@st.cache_resource
def load_model_and_scaler():
    df = load_cleaned_dataset()

    X = df[["pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C"]]
    y = df["Label"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler


try:
    model, scaler = load_model_and_scaler()
    if hasattr(model, "multi_class") and getattr(model, "multi_class") is None:
        warnings.warn("Patched LogisticRegression missing multi_class; setting to 'ovr'.")
        model.multi_class = "ovr"
    elif not hasattr(model, "multi_class"):
        warnings.warn("Patched LogisticRegression missing multi_class attribute; setting to 'ovr'.")
        setattr(model, "multi_class", "ovr")
except Exception as exc:
    model = None
    scaler = None
    MODEL_LOAD_ERROR = f"Unable to load ML resources: {exc}"


# ============================================================================
# REFERENCE DATASET (for EDA comparisons only — uses notebook-cleaned data)
# ============================================================================
@st.cache_data
def load_reference_dataset():
    return load_cleaned_dataset()


try:
    reference_df = load_reference_dataset()
except Exception:
    reference_df = pd.DataFrame()


# ============================================================================
# ORIGINAL NOTEBOOK LOGIC — DO NOT MODIFY
# ============================================================================

def ml_prediction(ph, tds, turbidity, temperature):
    new_sample = pd.DataFrame(
        [[ph, tds, turbidity, temperature]],
        columns=scaler.feature_names_in_
    )
    new_scaled = scaler.transform(new_sample)
    pred = model.predict(new_scaled)[0]
    prob = model.predict_proba(new_scaled)[0]
    return pred, prob


def calculate_purity(ph, tds, turbidity, temperature):
    # pH (WHO: 6.5–8.5)
    if 6.5 <= ph <= 8.5:
        ph_score = 100
    else:
        ph_score = max(0, 100 - abs(ph - 7.5) * 20)

    # TDS: full score up to 500 mg/L, then degrades above 500
    if tds <= 500:
        tds_score = 100
    else:
        tds_score = max(0, 100 - ((tds - 500) / 500) * 100)

    # Turbidity: binary — 0 NTU = clear (100), anything else = turbid (0)
    if turbidity == 0:
        turb_score = 100
    else:
        turb_score = 0

    # Temperature (comfort/aesthetic, not a WHO health limit)
    if 20 <= temperature <= 30:
        temp_score = 100
    else:
        temp_score = max(0, 100 - abs(temperature - 25) * 5)

    purity = (
        0.30 * ph_score +
        0.30 * tds_score +
        0.30 * turb_score +
        0.10 * temp_score
    )
    return round(purity, 2)


def classify_water(purity):
    if purity >= 90:
        return "Excellent"
    elif purity >= 75:
        return "Good"
    elif purity >= 50:
        return "Fair"
    elif purity >= 25:
        return "Poor"
    else:
        return "Very Poor"


def recommend_treatment(ph, tds, turbidity, purity):
    treatment = []

    if turbidity > 5:
        treatment.append("Coagulation + Filtration")
    if tds > 500:
        treatment.append("Reverse Osmosis / Ion Exchange")
    if ph < 6.5:
        treatment.append("Lime dosing (raise pH)")
    elif ph > 8.5:
        treatment.append("Acid neutralization")

    if purity >= 90:
        treatment.append("Disinfection only (UV/Chlorine)")
    elif purity >= 75:
        treatment.append("Filtration + Disinfection")
    elif purity >= 50:
        treatment.append("Advanced filtration required")
    else:
        treatment.append("Full multi-stage treatment system")

    return treatment


def decision_fusion(ml_pred, ml_prob, purity):
    ml_safe_prob = ml_prob[1]

    if ml_pred == 1 and purity >= 75:
        decision = "SAFE (High Confidence)"
        agreement = 1
    elif ml_pred == 0 and purity < 50:
        decision = "UNSAFE (High Confidence)"
        agreement = 1
    else:
        decision = "UNCERTAIN - Further Testing Required"
        agreement = 0

    confidence = (ml_safe_prob * 0.5) + (purity / 100 * 0.5)
    return decision, confidence, agreement


# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) — new analysis code, does NOT touch or
# alter any of the original notebook functions above.
# ============================================================================

def _score_breakdown(ph, tds, turbidity, temperature):
    """Independent recomputation of the WQI sub-scores, for display only."""
    ph_score = 100 if 6.5 <= ph <= 8.5 else max(0, 100 - abs(7.5 - ph) * 20)
    tds_score = max(0, 100 - (tds / 500) * 100)
    turb_score = max(0, 100 - (turbidity / 5) * 100)
    temp_score = 100 if 20 <= temperature <= 30 else max(0, 100 - abs(25 - temperature) * 5)
    return {
        "pH": ph_score * 0.30,
        "TDS": tds_score * 0.30,
        "Turbidity": turb_score * 0.30,
        "Temperature": temp_score * 0.10,
    }


def _percentile_in_dataset(df, column, value):
    series = df[column].dropna()
    return (series < value).mean() * 100


# Chart palette matched to the app's watercolor-ocean theme
CHART_ACCENT = "#2eb8c9"
CHART_HIST = "#10638a"
CHART_MARK = "#ff6b5e"
CHART_SAFE_BAND = "#2ecc71"
CHART_TEXT = "#e4f4f8"


def _style_axes_for_dark(ax, fig, white_bg):
    if white_bg:
        return
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors=CHART_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#3d6a80")
    ax.title.set_color("#ffffff")
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)


def _build_distribution_fig(ph, tds, turbidity, temperature, white_bg=False):
    params = [
        ("pH", ph, "pH"),
        ("TDS_ppm", tds, "TDS (ppm)"),
        ("Turbidity_NTU", turbidity, "Turbidity (NTU)"),
        ("Temperature_C", temperature, "Temperature (°C)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    if white_bg:
        fig.patch.set_facecolor("white")

    for ax, (col, val, label) in zip(axes.flatten(), params):
        _style_axes_for_dark(ax, fig, white_bg)
        if white_bg:
            ax.set_facecolor("white")
        if reference_df is not None and not reference_df.empty and col in reference_df.columns:
            ax.hist(reference_df[col], bins=40, color=CHART_HIST, edgecolor="none", alpha=0.85)
        ax.axvline(val, color=CHART_MARK, linewidth=2, label=f"Your reading: {val}")
        low, high = WHO_LIMITS[col]
        ax.axvspan(low, high, color=CHART_SAFE_BAND, alpha=0.15)
        title_color = "#023e58" if white_bg else "#ffffff"
        ax.set_title(label, fontsize=11, color=title_color)
        legend = ax.legend(fontsize=8, loc="upper right")
        if not white_bg:
            legend.get_frame().set_alpha(0)
            for text in legend.get_texts():
                text.set_color(CHART_TEXT)
    plt.tight_layout()
    return fig, params


def _build_score_fig(ph, tds, turbidity, temperature, white_bg=False):
    breakdown = _score_breakdown(ph, tds, turbidity, temperature)
    fig, ax = plt.subplots(figsize=(8, 4))
    if white_bg:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
    _style_axes_for_dark(ax, fig, white_bg)

    names = list(breakdown.keys())
    values = list(breakdown.values())
    colors = ["#10638a", "#16829f", "#2eb8c9", "#4fd8ec"]
    bars = ax.bar(names, values, color=colors)

    label_color = "#023e58" if white_bg else CHART_TEXT
    title_color = "#023e58" if white_bg else "#ffffff"
    ax.set_ylabel("Contribution to Purity Index (points)", color=label_color)
    ax.set_title("What is driving this reading's purity score", color=title_color)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}",
                 ha="center", fontsize=9, color=label_color)
    ax.set_ylim(0, max(35, max(values) + 5))
    plt.tight_layout()
    return fig, breakdown


def _build_trend_fig(hist_df_sorted, white_bg=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    if white_bg:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
    _style_axes_for_dark(ax, fig, white_bg)

    ax.plot(hist_df_sorted["DateTime"], hist_df_sorted["Purity_Index"],
             marker="o", color=CHART_ACCENT, label="Purity Index (%)")
    ax.plot(hist_df_sorted["DateTime"], hist_df_sorted["Confidence"],
             marker="o", color=CHART_MARK, label="Confidence (%)")

    label_color = "#023e58" if white_bg else CHART_TEXT
    title_color = "#023e58" if white_bg else "#ffffff"
    ax.set_ylabel("%", color=label_color)
    ax.set_title("Purity Index & Confidence across all tests", color=title_color)
    legend = ax.legend(fontsize=9)
    if not white_bg:
        legend.get_frame().set_alpha(0)
        for text in legend.get_texts():
            text.set_color(CHART_TEXT)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    return fig


def _fig_to_base64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    buf.close()
    return encoded


def build_eda_print_assets(ph, tds, turbidity, temperature, purity):
    """Builds white-background PNGs (base64) of all EDA charts for the printable report."""
    assets = {}

    fig1, params = _build_distribution_fig(ph, tds, turbidity, temperature, white_bg=True)
    assets["distribution"] = _fig_to_base64_png(fig1)
    plt.close(fig1)

    fig2, breakdown = _build_score_fig(ph, tds, turbidity, temperature, white_bg=True)
    assets["score"] = _fig_to_base64_png(fig2)
    plt.close(fig2)
    assets["breakdown"] = breakdown

    hist_df = load_history()
    if len(hist_df) >= 2:
        hist_df_sorted = hist_df.copy()
        hist_df_sorted["DateTime"] = pd.to_datetime(hist_df_sorted["DateTime"])
        hist_df_sorted = hist_df_sorted.sort_values("DateTime")
        fig3 = _build_trend_fig(hist_df_sorted, white_bg=True)
        assets["trend"] = _fig_to_base64_png(fig3)
        plt.close(fig3)
        assets["avg_purity"] = hist_df_sorted["Purity_Index"].mean()
        assets["num_tests"] = len(hist_df_sorted)
    else:
        assets["trend"] = None

    who_rows = []
    for col, val, label in params:
        low, high = WHO_LIMITS[col]
        meets = low <= val <= high
        who_rows.append({
            "Parameter": label, "Reading": val,
            "WHO Range": f"{low} – {high}",
            "Status": "Within range" if meets else "Outside range",
        })
    assets["who_rows"] = who_rows
    assets["percentiles"] = {
        label: _percentile_in_dataset(reference_df, col, val)
        for col, val, label in params
    } if reference_df is not None and not reference_df.empty else {}

    return assets


def render_eda(ph, tds, turbidity, temperature, purity):
    if reference_df is None or reference_df.empty:
        st.warning("Reference dataset is unavailable. Exploratory analysis is disabled.")
        return

    st.markdown("<h4>📊 Exploratory Data Analysis</h4>", unsafe_allow_html=True)
    st.markdown(
        "<p>How this reading compares to the reference dataset and what is driving its score.</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribution Comparison", "Score Breakdown", "Trend Over Time", "WHO Compliance"]
    )

    with tab1:
        fig, params = _build_distribution_fig(ph, tds, turbidity, temperature, white_bg=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("**Percentile within reference dataset:**")
        cols = st.columns(4)
        for c, (col, val, label) in zip(cols, params):
            pct = _percentile_in_dataset(reference_df, col, val)
            c.metric(label, f"{val}", f"{pct:.0f}th pct")

    with tab2:
        fig2, breakdown = _build_score_fig(ph, tds, turbidity, temperature, white_bg=False)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        weakest = min(breakdown, key=breakdown.get)
        st.info(
            f"**Weakest contributing factor:** {weakest} — improving this parameter "
            "would raise the overall purity index the most."
        )

    with tab3:
        hist_df = load_history()
        if len(hist_df) < 2:
            st.info("Run a few more tests to start seeing trends over time.")
        else:
            hist_df_sorted = hist_df.copy()
            hist_df_sorted["DateTime"] = pd.to_datetime(hist_df_sorted["DateTime"])
            hist_df_sorted = hist_df_sorted.sort_values("DateTime")

            fig3 = _build_trend_fig(hist_df_sorted, white_bg=False)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

            avg_purity = hist_df_sorted["Purity_Index"].mean()
            delta = purity - avg_purity
            direction = "above" if delta >= 0 else "below"
            st.info(
                f"This reading's purity index ({purity:.1f}%) is **{abs(delta):.1f} points "
                f"{direction}** the historical average ({avg_purity:.1f}%) across "
                f"{len(hist_df_sorted)} tests."
            )

    with tab4:
        rows = []
        for col, val, label in params:
            low, high = WHO_LIMITS[col]
            meets = low <= val <= high
            rows.append({
                "Parameter": label,
                "Reading": val,
                "WHO Range": f"{low} – {high}",
                "Status": "Within acceptable range" if meets else "Outside acceptable range",
                "Note": "Good" if meets else "Review recommended",
                "Badge": "badge-safe" if meets else "badge-unsafe",
            })

        table_rows = "".join(
            f"<tr><td data-label='Parameter'>{r['Parameter']}</td>"
            f"<td data-label='Reading'>{r['Reading']:g}</td>"
            f"<td data-label='WHO Range'>{r['WHO Range']}</td>"
            f"<td data-label='Status'><span class='badge {r['Badge']}'>{r['Status']}</span></td>"
            f"<td data-label='Note'>{r['Note']}</td></tr>"
            for r in rows
        )

        card_rows = "".join(
            f"<div class='who-card'><div class='who-card-row'><span class='who-card-label'>Parameter</span><span class='who-card-value'>{r['Parameter']}</span></div>"
            f"<div class='who-card-row'><span class='who-card-label'>Reading</span><span class='who-card-value'>{r['Reading']:g}</span></div>"
            f"<div class='who-card-row'><span class='who-card-label'>WHO Range</span><span class='who-card-value'>{r['WHO Range']}</span></div>"
            f"<div class='who-card-row'><span class='who-card-label'>Status</span><span class='who-card-value'><span class='badge {r['Badge']}'>{r['Status']}</span></span></div>"
            f"<div class='who-card-row'><span class='who-card-label'>Note</span><span class='who-card-value'>{r['Note']}</span></div></div>"
            for r in rows
        )

        st.markdown(
            "<div class='who-table-container'><table class='who-table'><thead><tr><th>Parameter</th><th>Reading</th>"
            "<th>WHO Range</th><th>Status</th><th>Note</th></tr></thead>"
            f"<tbody>{table_rows}</tbody></table></div>"
            f"<div class='who-card-list'>{card_rows}</div>",
            unsafe_allow_html=True,
        )


# ============================================================================
# HISTORY (CSV persistence)
# ============================================================================

def init_history_file():
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HISTORY_COLUMNS)


def save_result(row: dict):
    init_history_file()
    with open(HISTORY_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row[c] for c in HISTORY_COLUMNS])


def load_history() -> pd.DataFrame:
    init_history_file()
    try:
        return pd.read_csv(HISTORY_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_history(df: pd.DataFrame):
    init_history_file()
    df.to_csv(HISTORY_PATH, index=False)


# ============================================================================
# STYLING — single source of truth is style.css (watercolor-ocean theme)
# ============================================================================

def load_css():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

if MODEL_LOAD_ERROR:
    st.error(MODEL_LOAD_ERROR)
    st.stop()



# Add this near your file paths setup in app.py
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ----------------------------------------------------------------------------
# Hero / title-card image (your own tap-pour photo — assets/img_tap_pour.jpg)
# ----------------------------------------------------------------------------


@st.cache_data
def load_hero_image_b64():
    path = os.path.join(ASSETS_DIR, "img_tap_pour.jpg")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


_hero_img_b64 = load_hero_image_b64()

if _hero_img_b64:
    st.markdown(f"""
    <style>
    .hero {{
        background-image:
            linear-gradient(135deg, rgba(6, 20, 34, 0.82) 0%, rgba(10, 60, 90, 0.58) 55%, rgba(46, 184, 201, 0.35) 100%),
            url("data:image/jpg;base64,{_hero_img_b64}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Full-page background image (assets/bg_watercolor_swirl.jpg) — replaces the
# CSS-recreated gradient with your actual photo. A dark overlay is layered
# over it so text stays readable; falls back to the CSS gradient in
# style.css if this file is missing.
# ----------------------------------------------------------------------------
@st.cache_data
def load_bg_image_b64():
    path = BACKGROUND_IMAGE_PATH
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


_bg_img_b64 = load_bg_image_b64()

if _bg_img_b64:
    st.markdown(
        f"""
    <style>
    html, body {{
        min-height: 100%;
        background-image: url("data:image/jpg;base64,{_bg_img_b64}");
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottomBlockContainer"] {{
        background: transparent !important;
        background-color: transparent !important;
    }}

    .block-container {{
        position: relative;
        z-index: 1;
    }}
    </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# SESSION STATE / ROUTING (page persists across browser refresh via URL)
# ============================================================================

_VALID_PAGES = {
    "landing", "dashboard_home", "new_test", "result",
    "history", "record_detail", "recommendations",
}

if "page" not in st.session_state:
    _requested_page = st.query_params.get("page", "landing")
    st.session_state.page = _requested_page if _requested_page in _VALID_PAGES else "landing"
if "selected_record" not in st.session_state:
    st.session_state.selected_record = None
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# A refresh restores the page from the URL, but in-memory data (the latest
# result / selected history record) is lost. Fall back to a sensible page
# instead of showing an empty "no result" screen.
if st.session_state.page == "result" and st.session_state.latest_result is None:
    st.session_state.page = "dashboard_home"
    st.query_params["page"] = "dashboard_home"
if st.session_state.page == "record_detail" and st.session_state.selected_record is None:
    st.session_state.page = "history"
    st.query_params["page"] = "history"

# (history actions handled via Streamlit controls; no experimental query-param hacks)


def go_to(page_name):
    st.session_state.page = page_name
    st.query_params["page"] = page_name


def scroll_to_top():
    components.html(
        "<script>window.scrollTo({top:0,behavior:'smooth'});</script>", height=1
    )


# ============================================================================
# DECISION COLOR HELPERS
# ============================================================================

def decision_badge_class(decision: str) -> str:
    decision = decision or ""
    if decision.startswith("SAFE"):
        return "badge-safe"
    elif "UNSAFE" in decision:
        return "badge-unsafe"
    return "badge-uncertain"


def wqi_badge_class(wqi_class: str) -> str:
    mapping = {
        "Excellent": "badge-safe",
        "Good": "badge-safe",
        "Fair": "badge-uncertain",
        "Poor": "badge-unsafe",
        "Very Poor": "badge-unsafe",
    }
    return mapping.get(wqi_class, "badge-uncertain")


# ============================================================================
# PAGE: LANDING / FRONT PAGE
# ============================================================================

def render_landing():
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Welcome to FlowGuard</h1>
        <p class="hero-subtitle">A Smart Water Quality Assessment &amp; Treatment Recommendation System —
        an AI-powered decision support tool for safe drinking water</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="detail-card">
        <h3>About This Project</h3>
        <p>FlowGuard evaluates the physical and basic physicochemical quality of water using
        <strong>pH</strong>, <strong>turbidity</strong>, <strong>TDS</strong>, and <strong>temperature</strong>.
        The results provide an indication of water quality but do not replace comprehensive
        laboratory testing required to confirm microbiological and chemical safety for drinking.</p>        
        <h4>Typical sources this system can assess</h4>
        <ul>
            <li>Tap water</li>
            <li>Borehole (ground) water</li>
            <li>Well water</li>
            <li>River or surface water</li>
            <li>Rainwater</li>
            <li>Reservoir or storage tank water</li>
        </ul>
        <p>This tool helps identify samples that may require further treatment or referral to laboratory testing. Use FlowGuard as a screening and monitoring aid — not a substitute for formal certification.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider thin'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">🎯</div>
            <h3>Our Purpose and Goals</h3>
            <p>Provide awareness-building education about safe water, especially in
            rural and underserved communities, and make reliable water quality testing
            more accessible where lab services are limited.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">🧠</div>
            <h3>Our Approach</h3>
            <p>A trained Machine Learning model is fused with an engineering-based
            Water Quality Index (WQI) to classify water as Safe, Uncertain, or Unsafe,
            and to recommend an appropriate treatment plan.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">🌍</div>
            <h3>Our Impact</h3>
            <p>Help communities living with limited resources protect health by enabling
            practical on-site water assessment, reducing the need for distant testing
            infrastructure and empowering safer water choices.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="detail-card">
            <h3>How It Works</h3>
            <ol>
                <li>Water sensor readings are collected (pH, TDS, Turbidity, Temperature)</li>
                <li>A trained Logistic Regression model predicts Safe / Unsafe</li>
                <li>An engineering Water Quality Index (WQI) calculates a purity score</li>
                <li>The two results are fused into one final, confidence-scored decision</li>
                <li>A tailored treatment plan is generated automatically</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="detail-card">
            <h3>Parameters Measured</h3>
            <table class="param-table">
                <tr><td><b>pH</b></td><td>Acidity / alkalinity of water</td></tr>
                <tr><td><b>TDS (ppm)</b></td><td>Total Dissolved Solids — dissolved salts/minerals/metals</td></tr>
                <tr><td><b>Turbidity (NTU)</b></td><td>Water clarity / cloudiness</td></tr>
                <tr><td><b>Temperature (°C)</b></td><td>Water temperature</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="detail-card">
        <h3>Hardware &amp; Data Collection</h3>
        <p>Readings are designed to be collected from a low-cost Arduino-based sensor rig —
        a glass pH probe, an optical/IR turbidity sensor, a waterproof DS18B20 temperature
        probe, and a TDS module — a sensor combination chosen for being comparitively much more available and accessible in Ghana, while still giving a reliable water-quality signal.
        These four readings feed directly into the assessment engine below.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Begin Analysis →", key="begin_btn", use_container_width=True):
            go_to("dashboard_home")
            st.rerun()


# ============================================================================
# NAV SHELL
# ============================================================================

def render_nav():
    # Title at left, navigation buttons at right. Use Streamlit buttons so
    # navigation updates session state reliably across Streamlit versions.
    left, *buttons = st.columns([2, 1, 1, 1, 1])
    with left:
        st.markdown("<div class='navbar'><span class='navbar-title'>🌊 FlowGuard</span></div>", unsafe_allow_html=True)

    # Map label -> page_name
    nav_map = [
        ("Home", "landing", "nav_home"),
        ("Dashboard", "dashboard_home", "nav_dashboard"),
        ("New Test", "new_test", "nav_newtest"),
        ("Previous Tests", "history", "nav_history"),
        ("Recommendations", "recommendations", "nav_recs"),
    ]

    for col, (label, page, key) in zip(buttons, nav_map):
        with col:
            if st.button(label, key=key, use_container_width=True):
                go_to(page)
                st.experimental_rerun()


# ============================================================================
# PAGE: DASHBOARD HOME
# ============================================================================

def render_dashboard_home():
    df = load_history()
    total_tests = len(df)
    safe_count = len(df[df["Final_Decision"].astype(str).str.contains(r"SAFE \(High", regex=True)]) if total_tests else 0
    unsafe_count = len(df[df["Final_Decision"].astype(str).str.contains("UNSAFE", regex=True)]) if total_tests else 0
    uncertain_count = total_tests - safe_count - unsafe_count

    st.markdown("<h3>Overview</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='stat-card'><div class='stat-num'>{total_tests}</div><div class='stat-label'>Total Tests</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stat-card stat-safe'><div class='stat-num'>{safe_count}</div><div class='stat-label'>Safe</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stat-card stat-uncertain'><div class='stat-num'>{uncertain_count}</div><div class='stat-label'>Uncertain</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='stat-card stat-unsafe'><div class='stat-num'>{unsafe_count}</div><div class='stat-label'>Unsafe</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3>Recent Tests</h3>", unsafe_allow_html=True)

    if total_tests == 0:
        st.info("No tests have been run yet. Click **New Test** above to run your first water quality assessment.")
    else:
        # FIX: this used to be nested inside `if total_tests == 0`, so recent
        # tests never rendered once there was real history. Now it correctly
        # runs whenever tests exist.
        recent = df.tail(5).iloc[::-1]
        for _, row in recent.iterrows():
            badge = decision_badge_class(str(row["Final_Decision"]))
            st.markdown(f"""
            <div class="record-row">
                <div class="record-main">
                    <span class="record-date">{row['DateTime']}</span>
                    <span class="badge {badge}">{row['Final_Decision']}</span>
                </div>
                <div class="record-sub">pH {row['pH']} · TDS {row['TDS_ppm']} ppm · Turbidity {row['Turbidity_NTU']} NTU · {row['Temperature_C']}°C</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<p style='font-size:13px;color:#9fc9d9;margin-top:12px;'>Note: This dashboard shows indicators based on pH, TDS, turbidity, and temperature. These indicators are screening-level and do not replace laboratory testing for drinking-water safety.</p>", unsafe_allow_html=True)


# ============================================================================
# PAGE: NEW TEST
# ============================================================================

def render_new_test():
    st.markdown("<h3>Run a New Water Quality Test</h3>", unsafe_allow_html=True)
    st.markdown("<p>Enter sensor readings below to assess water safety.</p>", unsafe_allow_html=True)

    # ensure session state keys exist so +/- buttons can modify them
    if "ph_input" not in st.session_state:
        st.session_state["ph_input"] = 7.0
    if "tds_input" not in st.session_state:
        st.session_state["tds_input"] = 150.0
    if "turbidity_input" not in st.session_state:
        st.session_state["turbidity_input"] = 1.0
    if "temp_input" not in st.session_state:
        st.session_state["temp_input"] = 25.0

    # Compact, responsive input layout: four inputs in a single row on wide screens,
    # stacking naturally on mobile. Wrapped in a styled card for visual separation.
    with st.form("test_form"):
        col1, col2 = st.columns(2)
        with col1:
            ph_col, ph_btn_col = st.columns([9, 1])
            with ph_col:
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=st.session_state["ph_input"], step=0.01, format="%.2f", key="ph_input")
            with ph_btn_col:
                if st.button("+", key="ph_plus"):
                    st.session_state["ph_input"] = round(min(st.session_state["ph_input"] + 0.01, 14.0), 2)
                    st.experimental_rerun()
                if st.button("-", key="ph_minus"):
                    st.session_state["ph_input"] = round(max(st.session_state["ph_input"] - 0.01, 0.0), 2)
                    st.experimental_rerun()

            tds_col, tds_btn_col = st.columns([9, 1])
            with tds_col:
                tds = st.number_input("TDS (ppm)", min_value=0.0, max_value=5000.0, value=st.session_state["tds_input"], step=0.1, format="%.1f", key="tds_input")
            with tds_btn_col:
                if st.button("+", key="tds_plus"):
                    st.session_state["tds_input"] = round(min(st.session_state["tds_input"] + 0.1, 5000.0), 1)
                    st.experimental_rerun()
                if st.button("-", key="tds_minus"):
                    st.session_state["tds_input"] = round(max(st.session_state["tds_input"] - 0.1, 0.0), 1)
                    st.experimental_rerun()

        with col2:
            turb_col, turb_btn_col = st.columns([9, 1])
            with turb_col:
                turbidity = st.number_input(
                    "Turbidity (NTU)",
                    min_value=0.0,
                    max_value=200.0,
                    value=st.session_state["turbidity_input"],
                    step=0.01,
                    format="%.2f",
                    help="Raw turbidity reading from sensor. WHO limit: 5 NTU. Enter 0 for perfectly clear water.",
                    key="turbidity_input",
                )
            with turb_btn_col:
                if st.button("+", key="turb_plus"):
                    st.session_state["turbidity_input"] = round(min(st.session_state["turbidity_input"] + 0.01, 200.0), 2)
                    st.experimental_rerun()
                if st.button("-", key="turb_minus"):
                    st.session_state["turbidity_input"] = round(max(st.session_state["turbidity_input"] - 0.01, 0.0), 2)
                    st.experimental_rerun()

            temp_col, temp_btn_col = st.columns([9, 1])
            with temp_col:
                temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=st.session_state["temp_input"], step=0.01, format="%.2f", key="temp_input")
            with temp_btn_col:
                if st.button("+", key="temp_plus"):
                    st.session_state["temp_input"] = round(min(st.session_state["temp_input"] + 0.01, 60.0), 2)
                    st.experimental_rerun()
                if st.button("-", key="temp_minus"):
                    st.session_state["temp_input"] = round(max(st.session_state["temp_input"] - 0.01, -10.0), 2)
                    st.experimental_rerun()

        submitted = st.form_submit_button("Run Assessment", use_container_width=True)

    if submitted:
        with st.spinner("Assessing water quality with the ML model and WQI..."):
            ml_pred, ml_prob = ml_prediction(ph, tds, turbidity, temperature)
            purity = calculate_purity(ph, tds, turbidity, temperature)
            classification = classify_water(purity)
            treatment = recommend_treatment(ph, tds, turbidity, purity)
            final_decision, confidence, agreement = decision_fusion(ml_pred, ml_prob, purity)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            save_result({
                "DateTime": now_str,
                "pH": ph,
                "TDS_ppm": tds,
                "Turbidity_NTU": turbidity,
                "Temperature_C": temperature,
                "ML_Prediction": "SAFE" if ml_pred == 1 else "UNSAFE",
                "ML_Safe_Prob": round(ml_prob[1] * 100, 2),
                "Purity_Index": round(purity, 2),
                "WQI_Class": classification,
                "Final_Decision": final_decision,
                "Confidence": round(confidence * 100, 2),
                "Treatment": "; ".join(treatment),
            })

        st.session_state.latest_result = {
            "DateTime": now_str,
            "pH": ph,
            "TDS_ppm": tds,
            "Turbidity_NTU": turbidity,
            "Temperature_C": temperature,
            "ML_Prediction": "SAFE" if ml_pred == 1 else "UNSAFE",
            "ML_Safe_Prob": round(ml_prob[1] * 100, 2),
            "Purity_Index": round(purity, 2),
            "WQI_Class": classification,
            "Final_Decision": final_decision,
            "Confidence": round(confidence * 100, 2),
            "Treatment": treatment,
        }
        go_to("result")
        st.rerun()


# ============================================================================
# RESULT BLOCK (shared by New Test, History detail, Recommendations)
# ============================================================================

def render_result_block(dt_str, ph, tds, turbidity, temperature,
                         ml_pred, ml_prob, purity, classification,
                         final_decision, confidence, treatment):

    decision_class = decision_badge_class(final_decision)
    wqi_class = wqi_badge_class(classification)

    badge_colors = {
        "badge-safe": ("rgba(46, 204, 113, 0.20)", "#4be08a"),
        "badge-unsafe": ("rgba(231, 76, 60, 0.20)", "#ff6b5e"),
        "badge-uncertain": ("rgba(241, 196, 15, 0.20)", "#ffcf4d"),
    }
    decision_bg, decision_fg = badge_colors.get(decision_class, badge_colors["badge-uncertain"])
    wqi_bg, wqi_fg = badge_colors.get(wqi_class, badge_colors["badge-uncertain"])

    def badge_html(text, bg, fg, big=False):
        pad = "8px 18px" if big else "5px 14px"
        fs = "15px" if big else "12px"
        return (f'<span style="display:inline-block;padding:{pad};border-radius:24px;'
                f'font-size:{fs};font-weight:700;background:{bg};color:{fg} !important;'
                f'white-space:nowrap;">{text}</span>')

    widget_html = f"""
    <div class='result-card'>
      <div class='result-header'>
        <div>
          <h3>Water Quality Test Result</h3>
          <div class='result-date'>{dt_str}</div>
        </div>
        <div>{badge_html(final_decision, decision_bg, decision_fg, big=True)}</div>
      </div>

      <div class='result-grid'>
        <div class='metric-box'><div class='metric-label'>pH</div><div class='metric-value'>{ph}</div></div>
        <div class='metric-box'><div class='metric-label'>TDS</div><div class='metric-value'>{tds} ppm</div></div>
        <div class='metric-box'><div class='metric-label'>Turbidity</div><div class='metric-value'>{turbidity} NTU</div></div>
        <div class='metric-box'><div class='metric-label'>Temperature</div><div class='metric-value'>{temperature} °C</div></div>
      </div>

      <div class='result-grid two-col'>
        <div class='widget-card'>
          <h4>Machine Learning</h4>
          <p><strong>Prediction:</strong> {'SAFE' if ml_pred == 1 else 'UNSAFE'}</p>
          <p><strong>Safe Probability:</strong> {ml_prob[1] * 100:.2f}%</p>
        </div>
        <div class='widget-card'>
          <h4>Engineering WQI</h4>
          <p><strong>Purity Index:</strong> {purity:.2f}%</p>
          <p><strong>Class:</strong> {badge_html(classification, wqi_bg, wqi_fg)}</p>
        </div>
      </div>

      <div class='result-grid two-col'>
        <div class='widget-card'>
          <h4>Final AI Decision</h4>
          <p><strong>Decision:</strong> {final_decision}</p>
          <p><strong>Confidence Score:</strong> {confidence * 100:.2f}%</p>
        </div>
        <div class='widget-card'>
          <h4>Recommendations</h4>
          <p>The following treatment recommendations are suggested based on this water quality assessment.</p>
          <ul class='treatment-list'>
            {''.join(f'<li>{t}</li>' for t in treatment)}
          </ul>
        </div>
      </div>
    </div>
    """

    st.markdown(widget_html, unsafe_allow_html=True)
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-weight:bold;'>Assessment confidence:</p>", unsafe_allow_html=True)
    st.progress(min(int(confidence * 100), 100))
    st.markdown(
        "<p class='report-disclaimer' style='font-size:12px;color:#9fc9d9;margin-top:10px;'>"
        "This assessment evaluates basic physicochemical parameters (pH, TDS, turbidity, temperature) "
        "and provides an indication of water quality; it does not replace comprehensive laboratory testing "
        "for microbiological or chemical safety.</p>",
        unsafe_allow_html=True,
    )
    render_print_button(dt_str, ph, tds, turbidity, temperature, ml_prob, purity,
                         classification, final_decision, confidence, treatment)


def render_print_button(dt_str, ph, tds, turbidity, temperature, ml_prob, purity,
                         classification, final_decision, confidence, treatment):

    recommendations_html = "".join(f"<li>{t}</li>" for t in treatment)
    summary_text = (
        f"Water quality test conducted on {dt_str}. The machine learning prediction is "
        f"{'SAFE' if ml_prob[1] >= 0.5 else 'UNSAFE'} with {ml_prob[1]*100:.2f}% safe probability. "
        f"The calculated purity index is {purity:.2f}%, which is classified as {classification}. "
        f"The final decision is {final_decision}."
    )

    eda = build_eda_print_assets(ph, tds, turbidity, temperature, purity)

    distribution_block = f'<img src="data:image/png;base64,{eda["distribution"]}" style="width:100%;border-radius:10px;margin-top:10px;" />'
    score_block = f'<img src="data:image/png;base64,{eda["score"]}" style="width:100%;border-radius:10px;margin-top:10px;" />'

    if eda.get("trend"):
        trend_block = (
            f'<img src="data:image/png;base64,{eda["trend"]}" style="width:100%;border-radius:10px;margin-top:10px;" />'
            f'<p style="font-size:13px;color:#555;margin-top:8px;">This reading\'s purity index '
            f'({purity:.2f}%) compared to the historical average of {eda["avg_purity"]:.2f}% '
            f'across {eda["num_tests"]} tests.</p>'
        )
    else:
        trend_block = '<p style="font-size:13px;color:#777;">Not enough historical tests yet to show a trend.</p>'

    who_rows_html = "".join(
        f'<tr><td>{r["Parameter"]}</td><td>{r["Reading"]}</td><td>{r["WHO Range"]}</td>'
        f'<td style="color:{"#1e7e45" if r["Status"]=="Within range" else "#c0392b"};font-weight:600;">{r["Status"]}</td></tr>'
        for r in eda["who_rows"]
    )

    percentile_html = "".join(
        f'<div class="card" style="text-align:center;"><p style="margin:0;font-size:12px;color:#555;">{label}</p>'
        f'<p style="margin:2px 0 0 0;font-size:18px;font-weight:700;color:#023e58;">{pct:.0f}th pct</p></div>'
        for label, pct in eda.get("percentiles", {}).items()
    )

    print_html = f"""
    <html>
    <head>
    <meta charset="utf-8" />
    <title>FlowGuard — Water Quality Report - {dt_str}</title>
    <style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #12232e; background: linear-gradient(180deg, #eaf6ff 0%, #f8fbff 45%, #ffffff 100%); padding: 24px; }}
    .container {{ max-width: 960px; margin: 0 auto; background: #ffffff; padding: 32px; border-radius: 24px; box-shadow: 0 24px 60px rgba(12, 78, 110, 0.10); border: 1px solid rgba(18, 100, 153, 0.12); }}
    .report-header {{ display: flex; align-items: center; gap: 18px; margin-bottom: 24px; }}
    .heading-icon {{ width: 62px; height: 62px; display: grid; place-items: center; background: linear-gradient(135deg, #3fc5ee 0%, #10638a 100%); color: #ffffff; border-radius: 20px; box-shadow: 0 18px 35px rgba(16, 99, 138, 0.18); font-size: 28px; }}
    .heading {{ color: #023e58; margin: 0; font-size: 32px; line-height: 1.1; }}
    .subheading {{ color: #516c81; margin: 8px 0 0 0; font-size: 15px; line-height: 1.75; max-width: 760px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 18px; margin-bottom: 24px; }}
    .summary-card {{ background: linear-gradient(180deg, rgba(14, 107, 156, 0.08), rgba(255, 255, 255, 0.85)); border: 1px solid rgba(14, 107, 156, 0.18); padding: 20px; border-radius: 18px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.8); }}
    .readings-analysis-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 18px; margin-top: 24px; }}
    .section-title {{ font-size: 18px; color: #0f4f6b; margin-bottom: 14px; letter-spacing: 0.02em; }}
    .card {{ background: #f4fbff; padding: 20px; border-radius: 18px; border: 1px solid #d6ecf7; box-shadow: 0 12px 30px rgba(16, 99, 138, 0.06); }}
    .card h2 {{ margin: 0 0 14px 0; color: #023e58; font-size: 20px; }}
    .card p {{ margin: 0 0 10px 0; color: #12232e; line-height: 1.75; }}
    .list-title {{ margin-bottom: 10px; color: #023e58; font-weight: 700; }}
    .recommendations {{ margin-top: 18px; }}
    .recommendations ul {{ padding-left: 22px; margin-top: 10px; }}
    .recommendations li {{ margin-bottom: 10px; color: #12232e; }}
    .section {{ margin-top: 30px; }}
    .section h2 {{ color: #023e58; border-bottom: 2px solid #d6ecf7; padding-bottom: 8px; margin-bottom: 18px; }}
    table.who-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: #f9fdff; box-shadow: inset 0 0 0 1px rgba(55, 152, 217, 0.08); }}
    table.who-table th, table.who-table td {{ text-align: left; padding: 12px 14px; border-bottom: 1px solid #e4f4fb; font-size: 13px; }}
    table.who-table th {{ background: #eaf6ff; color: #0f4f6b; }}
    table.who-table tr:nth-child(even) {{ background: rgba(46, 184, 201, 0.05); }}
    .footer {{ margin-top: 34px; color: #6b7c8e; font-size: 13px; text-align: center; }}
    .print-button {{ display:inline-flex; align-items:center; gap:10px; margin-top:20px; padding:12px 22px; background: linear-gradient(135deg, #10638a, #2eb8c9); color:#fff; border-radius:12px; text-decoration:none; cursor:pointer; border:none; font-size:14px; font-weight:700; box-shadow:0 18px 30px rgba(16,99,138,0.2); }}
    .print-button.print-close {{ background: linear-gradient(135deg, #4b5563, #1f2a38); }}
    .report-actions {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:18px; }}
    .report-note {{ margin-top:14px; color:#4a6374; font-size:13px; line-height:1.6; }}
    .section-divider {{ height: 1px; width: 100%; margin: 30px 0; background: linear-gradient(90deg, transparent, #a7d9f2, transparent); }}
    @media print {{ .print-button {{ display:none; }} body {{ background: #fff; }} .container {{ box-shadow:none; }} }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="report-header">
            <div class="heading-icon">🌊</div>
            <div>
                <h1 class="heading">FlowGuard — Water Quality Report</h1>
                <p class="subheading">This report summarizes the assessment results, exploratory analysis, and treatment recommendations.</p>
            </div>
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <div class="section-title">Final Decision</div>
                <p>{final_decision}</p>
            </div>
            <div class="summary-card">
                <div class="section-title">Confidence</div>
                <p>{confidence:.1f}%</p>
            </div>
        </div>

        <div class="card">
            <h2>Summary</h2>
            <p>{summary_text}</p>
        </div>

        <p style="font-size:13px;color:#516c81;margin-top:12px;">This assessment evaluates basic physicochemical parameters (pH, TDS, turbidity, temperature) and provides an indication of water quality; it does not replace comprehensive laboratory testing for microbiological or chemical safety.</p>

        <div class="readings-analysis-grid">
            <div class="card">
                <h2>Readings</h2>
                <p><strong>pH:</strong> {ph}</p>
                <p><strong>TDS:</strong> {tds} ppm</p>
                <p><strong>Turbidity:</strong> {turbidity} NTU</p>
                <p><strong>Temperature:</strong> {temperature} °C</p>
            </div>
            <div class="card">
                <h2>Analysis</h2>
                <p><strong>ML Safe Probability:</strong> {ml_prob[1]*100:.2f}%</p>
                <p><strong>Purity Index:</strong> {purity:.2f}%</p>
                <p><strong>WQI Class:</strong> {classification}</p>
                <p><strong>Final Decision:</strong> {final_decision}</p>
                <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
            </div>
        </div>

        <div class="card recommendations">
            <h2 class="list-title">Recommended Treatment</h2>
            <p>The following recommendations are based on the current water quality readings.</p>
            <ul>{recommendations_html}</ul>
        </div>

        <div class="section">
            <h2>📊 Exploratory Data Analysis</h2>

            <h3 style="color:#023e58;">Distribution Comparison vs. Reference Dataset</h3>
            <p style="color:#4a6374;margin-top:8px;margin-bottom:18px;line-height:1.6;">This chart compares your current water readings with the reference dataset. It shows how your values fit into the wider population and whether they are common or more extreme compared to past measurements.</p>
            {distribution_block}
            <div class="pct-box">{percentile_html}</div>
            <p style="color:#4a6374;margin-top:12px;line-height:1.6;">Each percentile indicates where your reading stands relative to the reference dataset. For example, the 80th percentile means this value is higher than 80% of the historical samples.</p>

            <h3 style="color:#023e58;margin-top:24px;">Purity Score Breakdown</h3>
            <p style="color:#4a6374;margin-top:8px;margin-bottom:18px;line-height:1.6;">This graph shows how each parameter contributes to the overall purity index. Use it to identify which measurement has the biggest impact on water quality.</p>
            {score_block}

            <h3 style="color:#023e58;margin-top:24px;">Trend Over Time</h3>
            <p style="color:#4a6374;margin-top:8px;margin-bottom:18px;line-height:1.6;">This trend chart tracks how your water quality has changed over previous tests. It helps you see whether conditions are improving, stabilizing, or getting worse over time.</p>
            {trend_block}

            <h3 style="color:#023e58;margin-top:24px;">WHO Compliance</h3>
            <table class="who-table">
                <tr><th>Parameter</th><th>Reading</th><th>WHO Range</th><th>Status</th></tr>
                {who_rows_html}
            </table>
        </div>

        <div class="report-actions">
            <button class="print-button" onclick="window.print();">Print this report</button>
            <button class="print-button print-close" onclick="window.close();">Close report</button>
        </div>
        <p class="report-note">After printing, close this tab to return to the app. If the print dialog does not appear immediately, use the browser's print action.</p>
        <div class="footer">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    </body>
    </html>
    """
    render_print_window_button("🖨️ Open Printable Report", f"print_btn_{dt_str}", print_html)


def render_print_window_button(label: str, unique_key: str, html_content: str, width_full: bool = True):
    """Reusable 'open a printable report in a new tab' button.

    A JS popup (window.open + document.write) is used instead of a data: URI
    link, since data: URI navigation is blocked/restricted by Chrome for
    larger payloads. Writing the HTML into a blank window avoids that.
    """
    escaped_html = html_content.replace("\\", "\\\\").replace("`", "\\`").replace("</script>", "<\\/script>")
    safe_id = "print_btn_" + "".join(ch if ch.isalnum() else "_" for ch in unique_key)
    btn_width = "100%" if width_full else "auto"

    components.html(f"""
        <button id="{safe_id}" style="
            background:linear-gradient(135deg, #10638a 0%, #2eb8c9 100%);color:white;border:1px solid rgba(46,184,201,0.35);padding:10px 18px;
            min-height:44px;border-radius:10px;cursor:pointer;font-size:14px;font-weight:600;white-space:nowrap;line-height:1.2;box-shadow:0 10px 26px rgba(0,0,0,0.35);width:{btn_width};display:inline-block;">
            {label}
        </button>
        <script>
        document.getElementById("{safe_id}").addEventListener("click", function() {{
            var reportWindow = window.open('', '_blank');
            if (!reportWindow) {{
                alert('Please allow pop-ups for this site to open the printable report.');
                return;
            }}
            reportWindow.document.open();
            reportWindow.document.write(`{escaped_html}`);
            reportWindow.document.close();
            reportWindow.focus();
        }});
        </script>
    """, height=60)


_PRINT_REPORT_CSS = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #12232e; background: #f8fbff; padding: 24px; }
.container { max-width: 950px; margin: 0 auto; background: #ffffff; padding: 28px 32px; border-radius: 18px; box-shadow: 0 18px 45px rgba(12, 78, 110, 0.08); }
.heading { color: #023e58; margin-bottom: 10px; }
.subheading { color: #555; margin-top: 0; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e0eef5; font-size: 13px; }
th { background: #f4fbff; color: #023e58; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 14px; font-size: 12px; font-weight: 700; }
.badge-safe { background: #d6f5e1; color: #1e7e45; }
.badge-unsafe { background: #fdd9d5; color: #c0392b; }
.badge-uncertain { background: #fff1cc; color: #b8860b; }
.record { border: 1px solid #e0eef5; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; }
.record-head { display: flex; justify-content: space-between; align-items: center; }
.record-sub { color: #555; font-size: 13px; margin-top: 6px; }
.record ul { margin-top: 8px; padding-left: 20px; }
.footer { margin-top: 32px; color: #6b7c8e; font-size: 13px; }
.print-button { display:inline-block;margin-top:20px;padding:12px 20px;background:#10638a;color:#fff;border-radius:10px;text-decoration:none;cursor:pointer; border:none; font-size:14px; }
@media print { .print-button { display:none; } body { background: #fff; } .container { box-shadow:none; } }
.report-actions { display:flex; flex-wrap:wrap; gap:12px; margin-top:18px; }
.print-close { background: linear-gradient(135deg, #4b5563, #1f2a38); }
.report-note { margin-top:14px; color:#4a6374; font-size:13px; line-height:1.6; }
"""


def build_history_print_html(df: pd.DataFrame) -> str:
    """Printable report of the full test history table."""
    df_sorted = df.copy().iloc[::-1]
    rows_html = "".join(
        f"<tr><td>{r['DateTime']}</td><td>{r['pH']}</td><td>{r['TDS_ppm']}</td>"
        f"<td>{r['Turbidity_NTU']}</td><td>{r['Temperature_C']}</td>"
        f"<td>{r['WQI_Class']}</td><td>{r['Purity_Index']}%</td>"
        f"<td><span class='badge {decision_badge_class(str(r['Final_Decision']))}'>{r['Final_Decision']}</span></td></tr>"
        for _, r in df_sorted.iterrows()
    )
    return f"""
    <html><head><meta charset="utf-8" /><title>FlowGuard — Test History Report</title>
    <style>{_PRINT_REPORT_CSS}</style></head>
    <body><div class="container">
        <h1 class="heading">💧 FlowGuard — Test History Report</h1>
        <p class="subheading">Full record of {len(df)} water quality test(s) run in this app.</p>
        <p style="font-size:13px;color:#516c81;margin-top:8px;">This report is based on basic physicochemical readings (pH, TDS, turbidity, temperature) and provides screening-level information; it does not replace laboratory testing for microbiological or chemical safety.</p>
        <table>
            <tr><th>Date/Time</th><th>pH</th><th>TDS (ppm)</th><th>Turbidity (NTU)</th>
                <th>Temp (°C)</th><th>WQI Class</th><th>Purity</th><th>Final Decision</th></tr>
            {rows_html}
        </table>
        <div class="report-actions">
            <button class="print-button" onclick="window.print();">Print this report</button>
            <button class="print-button print-close" onclick="window.close();">Close report</button>
        </div>
        <p class="report-note">After printing, close this tab to return to the app. If the print dialog does not appear immediately, use the browser's print action.</p>
        <div class="footer">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div></body></html>
    """


def build_recommendations_print_html(df: pd.DataFrame) -> str:
    """Printable report of treatment recommendations for every test on record."""
    df_sorted = df.copy().iloc[::-1]
    records_html = ""
    for _, r in df_sorted.iterrows():
        treatments = str(r["Treatment"]).split("; ")
        treat_items = "".join(f"<li>{t}</li>" for t in treatments)
        badge = decision_badge_class(str(r["Final_Decision"]))
        records_html += f"""
        <div class="record">
            <div class="record-head">
                <strong>{r['DateTime']}</strong>
                <span class="badge {badge}">{r['Final_Decision']}</span>
            </div>
            <div class="record-sub">WQI Class: {r['WQI_Class']} · Purity Index: {r['Purity_Index']}%</div>
            <ul>{treat_items}</ul>
        </div>
        """
    return f"""
    <html><head><meta charset="utf-8" /><title>FlowGuard — Recommendations Report</title>
    <style>{_PRINT_REPORT_CSS}</style></head>
    <body><div class="container">
        <h1 class="heading">💊 FlowGuard — Treatment Recommendations Report</h1>
        <p class="subheading">Recommended treatment plans for all {len(df)} test(s) on record.</p>
        <p style="font-size:13px;color:#516c81;margin-top:8px;">These recommendations are generated from basic physicochemical indicators (pH, TDS, turbidity, temperature) and are intended as guidance; lab confirmation may be required for definitive action.</p>
        {records_html}
        <div class="report-actions">
            <button class="print-button" onclick="window.print();">Print this report</button>
            <button class="print-button print-close" onclick="window.close();">Close report</button>
        </div>
        <p class="report-note">After printing, close this tab to return to the app. If the print dialog does not appear immediately, use the browser's print action.</p>
        <div class="footer">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div></body></html>
    """


# ============================================================================
# PAGE: RESULT (from a freshly-run test)
# ============================================================================

def render_result_page():
    result = st.session_state.latest_result
    if result is None:
        st.warning("No result available. Run a new test first.")
        if st.button("Go to New Test"):
            go_to("new_test")
            st.rerun()
        return

    ph = float(result["pH"])
    tds = float(result["TDS_ppm"])
    turbidity = float(result["Turbidity_NTU"])
    temperature = float(result["Temperature_C"])
    purity = float(result["Purity_Index"])
    classification = result["WQI_Class"]
    final_decision = result["Final_Decision"]
    confidence = float(result["Confidence"]) / 100
    treatment = result["Treatment"]
    if isinstance(treatment, str):
        treatment = treatment.split("; ") if treatment else []
    dt_str = result["DateTime"]  # FIX: was previously referenced undefined

    if st.button("← Back to New Test"):
        go_to("new_test")
        st.rerun()

    st.markdown("### Result")
    st.markdown(f"<p>Test run on: {dt_str}</p>", unsafe_allow_html=True)

    render_result_block(
        dt_str, ph, tds, turbidity, temperature,
        1 if result["ML_Prediction"] == "SAFE" else 0,
        [1 - float(result["ML_Safe_Prob"]) / 100, float(result["ML_Safe_Prob"]) / 100],
        purity, classification, final_decision, confidence, treatment,
    )
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    render_eda(ph, tds, turbidity, temperature, purity)


# ============================================================================
# PAGE: HISTORY / PREVIOUS TESTS
# ============================================================================

def render_history():
    st.markdown("<h3>Previous Tests</h3>", unsafe_allow_html=True)
    df = load_history()

    if len(df) == 0:
        st.info("No previous tests found. Run a test from the **New Test** tab.")
        return

    df_display = df.copy().iloc[::-1].reset_index(drop=False)

    search = st.text_input("🔍 Filter by date (YYYY-MM-DD) or decision keyword")
    if search:
        mask = df_display.apply(
            lambda r: search.lower() in str(r["DateTime"]).lower()
            or search.lower() in str(r["Final_Decision"]).lower(),
            axis=1,
        )
        df_display = df_display[mask]

    for idx, row in df_display.iterrows():
        with st.container():
            cols = st.columns([3, 2, 2, 2, 1.5, 1.3, 1.3])
            cols[0].markdown(f"<b>{row['DateTime']}</b>", unsafe_allow_html=True)
            cols[1].markdown(f"pH: {row['pH']}", unsafe_allow_html=True)
            cols[2].markdown(f"TDS: {row['TDS_ppm']}", unsafe_allow_html=True)
            cols[3].markdown(f"Turb: {row['Turbidity_NTU']}", unsafe_allow_html=True)
            cols[4].markdown(f"<b>{row['Final_Decision']}</b>", unsafe_allow_html=True)

            if cols[5].button("View", key=f"view_{idx}_{row['DateTime']}"):
                st.session_state.selected_record = row.to_dict()
                go_to("record_detail")
                st.rerun()
            if cols[6].button("Delete", key=f"delete_{idx}_{row['index']}"):
                df = df.drop(index=[row["index"]]).reset_index(drop=True)
                save_history(df)
                st.success(f"Deleted record from {row['DateTime']}")
                st.rerun()
            st.markdown("<div class='section-divider thin'></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Build CSV with disclaimer header and printable HTML, then show both controls
    disclaimer_header = (
        "# FlowGuard — Screening-level water quality data\n"
        "# This CSV contains basic physicochemical readings (pH, TDS, turbidity, temperature).\n"
        "# These values provide screening-level indicators and do not replace comprehensive laboratory testing for microbiological or chemical safety.\n"
        "# Generated by FlowGuard\n\n"
    )
    csv_content = disclaimer_header + df.to_csv(index=False)
    csv_bytes = csv_content.encode("utf-8")

    print_html = build_history_print_html(df)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="⬇️ Download Full History (CSV)",
            data=csv_bytes,
            file_name="water_quality_history_with_disclaimer.csv",
            mime="text/csv",
            key="download_history",
        )
    with col2:
        render_print_window_button("🖨️ Print Test History", "print_history", print_html)


# ============================================================================
# PAGE: RECORD DETAIL (re-render full result for a past test)
# ============================================================================

def render_record_detail():
    row = st.session_state.selected_record
    if row is None:
        st.warning("No record selected.")
        return

    if st.button("← Back to Previous Tests"):
        go_to("history")
        st.rerun()

    ph = float(row["pH"])
    tds = float(row["TDS_ppm"])
    turbidity = float(row["Turbidity_NTU"])
    temperature = float(row["Temperature_C"])
    purity = float(row["Purity_Index"])
    classification = row["WQI_Class"]
    final_decision = row["Final_Decision"]
    confidence = float(row["Confidence"]) / 100
    treatment = str(row["Treatment"]).split("; ")
    ml_pred = 1 if row["ML_Prediction"] == "SAFE" else 0
    ml_safe_prob = float(row["ML_Safe_Prob"]) / 100
    ml_prob = [1 - ml_safe_prob, ml_safe_prob]

    render_result_block(
        row["DateTime"], ph, tds, turbidity, temperature,
        ml_pred, ml_prob, purity, classification,
        final_decision, confidence, treatment,
    )
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    render_eda(ph, tds, turbidity, temperature, purity)


# ============================================================================
# PAGE: RECOMMENDATIONS (all treatment plans, grouped)
# ============================================================================

def render_recommendations():
    st.markdown("<h3>Treatment Recommendations</h3>", unsafe_allow_html=True)
    df = load_history()

    if len(df) == 0:
        st.info("No recommendations yet. Run a test from the **New Test** tab.")
        return

    df_display = df.copy().iloc[::-1].reset_index(drop=True)

    for _, row in df_display.iterrows():
        badge = decision_badge_class(str(row["Final_Decision"]))
        treatments = str(row["Treatment"]).split("; ")
        treat_html = "".join(f"<li>{t}</li>" for t in treatments)
        st.markdown(f"""
        <div class="record-row">
            <div class="record-main">
                <span class="record-date">{row['DateTime']}</span>
                <span class="badge {badge}">{row['Final_Decision']}</span>
            </div>
            <div class="record-sub">WQI Class: {row['WQI_Class']} · Purity: {row['Purity_Index']}%</div>
            <ul class="treatment-list compact">{treat_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    render_print_window_button(
        "🖨️ Print Recommendations Report", "recommendations_report",
        build_recommendations_print_html(df),
    )


# ============================================================================
# ROUTER
# ============================================================================

if st.session_state.page == "landing":
    render_landing()
    scroll_to_top()
else:
    render_nav()
    scroll_to_top()
    if st.session_state.page == "dashboard_home":
        render_dashboard_home()
    elif st.session_state.page == "new_test":
        render_new_test()
    elif st.session_state.page == "result":
        render_result_page()
    elif st.sess4cldddddddddddddddddddddddddddddddddddddddddddddddddddddcion_state.page == "history":
        render_history()
    elif st.session_state.page == "record_detail":
        render_record_detail()
    elif st.session_state.page == "recommendations":
        render_recommendations()