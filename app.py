import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import ttest_ind

st.set_page_config(
    page_title="AutoInsight | Used Car Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned_cars.csv"
CLUSTER_PATH = ROOT / "data" / "clustered_cars.csv"
PRICE_MODEL_PATH = ROOT / "models" / "price_model.pkl"
CLASS_MODEL_PATH = ROOT / "models" / "classification_model.pkl"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#0b1120; --panel:#111a2e; --panel2:#15223a; --line:rgba(148,163,184,.16); --muted:#91a3bd; --text:#edf4ff; --teal:#38d6c2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 80% -10%, rgba(56,214,194,.10), transparent 32%), #0b1120; color:var(--text); }
[data-testid="stHeader"] { background:rgba(11,17,32,.85); }
[data-testid="stSidebar"] { background:#0e1729; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#dce7f8; }
.block-container { padding-top:1.7rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:#f4f8ff; }
h1 { font-weight:800 !important; }
p, label, .stMarkdown { color:#d5e0f0; }
small, .muted { color:var(--muted) !important; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(21,34,58,.98),rgba(15,25,44,.98)); border:1px solid var(--line); padding:18px 20px; border-radius:18px; box-shadow:0 10px 28px rgba(0,0,0,.12); }
div[data-testid="stMetricLabel"] { color:#9eb0c9 !important; font-size:.86rem; }
div[data-testid="stMetricValue"] { color:#f4f8ff !important; font-family:'Manrope',sans-serif; font-weight:800; }
div[data-testid="stMetricDelta"] { color:var(--teal) !important; }
.hero { padding:30px 32px; border-radius:24px; background:linear-gradient(115deg,rgba(21,34,58,.98),rgba(15,48,64,.92)); border:1px solid rgba(56,214,194,.22); margin-bottom:22px; position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; right:-45px; top:-90px; width:260px; height:260px; border-radius:50%; border:1px solid rgba(56,214,194,.17); box-shadow:0 0 0 35px rgba(56,214,194,.025),0 0 0 70px rgba(56,214,194,.02); }
.eyebrow { color:#5eead4; font-size:.76rem; text-transform:uppercase; letter-spacing:.16em; font-weight:700; margin-bottom:8px; }
.hero h1 { font-size:2.3rem; margin:0 0 8px 0; }
.hero p { color:#b7c8df; max-width:720px; margin:0; line-height:1.65; }
.section-head { margin:25px 0 12px; font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:750; color:#edf4ff; }
.panel { background:rgba(17,26,46,.9); border:1px solid var(--line); border-radius:18px; padding:20px 22px; }
.pill { display:inline-block; border:1px solid rgba(56,214,194,.28); background:rgba(56,214,194,.08); color:#5eead4; padding:5px 10px; border-radius:999px; font-size:.76rem; font-weight:700; }
div.stButton > button { background:linear-gradient(100deg,#2dd4bf,#4ce0d0); color:#071522; border:0; border-radius:12px; padding:.62rem 1.2rem; font-weight:800; min-height:44px; }
div.stButton > button:hover { background:#99f6e4; color:#071522; border:0; }
.stDownloadButton button { border-radius:10px; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#111a2e; border-color:#30415c; border-radius:10px; }
.stTextInput input, .stNumberInput input { color:#f4f8ff; }
hr { border-color:var(--line); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#cbd8ea"),
    margin=dict(l=18, r=18, t=55, b=18),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.12)"),
    yaxis=dict(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.12)")
)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_clusters():
    if CLUSTER_PATH.exists():
        return pd.read_csv(CLUSTER_PATH)
    return None

@st.cache_resource
def load_price_model():
    if PRICE_MODEL_PATH.exists():
        return joblib.load(PRICE_MODEL_PATH)
    return None

@st.cache_resource
def load_class_model():
    if CLASS_MODEL_PATH.exists():
        return joblib.load(CLASS_MODEL_PATH)
    return None

df = load_data()
cluster_df = load_clusters()
price_model = load_price_model()
class_model = load_class_model()

if df is None:
    st.error("Dataset not found. Place cleaned_cars.csv inside the data folder.")
    st.code("data/cleaned_cars.csv")
    st.stop()

if "Car_Age" not in df.columns and "Year" in df.columns:
    df["Car_Age"] = 2026 - df["Year"]

features = [
    "Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats",
    "Fuel_Type", "Transmission", "Owner_Type", "Location"
]

def money(value):
    return f"₹{value:,.2f} L"

def hero(kicker, title, description):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True
    )

def chart(fig, height=350):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def missing_artifacts():
    missing = []
    if price_model is None:
        missing.append("models/price_model.pkl")
    if class_model is None:
        missing.append("models/classification_model.pkl")
    if missing:
        st.warning("Some trained models are missing. Run the relevant training scripts first: " + ", ".join(missing))

with st.sidebar:
    st.markdown("""
    <div style="padding:8px 4px 22px">
      <div style="display:flex;align-items:center;gap:11px">
        <div style="width:42px;height:42px;border-radius:14px;background:linear-gradient(135deg,#2dd4bf,#0f766e);display:flex;align-items:center;justify-content:center;font-size:22px">🚘</div>
        <div><div style="font-family:Manrope;font-weight:800;font-size:1.13rem;color:#f4f8ff">AutoInsight</div><div style="font-size:.73rem;color:#8fa4bf">USED CAR INTELLIGENCE</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="font-size:.72rem;letter-spacing:.13em;color:#7186a4;font-weight:700;margin:5px 0 10px">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Price Predictor",
            "Price Classification",
            "Explore Dataset",
            "Car Recommendations",
            "K-Means Clusters",
            "PCA & Model Evaluation",
            "Statistical Analysis"
        ],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown('<div style="font-size:.72rem;letter-spacing:.13em;color:#7186a4;font-weight:700;margin-bottom:8px">DATA STATUS</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="pill">● Dataset loaded</span>', unsafe_allow_html=True)
    st.caption(f"{len(df):,} records · {len(df.columns)} columns")
    st.caption("Models run locally in this app.")

if page == "Overview":
    hero("USED CAR INTELLIGENCE PLATFORM", "Make your next car decision with confidence.", "Explore market patterns, estimate a car's value, and discover vehicles that match your preferences — powered by machine learning.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cars analysed", f"{len(df):,}")
    c2.metric("Average listed price", money(df["Price"].mean()))
    c3.metric("Median listed price", money(df["Price"].median()))
    c4.metric("Brands represented", f"{df['Brand'].nunique():,}" if "Brand" in df else "—")
    st.markdown('<div class="section-head">Market snapshot</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        price_fig = px.histogram(df, x="Price", nbins=45, title="Price distribution", labels={"Price":"Price (₹ lakh)", "count":"Number of cars"}, color_discrete_sequence=["#38d6c2"])
        chart(price_fig, 360)
    with right:
        fuel_counts = df["Fuel_Type"].value_counts().rename_axis("Fuel type").reset_index(name="Cars")
        fuel_fig = px.pie(fuel_counts, names="Fuel type", values="Cars", hole=.68, title="Inventory by fuel type", color_discrete_sequence=["#38d6c2","#6489f5","#f5b65b","#a78bfa"])
        fuel_fig.update_traces(textposition="inside", textinfo="percent")
        chart(fuel_fig, 360)
    left, right = st.columns(2)
    with left:
        year_fig = px.scatter(df.sample(min(1500, len(df)), random_state=42), x="Year", y="Price", color="Transmission", opacity=.65, title="Price vs manufacturing year", labels={"Year":"Manufacturing year","Price":"Price (₹ lakh)"}, color_discrete_sequence=["#38d6c2","#818cf8"])
        chart(year_fig, 330)
    with right:
        power_fig = px.scatter(df.sample(min(1500, len(df)), random_state=42), x="Power", y="Price", color="Fuel_Type", opacity=.6, title="Price vs engine power", labels={"Power":"Power (bhp)","Price":"Price (₹ lakh)"})
        chart(power_fig, 330)
    st.markdown('<div class="panel"><span class="pill">PROJECT MODULES</span><br><br><b>Predict · Classify · Explore · Recommend · Cluster · Evaluate</b><p style="color:#9fb1ca;margin-bottom:0">This workspace brings the nine data science lab experiments into one interactive application.</p></div>', unsafe_allow_html=True)

elif page == "Price Predictor":
    hero("MACHINE LEARNING · REGRESSION", "Used car price predictor", "Enter the vehicle's details to estimate its resale value. The result is an estimate based on patterns learned from the training dataset.")
    missing_artifacts()
    if price_model is None:
        st.info("The price prediction model was not found. Run: python src/linear_regression.py")
    else:
        st.markdown('<div class="section-head">Vehicle specifications</div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.selectbox("Manufacturing year", list(range(2019, 1997, -1)), index=4)
                km = st.number_input("Kilometers driven", min_value=0, max_value=1_000_000, value=50000, step=1000)
                mileage = st.number_input("Mileage (km/l or dataset unit)", min_value=1.0, max_value=100.0, value=18.0, step=0.5)
            with col2:
                engine = st.number_input("Engine capacity (cc)", min_value=500.0, max_value=10000.0, value=1500.0, step=50.0)
                power = st.number_input("Power (bhp)", min_value=20.0, max_value=1500.0, value=100.0, step=5.0)
                seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
            with col3:
                fuel_options = sorted(df["Fuel_Type"].dropna().unique().tolist())
                fuel = st.selectbox("Fuel type", fuel_options, index=fuel_options.index("Petrol") if "Petrol" in fuel_options else 0)
                transmission_options = sorted(df["Transmission"].dropna().unique().tolist())
                transmission = st.selectbox("Transmission", transmission_options, index=transmission_options.index("Manual") if "Manual" in transmission_options else 0)
                owner_options = sorted(df["Owner_Type"].dropna().unique().tolist())
                owner = st.selectbox("Owner type", owner_options, index=owner_options.index("First") if "First" in owner_options else 0)
                location_options = sorted(df["Location"].dropna().unique().tolist())
                location = st.selectbox("City", location_options)
            submitted = st.form_submit_button("Estimate resale price  →", use_container_width=True)
        if submitted:
            input_df = pd.DataFrame([{
                "Year": year,
                "Kilometers_Driven": km,
                "Mileage": mileage,
                "Engine": engine,
                "Power": power,
                "Seats": seats,
                "Fuel_Type": fuel,
                "Transmission": transmission,
                "Owner_Type": owner,
                "Location": location
            }])
            try:
                prediction = float(price_model.predict(input_df)[0])
                prediction = max(0.0, prediction)
                st.markdown('<div class="section-head">Your estimate</div>', unsafe_allow_html=True)
                a, b = st.columns([1.25, 1])
                with a:
                    st.markdown(f'<div class="hero"><div class="eyebrow">ESTIMATED RESALE VALUE</div><h1 style="font-size:2.8rem">{money(prediction)}</h1><p>Predicted value for a {year} {fuel.lower()} vehicle in {location}.</p></div>', unsafe_allow_html=True)
                with b:
                    st.metric("Vehicle age", f"{2026-year} years")
                    st.metric("Annualized usage", f"{km / max(2026-year, 1):,.0f} km/year")
                st.caption("This is a model estimate, not a guaranteed selling price. Actual prices vary with condition, service history, trim, ownership, and local demand.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

elif page == "Price Classification":
    hero("MACHINE LEARNING · CLASSIFICATION", "Understand the price segment", "Classify a vehicle into a relative Low, Medium, or High price segment using the trained Logistic Regression model.")
    missing_artifacts()
    if class_model is None:
        st.info("The classification model was not found. Run: python src/classification.py")
    else:
        st.markdown("""<div class="panel">Enter the same vehicle details used by the predictor. Categories are relative to the training dataset's price distribution, not fixed market-price bands.</div>""", unsafe_allow_html=True)
        with st.form("classification_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                year = st.selectbox("Manufacturing year", list(range(2019, 1997, -1)), index=4, key="cl_year")
                km = st.number_input("Kilometers driven", min_value=0, max_value=1_000_000, value=50000, step=1000, key="cl_km")
                mileage = st.number_input("Mileage", min_value=1.0, max_value=100.0, value=18.0, step=0.5, key="cl_mileage")
            with col2:
                engine = st.number_input("Engine (cc)", min_value=500.0, max_value=10000.0, value=1500.0, step=50.0, key="cl_engine")
                power = st.number_input("Power (bhp)", min_value=20.0, max_value=1500.0, value=100.0, step=5.0, key="cl_power")
                seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2, key="cl_seats")
            with col3:
                fuel = st.selectbox("Fuel type", sorted(df["Fuel_Type"].dropna().unique()), key="cl_fuel")
                transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()), key="cl_trans")
                owner = st.selectbox("Owner type", sorted(df["Owner_Type"].dropna().unique()), key="cl_owner")
                location = st.selectbox("City", sorted(df["Location"].dropna().unique()), key="cl_location")
            submitted = st.form_submit_button("Classify vehicle  →", use_container_width=True)
        if submitted:
            x = pd.DataFrame([{"Year":year,"Kilometers_Driven":km,"Mileage":mileage,"Engine":engine,"Power":power,"Seats":seats,"Fuel_Type":fuel,"Transmission":transmission,"Owner_Type":owner,"Location":location}])
            try:
                category = class_model.predict(x)[0]
                probs = class_model.predict_proba(x)[0]
                labels = class_model.named_steps["classifier"].classes_
                probability = float(probs[list(labels).index(category)]) * 100
                st.markdown('<div class="section-head">Classification result</div>', unsafe_allow_html=True)
                a, b = st.columns([1, 1.4])
                with a:
                    st.markdown(f'<div class="hero"><div class="eyebrow">PREDICTED PRICE SEGMENT</div><h1>{category}</h1><p>Model confidence for the predicted class: {probability:.1f}%</p></div>', unsafe_allow_html=True)
                with b:
                    chart(px.bar(pd.DataFrame({"Segment":labels,"Probability":probs*100}), x="Segment", y="Probability", color="Segment", title="Predicted class probabilities", color_discrete_sequence=["#38d6c2","#6489f5","#f5b65b"]), 280)
            except Exception as exc:
                st.error(f"Classification failed: {exc}")

elif page == "Explore Dataset":
    hero("EXPLORATORY DATA ANALYSIS", "Explore the used-car market", "Filter the inventory, compare price patterns, and inspect the data behind the models.")
    with st.expander("Filter records", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            fuels = st.multiselect("Fuel type", sorted(df["Fuel_Type"].dropna().unique()), default=sorted(df["Fuel_Type"].dropna().unique()))
        with c2:
            transmissions = st.multiselect("Transmission", sorted(df["Transmission"].dropna().unique()), default=sorted(df["Transmission"].dropna().unique()))
        with c3:
            price_range = st.slider("Price range (₹ lakh)", float(df["Price"].min()), float(df["Price"].max()), (float(df["Price"].min()), float(min(df["Price"].quantile(.95), df["Price"].max()))))
    filtered = df[df["Fuel_Type"].isin(fuels) & df["Transmission"].isin(transmissions) & df["Price"].between(price_range[0], price_range[1])]
    a,b,c = st.columns(3)
    a.metric("Filtered cars", f"{len(filtered):,}")
    b.metric("Average price", money(filtered["Price"].mean()) if len(filtered) else "—")
    c.metric("Median price", money(filtered["Price"].median()) if len(filtered) else "—")
    if len(filtered):
        left, right = st.columns(2)
        with left:
            chart(px.histogram(filtered, x="Price", nbins=40, color="Transmission", barmode="overlay", opacity=.75, title="Price distribution", labels={"Price":"Price (₹ lakh)"}), 330)
        with right:
            chart(px.box(filtered, x="Fuel_Type", y="Price", color="Fuel_Type", title="Price by fuel type", labels={"Fuel_Type":"Fuel type","Price":"Price (₹ lakh)"}), 330)
        left, right = st.columns(2)
        with left:
            chart(px.scatter(filtered.sample(min(1200,len(filtered)), random_state=42), x="Kilometers_Driven", y="Price", color="Fuel_Type", opacity=.65, title="Price vs distance driven", labels={"Kilometers_Driven":"Kilometers driven","Price":"Price (₹ lakh)"}), 330)
        with right:
            chart(px.bar(filtered.groupby("Location", as_index=False)["Price"].median().sort_values("Price", ascending=False), x="Location", y="Price", title="Median price by city", labels={"Location":"City","Price":"Median price (₹ lakh)"}), 330)
    st.markdown('<div class="section-head">Dataset preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_used_cars.csv", "text/csv")

elif page == "Car Recommendations":
    hero("PREFERENCE-BASED SEARCH", "Find cars that fit your needs", "Set your budget and practical requirements to shortlist matching vehicles from the dataset.")
    c1,c2,c3 = st.columns(3)
    with c1:
        budget = st.number_input("Maximum budget (₹ lakh)", min_value=0.5, max_value=200.0, value=10.0, step=0.5)
        min_year = st.selectbox("Minimum manufacturing year", list(range(2019, 1997, -1)), index=4)
    with c2:
        max_km = st.number_input("Maximum kilometers driven", min_value=1000, max_value=1_000_000, value=80000, step=5000)
        fuel = st.selectbox("Fuel preference", ["Any"] + sorted(df["Fuel_Type"].dropna().unique().tolist()))
    with c3:
        transmission = st.selectbox("Transmission preference", ["Any"] + sorted(df["Transmission"].dropna().unique().tolist()))
        result_count = st.select_slider("Number of recommendations", options=[5,10,15,20], value=10)
    if st.button("Find matching cars  →", use_container_width=True):
        candidates = df[(df["Price"] <= budget) & (df["Year"] >= min_year) & (df["Kilometers_Driven"] <= max_km)].copy()
        if fuel != "Any":
            candidates = candidates[candidates["Fuel_Type"] == fuel]
        if transmission != "Any":
            candidates = candidates[candidates["Transmission"] == transmission]
        if candidates.empty:
            st.warning("No vehicles matched these filters. Try increasing the budget or mileage limit.")
        else:
            candidates["Recommendation score"] = (
                (budget - candidates["Price"]) / max(budget, 1)
                + (candidates["Year"] - min_year) / max(2026 - min_year, 1)
                + (max_km - candidates["Kilometers_Driven"]) / max(max_km, 1)
            )
            results = candidates.sort_values("Recommendation score", ascending=False).head(result_count)
            st.success(f"Found {len(results)} matching vehicles.")
            st.dataframe(results[["Name","Year","Fuel_Type","Transmission","Kilometers_Driven","Mileage","Price","Location"]], use_container_width=True, hide_index=True)
            st.caption("Recommendations are ranked by a simple preference score favouring lower price, newer year, and lower distance driven. This is a heuristic shortlist, not a verified market valuation.")

elif page == "K-Means Clusters":
    hero("UNSUPERVISED LEARNING", "Discover vehicle segments", "K-Means groups vehicles with similar age, mileage, engine, power, and usage patterns.")
    if cluster_df is None or "Cluster" not in cluster_df.columns:
        st.info("Clustered data not found. Run: python src/clustering.py")
    else:
        st.metric("Cars included in clustering", f"{len(cluster_df):,}")
        counts = cluster_df["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Cars")
        left,right = st.columns(2)
        with left:
            chart(px.bar(counts, x="Cluster", y="Cars", color="Cluster", title="Vehicles in each cluster", color_discrete_sequence=["#38d6c2","#6489f5","#f5b65b"]), 320)
        with right:
            avg = cluster_df.groupby("Cluster", as_index=False)[["Price","Car_Age","Kilometers_Driven","Mileage","Engine","Power"]].mean()
            chart(px.scatter(avg, x="Engine", y="Price", size="Power", color="Cluster", hover_data=["Car_Age","Kilometers_Driven"], title="Average cluster profile", labels={"Engine":"Average engine (cc)","Price":"Average price (₹ lakh)"}), 320)
        st.markdown('<div class="section-head">Cluster characteristics</div>', unsafe_allow_html=True)
        st.dataframe(cluster_df.groupby("Cluster")[["Price","Car_Age","Kilometers_Driven","Mileage","Engine","Power"]].mean().round(2), use_container_width=True)
        st.caption("The clustering dataset excludes the single record with more than 1,000,000 km, as documented in the clustering experiment.")

elif page == "PCA & Model Evaluation":
    hero("MODEL EVALUATION", "Compare predictive performance", "Review the metrics generated by your completed model experiments. These values come from your training runs.")
    metrics = pd.DataFrame([
        {"Model":"Linear Regression","Metric":"R²","Score":0.7300},
        {"Model":"Classification baseline","Metric":"Accuracy","Score":0.8222},
        {"Model":"Classification baseline","Metric":"Precision","Score":0.8248},
        {"Model":"Classification baseline","Metric":"Recall","Score":0.8222},
        {"Model":"Classification baseline","Metric":"F1-score","Score":0.8228},
        {"Model":"Classification with PCA","Metric":"Accuracy","Score":0.8137},
        {"Model":"Classification with PCA","Metric":"Precision","Score":0.8168},
        {"Model":"Classification with PCA","Metric":"Recall","Score":0.8137},
        {"Model":"Classification with PCA","Metric":"F1-score","Score":0.8146}
    ])
    a,b,c,d = st.columns(4)
    a.metric("Regression R²", "0.73")
    b.metric("Baseline accuracy", "82.22%")
    c.metric("PCA accuracy", "81.37%")
    d.metric("Variance retained", "95.85%")
    left,right = st.columns(2)
    with left:
        chart(px.bar(metrics[metrics["Metric"].isin(["Accuracy","Precision","Recall","F1-score"])], x="Metric", y="Score", color="Model", barmode="group", title="Classification: baseline vs PCA", labels={"Score":"Score"}), 360)
    with right:
        components = pd.DataFrame({"Stage":["Before PCA","After PCA"],"Components":[26,14]})
        chart(px.bar(components, x="Stage", y="Components", color="Stage", title="Dimensionality reduction", text="Components", color_discrete_sequence=["#6489f5","#38d6c2"]), 360)
    st.markdown('<div class="panel"><b>Interpretation</b><p style="color:#aabbd2">PCA reduced the processed feature space from 26 components to 14 while retaining 95.85% of the variance. On this run, classification accuracy decreased from 82.22% to 81.37%, so PCA reduced dimensionality but did not improve the score.</p><p style="color:#aabbd2">Regression metrics: MAE = 3.645 lakh, RMSE = 5.618 lakh, R² = 0.73. Results are from the current experiment runs and may change if the data or training split changes.</p></div>', unsafe_allow_html=True)
    st.dataframe(metrics, use_container_width=True, hide_index=True)

elif page == "Statistical Analysis":
    hero("STATISTICAL INSIGHTS", "What the data tells us", "Descriptive statistics and tests reveal patterns in the used-car inventory.")
    a,b,c,d = st.columns(4)
    a.metric("Mean price", money(df["Price"].mean()))
    b.metric("Price variance", f"{df['Price'].var():,.2f}")
    b.caption("₹ lakh squared")
    c.metric("Median mileage", f"{df['Mileage'].median():.2f}")
    d.metric("Median model year", f"{int(df['Year'].median())}")
    corr_cols = ["Price","Year","Kilometers_Driven","Mileage","Engine","Power"]
    corr = df[corr_cols].corr()
    left,right = st.columns(2)
    with left:
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=["#1e3a5f","#17243a","#38d6c2"], title="Correlation matrix")
        chart(fig, 420)
    with right:
        fuel_avg = df.groupby("Fuel_Type", as_index=False)["Price"].mean().sort_values("Price", ascending=False)
        chart(px.bar(fuel_avg, x="Fuel_Type", y="Price", color="Fuel_Type", title="Average price by fuel type", labels={"Fuel_Type":"Fuel type","Price":"Average price (₹ lakh)"}), 420)
    petrol = df.loc[df["Fuel_Type"] == "Petrol", "Price"].dropna()
    diesel = df.loc[df["Fuel_Type"] == "Diesel", "Price"].dropna()
    if len(petrol) and len(diesel):
        t_stat, p_value = ttest_ind(petrol, diesel, equal_var=False)
        st.markdown('<div class="section-head">Welch independent t-test · Petrol vs Diesel</div>', unsafe_allow_html=True)
        x,y,z = st.columns(3)
        x.metric("T-statistic", f"{t_stat:.3f}")
        y.metric("P-value", f"{p_value:.3e}")
        z.metric("Significance level", "0.05")
        if p_value < .05:
            st.success("The average prices differ significantly at the 5% significance level in this dataset.")
        else:
            st.info("The test did not detect a statistically significant difference at the 5% level.")
    st.markdown('<div class="section-head">Descriptive statistics</div>', unsafe_allow_html=True)
    st.dataframe(df[corr_cols].describe().round(3), use_container_width=True)

st.markdown('<div style="border-top:1px solid rgba(148,163,184,.14);margin-top:35px;padding-top:18px;color:#7186a4;font-size:.78rem">AUTOINSIGHT · USED CAR INTELLIGENCE &nbsp; · &nbsp; Built with Streamlit, scikit-learn and Plotly</div>', unsafe_allow_html=True)
