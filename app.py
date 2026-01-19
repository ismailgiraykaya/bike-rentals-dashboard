import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="DC Bike Rentals Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour

    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    df["season_name"] = df["season"].map(season_map)

    bins = [0, 6, 12, 18, 24]
    labels = ["night", "morning", "afternoon", "evening"]
    df["day_period"] = pd.cut(df["hour"], bins=bins, labels=labels, right=False)

    return df

st.title("Washington D.C. Bike Rentals (2011–2012)")
st.caption("Interactive Streamlit Dashboard")

df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")

year_sel = st.sidebar.multiselect("Year", df["year"].unique(), default=df["year"].unique())
season_sel = st.sidebar.multiselect("Season", df["season_name"].unique(), default=df["season_name"].unique())
working_sel = st.sidebar.radio("Working day", ["All", "Working", "Non-working"])
weather_sel = st.sidebar.multiselect("Weather", df["weather"].unique(), default=df["weather"].unique())

f = df[
    (df["year"].isin(year_sel)) &
    (df["season_name"].isin(season_sel)) &
    (df["weather"].isin(weather_sel))
]

if working_sel == "Working":
    f = f[f["workingday"] == 1]
elif working_sel == "Non-working":
    f = f[f["workingday"] == 0]

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(f))
c2.metric("Mean Rentals / Hour", round(f["count"].mean(), 1))
c3.metric("Total Rentals", int(f["count"].sum()))
c4.metric("Registered %", round(100 * f["registered"].sum() / f["count"].sum(), 1))

st.divider()

# 1) Hourly pattern
fig1 = px.line(f.groupby("hour")["count"].mean().reset_index(),
               x="hour", y="count", title="Mean Rentals by Hour")
st.plotly_chart(fig1, use_container_width=True)

# 2) Monthly trend by year
fig2 = px.line(f.groupby(["year", "month"])["count"].mean().reset_index(),
               x="month", y="count", color="year", title="Monthly Trend by Year")
st.plotly_chart(fig2, use_container_width=True)

# 3) Working vs Non-working
fig3 = px.bar(f.groupby("workingday")["count"].mean().reset_index(),
              x="workingday", y="count", title="Working vs Non-working Mean Rentals")
st.plotly_chart(fig3, use_container_width=True)

# 4) Weather with 95% CI (robust)
wx = (
    f.groupby("weather")["count"]
    .agg(mean="mean", n="size", std="std")
    .reset_index()
)

# SEM and 95% CI
wx["sem"] = wx["std"] / np.sqrt(wx["n"].clip(lower=1))
wx["ci"] = 1.96 * wx["sem"]
wx["ci"] = wx["ci"].fillna(0)  # küçük gruplarda NaN olursa sıfır yap

fig4 = px.bar(wx, x="weather", y="mean", error_y="ci", title="Weather Effect (95% CI)")
st.plotly_chart(fig4, use_container_width=True)
