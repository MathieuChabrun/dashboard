"""NY Clean School Zones Monitor - Streamlit dashboard (ST2IBI project).

Structure = Simon's decision-making process: each KPI belongs to one phase.
Run:  pip install -r requirements.txt  &&  python -m streamlit run app.py
Data: WRI Electric School Bus Adoption Dataset v10 (through 31 Dec 2025), New York extract produced by etl.py;
      CARB Table 3 (cancer risk from idling diesel school buses, by distance), typed in below.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NY Clean School Zones Monitor", page_icon="🚌", layout="wide")

WHO = 5.0  # WHO 2021 annual PM2.5 guideline, µg/m³
NY_FLEET = 46595
ACCENT, MUTED, WARN = "#1f6f5c", "#9bb5ad", "#c2410c"
SHADES = ["#a9cfc0", "#5ea58f", "#124d3f"]  # 1, 5, 10 buses (ordered, one hue)
PHASES = {1: "Intelligence", 2: "Design", 3: "Choice", 4: "Implementation"}

DATA = Path(__file__).parent / "data"


@st.cache_data
def load():
    d = pd.read_csv(DATA / "ny_districts.csv", dtype={"lea_id": str})
    b = pd.read_csv(DATA / "ny_buses.csv", dtype={"lea_id": str})
    return d, b


districts, buses = load()
fleet_sum = pd.read_csv(DATA / "ny_fleet_summary.csv").set_index("item")["value"]  # "Dataset of U.S. School Bus Fleets, State Summary v3" (10 May 2024)

# CARB Table 3 - potential cancer risk (per million) from idling diesel school buses (Sacramento meteorological data).
# CARB[idling minutes per bus per day][number of buses][receptor] = values at 20 m, 40 m, 60 m ("low-high", "<1" = under 1)
_L = ["<1", "<1", "<1"]
CARB = {
    2: {n: {"Residents": _L, "Teachers": _L, "Children": _L} for n in (1, 5, 10)},
    20: {1: {"Residents": ["1-1", "<1", "<1"], "Teachers": ["1-1", "<1", "<1"], "Children": ["<1", "<1", "<1"]},
         5: {"Residents": ["3-4", "1-2", "1-1"], "Teachers": ["2-2", "1-1", "<1"], "Children": ["1-1", "<1", "<1"]},
         10: {"Residents": ["3-4", "1-2", "1-1"], "Teachers": ["2-3", "1-1", "<1-1"], "Children": ["1-1", "<1", "<1"]}}}
DIST = [20, 40, 60]


def carb_high(v):
    """Upper end of a CARB cell, for plotting ('<1' is drawn at 0.5)."""
    if v == "<1":
        return 0.5
    return float(v.split("-")[-1])


def carb_label(v):
    return v.replace("-", "–") if v.split("-")[0] != v.split("-")[-1] or v.startswith("<") else v.split("-")[0]


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Filters")
locales = sorted(districts.locale.dropna().unique())
sel_loc = st.sidebar.multiselect("Locale", locales, default=locales)

df = districts[districts.locale.isin(sel_loc)].copy()
bus = buses[buses.lea_id.isin(df.lea_id)]

# ---------------------------------------------------------------- KPIs
known = df.dropna(subset=["buses_total"])
fleet = known.buses_total.sum()
kq = known.dropna(subset=["q_pm25"])
kq_fleet = kq.buses_total.sum()
k1 = kq[kq.q_pm25.isin([2, 3])].buses_total.sum() / kq_fleet * 100 if kq_fleet else np.nan
k2 = (1 - fleet_sum["fuel:Electric"] / fleet_sum["total"]) * 100
exp = df.dropna(subset=["pm25", "students"])
exp = exp[exp.students > 0]
who_ratio = (exp.pm25 * exp.students).sum() / exp.students.sum() / WHO if len(exp) else np.nan

v1 = k1
v2 = k2
v3 = float(fleet_sum["total"])  # New York total (state summary, 10 May 2024)


def pct(v):
    return "–" if pd.isna(v) else f"{v:.2f}%"


tiles = [
    (1, "Buses in Q2–Q3", pct(v1)),
    (2, "Fleet size", f"{v3:,.0f}"),
    (3, "Buses not electric", pct(v2)),
]

# ---------------------------------------------------------------- pages
st.title("NY Clean School Zones")
page = st.sidebar.radio("Page", list(PHASES), format_func=lambda n: f"{n}. {PHASES[n]}")
if page != 4:
    st.caption(f"{len(df):,} districts selected · New York State")
else:
    st.caption("CARB · Sacramento meteorological conditions · independent of New York filters")
st.subheader(f"{PHASES[page]}")
if page != 4:
    _, name, value = tiles[page - 1]
    st.metric(name, value)
else:
    st.markdown("**Potential cancer risk by distance from idling buses**")


def chart(fig, selectable=False):
    fig.update_layout(
        height=400, margin=dict(t=35, b=20, l=20, r=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b"), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)", zeroline=False)
    if selectable:
        return st.plotly_chart(
            fig, width="stretch", config={"displayModeBar": False},
            key="quartile_chart", on_select="rerun", selection_mode="points",
        )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


if page == 1:
    if kq_fleet:
        q = kq.groupby("q_pm25").buses_total.sum().reindex([1, 2, 3, 4], fill_value=0)
        shares = q / q.sum() * 100
        fig = go.Figure(go.Bar(
            x=[f"Q{i}" for i in q.index], y=shares,
            marker_color=[ACCENT if i in (2, 3) else MUTED for i in q.index],
            text=[f"{v:.1f}%" for v in shares], textposition="outside",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(yaxis_title="Share of buses (%)", yaxis_range=[0, 110],
                          xaxis_title="National PM2.5 quartile")
        event = chart(fig, selectable=True)
        st.caption("Share of buses by district air pollution quartile. Q2 and Q3 are highlighted. "
                   "The chart and KPI follow your filters.")
        points = event["selection"]["points"]
        if points and points[-1].get("x") in ("Q1", "Q2", "Q3", "Q4"):
            selected_q = int(points[-1]["x"][1])
            ranked = df[df.q_pm25.eq(selected_q)].sort_values(
                ["pm25", "students", "lea_name"], ascending=[False, False, True],
                na_position="last",
            )
            st.markdown(f"**Q{selected_q} · districts ranked by PM2.5**")
            st.caption("Highest PM2.5 first. This ranks air pollution, not individual health risk.")
            if ranked.empty:
                st.caption("No district in this quartile matches the current filters.")
            else:
                st.dataframe(
                    ranked[["lea_name", "pm25", "students", "buses_total"]],
                    hide_index=True, width="stretch", column_config={
                        "lea_name": "District",
                        "pm25": st.column_config.NumberColumn("PM2.5 (µg/m³)", format="%.2f"),
                        "students": st.column_config.NumberColumn("Students", format="%d"),
                        "buses_total": st.column_config.NumberColumn("Buses", format="%d"),
                    },
                )
        else:
            st.caption("Click a quartile bar to see its districts, ranked by air pollution.")
    else:
        st.caption("No fleet and air pollution data available for this selection.")

elif page == 3:
    fuels = fleet_sum[[i for i in fleet_sum.index if i.startswith("fuel:")]].copy()
    fuels.index = [i[5:].replace(" (all types)", "").replace(" (not otherwise specified)", "") for i in fuels.index]
    fuels = fuels[fuels > 0].sort_values()
    shares = fuels / fleet_sum["total"] * 100
    comment = "New York fleet by fuel type: 15 electric buses out of 46,595. Source: state fleet summary, May 2024. Statewide figures are independent of district filters."
    fig = go.Figure(go.Bar(
        y=shares.index, x=shares.values, orientation="h",
        marker_color=[ACCENT if i == "Electric" else MUTED for i in shares.index],
        text=[f"{v:.2f}%" for v in shares], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Share of buses (%)", xaxis_range=[0, 110])
    chart(fig)
    st.caption(comment)

elif page == 2:
    ages = fleet_sum[[i for i in fleet_sum.index if i.startswith("age:")]].copy()
    ages.index = [i[4:] for i in ages.index]
    ages = ages.reindex(["2020 or newer", "2010-2019", "2000-2009", "1999 or older"])
    fig = go.Figure(go.Bar(
        x=ages.index, y=ages.values, marker_color=ACCENT,
        text=[f"{v:,.0f}" for v in ages], textposition="outside",
        hovertemplate="%{x}: %{y:,.0f} buses<extra></extra>",
    ))
    fig.update_layout(yaxis_title="School buses", xaxis_title="Model year",
                      yaxis_range=[0, ages.max() * 1.2])
    chart(fig)
    st.caption("New York has 46,595 school buses. The chart shows their model years. "
               "Source: state fleet summary, May 2024; statewide figures are independent of district filters.")

else:
    who = st.selectbox("Who is exposed", ["Children", "Teachers", "Residents"])
    idle = st.radio("Idling time per bus per day", [20, 2],
                    format_func=lambda m: f"{m} min", horizontal=True)
    fig = go.Figure()
    for nb, color in zip([1, 5, 10], SHADES):
        vals = CARB[idle][nb][who]
        fig.add_bar(
            name=f"{nb} bus" + ("es" if nb > 1 else ""),
            x=[f"{d} m" for d in DIST], y=[carb_high(v) for v in vals],
            marker_color=color, text=[carb_label(v) for v in vals],
            textposition="outside", customdata=[carb_label(v) for v in vals],
            hovertemplate="%{x}: %{customdata} per million<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(barmode="group", xaxis_title="Distance from idling buses",
                      yaxis_title="Potential cancer risk (per million)",
                      yaxis_range=[0, max(3, max(carb_high(v) for nb in (1, 5, 10)
                                                for v in CARB[idle][nb][who])) * 1.35])
    # Keep the bus counts identifiable while sharing the minimal chart styling.
    for trace in fig.data:
        trace.text = [f"{trace.name}<br>{v}" for v in trace.text]
    chart(fig)
    st.caption("Potential cancer risk by distance for 1, 5 and 10 idling buses. "
               "Source: CARB, Table 3 (Sacramento conditions). Bars show the upper end "
               "of each range; <1 is drawn at 0.5 for display only. "
               "These are Sacramento scenarios, not estimates for New York schools.")
