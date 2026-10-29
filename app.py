"""NY Clean School Zones Monitor - Streamlit dashboard (ST2IBI project).

Structure = Simon's decision-making process: each KPI belongs to one phase.
Run:  pip install -r requirements.txt  &&  python -m streamlit run app.py
Data: WRI Electric School Bus Adoption Dataset v10 (through 31 Dec 2025), New York extract produced by etl.py;
      CARB Table 2 (cancer risk from idling diesel school buses, by distance), typed in below.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NY Clean School Zones Monitor", page_icon="🚌", layout="wide")

WHO = 5.0  # WHO 2021 annual PM2.5 guideline, µg/m³
REPORT = {"k1": 92.09, "k2": 99.97, "k3": 46595}  # figures quoted in the team's written report
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
sel_q = st.sidebar.multiselect("PM2.5 quartile (national)", [1, 2, 3, 4], default=[1, 2, 3, 4],
                               help="1 = cleanest quarter of US districts, 4 = most polluted.")
only_pri = st.sidebar.toggle("Priority districts only", value=False,
                             help="PM2.5 at or above the New York 75th percentile (7.2 µg/m³).")
k2_def = st.sidebar.radio("KPI 2 counts as electric", ["Operating", "Committed"], horizontal=True)
show = st.sidebar.radio("Values shown", ["Computed from the dataset", "As written in the team report"],
                        index=1, help="Report figures: KPI 2 and KPI 3 come from the fleet state summary of 10 May 2024 (15 electric of 46,595); the computed ones come from WRI v10.")

st.sidebar.divider()
st.sidebar.subheader("KPI 4 – zones recorded")
st.sidebar.caption("Not in any dataset: the team records it each term.")
if "zones" not in st.session_state:
    st.session_state.zones = pd.DataFrame({"Term": ["Term 1", "Term 2", "Term 3"], "Targeted": [0, 0, 0], "With zone": [0, 0, 0]})
zones = st.sidebar.data_editor(st.session_state.zones, num_rows="dynamic", hide_index=True, width="stretch",
                               key="zones_editor")

df = districts[districts.locale.isin(sel_loc)].copy()
if len(sel_q) < 4:
    df = df[df.q_pm25.isin(sel_q)]
if only_pri:
    df = df[df.priority_zone]
bus = buses[buses.lea_id.isin(df.lea_id)]

# ---------------------------------------------------------------- KPIs
known = df.dropna(subset=["buses_total"])
fleet = known.buses_total.sum()
kq = known.dropna(subset=["q_pm25"])
kq_fleet = kq.buses_total.sum()
k1 = kq[kq.q_pm25.isin([2, 3])].buses_total.sum() / kq_fleet * 100 if kq_fleet else np.nan
elec_col = "esb_operating" if k2_def == "Operating" else "esb_committed"
k2 = (1 - known[elec_col].sum() / fleet) * 100 if fleet else np.nan
zl = zones.dropna().tail(1)
k4 = np.nan
if len(zl) and zl["Targeted"].iloc[0] > 0:
    k4 = zl["With zone"].iloc[0] / zl["Targeted"].iloc[0] * 100
exp = df.dropna(subset=["pm25", "students"])
exp = exp[exp.students > 0]
who_ratio = (exp.pm25 * exp.students).sum() / exp.students.sum() / WHO if len(exp) else np.nan

use_rep = show.startswith("As written")
v1 = REPORT["k1"] if use_rep else k1
v2 = REPORT["k2"] if use_rep else k2
v3 = float(fleet_sum["total"])  # New York total (state summary, 10 May 2024)


def pct(v):
    return "–" if pd.isna(v) else f"{v:.2f}%"


tiles = [
    (1, "Buses in Q2–Q3", pct(v1)),
    (2, "Buses not electric", pct(v2)),
    (3, "Fleet size", f"{v3:,.0f}"),
    (4, "Zone coverage", "–" if pd.isna(k4) else f"{k4:.1f}%"),
]

# ---------------------------------------------------------------- header
st.title("NY Clean School Zones Monitor")
st.markdown("**Problem:** children in New York State are exposed to harmful emissions from school transportation. "
            "**Chosen action:** Clean School Zones around schools. Each KPI below belongs to one phase of **Simon's decision-making process**.")
st.caption(f"WRI Electric School Bus Adoption Dataset v10 (through 31 Dec 2025) · School bus fleet state summary v3 (10 May 2024) · CARB idling-risk table (Table 3) · "
           f"{len(df):,} of {len(districts):,} districts selected · values: {show.lower()}")

cols = st.columns(4)
for c, (n, name, val) in zip(cols, tiles):
    with c.container(border=True):
        st.caption(f"PHASE {n} · {PHASES[n].upper()}")
        st.metric(f"KPI {n} · {name}", val)
st.caption("Intelligence → Design → Choice → Implementation (then monitoring feeds back into Intelligence).")


def notes(why, represents, improve):
    st.markdown(f"**Why this KPI**  \n{why}")
    st.markdown(f"**What it represents**  \n{represents}")
    st.markdown(f"**What we must improve**  \n{improve}")


# ---------------------------------------------------------------- Phase 1
with st.container(border=True):
    st.subheader("Phase 1 · Intelligence — KPI 1: Buses in PM2.5 quartiles Q2–Q3")
    st.caption("Intelligence = scan the environment and confirm that the problem exists and how widespread it is.")
    a, b = st.columns([3, 2])
    with a:
        if kq_fleet:
            q = kq.groupby("q_pm25").buses_total.sum().reindex([1, 2, 3, 4], fill_value=0)
            sh = q / q.sum() * 100
            fig = go.Figure(go.Bar(x=[f"Q{i}" for i in q.index], y=sh.round(2),
                                   marker_color=[ACCENT if i in (2, 3) else MUTED for i in q.index],
                                   text=sh.round(1).astype(str) + "%", textposition="outside",
                                   hovertemplate="%{x}: %{y:.2f}% of buses<extra></extra>"))
            fig.update_layout(height=320, margin=dict(t=20, b=10), yaxis_title="% of buses", yaxis_range=[0, 110],
                              xaxis_title="National PM2.5 quartile of the district (1 = cleanest, 4 = most polluted)")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Computed on {kq_fleet:,.0f} buses in districts with a fleet size and a PM2.5 value. "
                       + (f"Student-weighted PM2.5 of the selection: {who_ratio*WHO:.2f} µg/m³, i.e. {who_ratio:.2f}× the WHO guideline (5 µg/m³)."
                          if not pd.isna(who_ratio) else ""))
        else:
            st.info("No district in the selection has both a fleet size and a PM2.5 value.")
    with b:
        notes("It is the first step of Simon's process: before choosing an action we must show that school buses really "
              "operate in polluted air, and where.",
              "The share of the school bus fleet located in districts whose PM2.5 falls in the middle national quartiles (Q2 and Q3).",
              "Almost all buses run in air that is neither clean nor extreme, so exposure is widespread rather than local. "
              "We need a clear threshold (the WHO guideline) and a list of the districts to prioritise.")

# ---------------------------------------------------------------- Phase 2
with st.container(border=True):
    st.subheader("Phase 2 · Design — KPI 2: Buses not electric")
    st.caption("Design = develop and compare possible courses of action. First option: replace the fleet with electric buses.")
    a, b = st.columns([3, 2])
    with a:
        fuels = fleet_sum[[i for i in fleet_sum.index if i.startswith("fuel:")]]
        fuels.index = [i[5:].replace(" (all types)", "").replace(" (not otherwise specified)", "") for i in fuels.index]
        fuels = fuels[fuels > 0].sort_values()
        fsh = fuels / fleet_sum["total"] * 100
        fig = go.Figure(go.Bar(y=fsh.index, x=fsh.values, orientation="h",
                               marker_color=[ACCENT if i == "Electric" else MUTED for i in fsh.index],
                               text=[f"{v:.2f}% ({int(n):,})" for v, n in zip(fsh.values, fuels.values)], textposition="outside",
                               hovertemplate="%{y}: %{x:.2f}% of buses<extra></extra>"))
        fig.update_layout(height=320, margin=dict(t=10, b=10, r=60), xaxis_title="Share of New York school buses by fuel (%)",
                          xaxis_range=[0, 75])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Fuel mix of {int(fleet_sum['total']):,} New York school buses, state summary of 10 May 2024: "
                   f"{int(fleet_sum['fuel:Electric'])} electric. KPI 2 = 1 − electric ÷ total = {(1 - fleet_sum['fuel:Electric'] / fleet_sum['total']) * 100:.2f}%.")
    with b:
        notes("Replacing the fleet is the obvious alternative, so we measure how far it has gone. If almost no bus is electric, "
              "this option cannot protect children soon.",
              "The share of school buses that are not electric (running, or committed, depending on the sidebar choice).",
              "The 2024 fleet summary counts 15 electric buses out of 46,595 (diesel 56.8%, gasoline 41.7%). "
              "The transition is too slow for an association to rely on, so we design measures that work with the buses already on the road.")

# ---------------------------------------------------------------- Phase 3
with st.container(border=True):
    st.subheader("Phase 3 · Choice — KPI 3: Fleet size")
    st.caption("Choice = select the best alternative. The fleet is too big to replace, so we choose to act around schools.")
    a, b = st.columns([3, 2])
    with a:
        ages = fleet_sum[[i for i in fleet_sum.index if i.startswith("age:")]]
        ages.index = [i[4:] for i in ages.index]
        ages = ages.reindex(["2020 or newer", "2010-2019", "2000-2009", "1999 or older"])
        ash = ages / ages.sum() * 100
        fig = go.Figure(go.Bar(x=ages.index, y=ages.values, marker_color=MUTED,
                               text=[f"{int(n):,} ({v:.0f}%)" for n, v in zip(ages.values, ash.values)], textposition="outside",
                               hovertemplate="%{x}: %{y:,.0f} buses<extra></extra>"))
        fig.update_layout(height=300, margin=dict(t=20, b=10), yaxis_title="School buses",
                          xaxis_title="Bus model year", yaxis_range=[0, ages.max() * 1.2])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{int(fleet_sum['total']):,} school buses in New York (state summary, 10 May 2024), most of them built in 2010–2019. "
                   f"In the current filter, {fleet:,.0f} buses are in districts that report their fleet size.")
        top = exp.sort_values("exposure", ascending=False).head(5)[["lea_name", "locale", "students", "pm25"]]
        st.markdown("**Where to start (highest exposure score = PM2.5 × students)**")
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            "lea_name": "District", "locale": "Locale", "students": st.column_config.NumberColumn("Students", format="%d"),
            "pm25": st.column_config.NumberColumn("PM2.5 (µg/m³)", format="%.2f")})
    with b:
        notes("Choosing means looking at what the association can really do. With a fleet this large we cannot replace buses, "
              "so we choose Clean School Zones, which act on where children stand rather than on the whole fleet.",
              "The total number of school buses in New York: the scale of the system we cannot change alone.",
              "We must decide which schools come first, starting with the districts that expose the most students, "
              "and accept that the fleet itself will change slowly.")

# ---------------------------------------------------------------- Phase 4
with st.container(border=True):
    st.subheader("Phase 4 · Implementation — KPI 4: Clean School Zone coverage")
    st.caption("Implementation = put the chosen solution into practice and monitor it. The distance of the zone comes from the CARB risk table.")
    a, b = st.columns([3, 2])
    with a:
        st.markdown("**At what distance should the zone start?**")
        r1, r2 = st.columns(2)
        who = r1.radio("Who is exposed", ["Children", "Teachers", "Residents"], horizontal=True)
        idle = r2.radio("Idling time per bus per day", [20, 2], format_func=lambda m: f"{m} min", horizontal=True)
        fig = go.Figure()
        for nb, col in zip([1, 5, 10], SHADES):
            vals = CARB[idle][nb][who]
            fig.add_bar(name=f"{nb} bus" + ("es" if nb > 1 else ""), x=[f"{d} m" for d in DIST],
                        y=[carb_high(v) for v in vals], marker_color=col,
                        text=[("<1" if v == "<1" else carb_label(v)) for v in vals], textposition="outside",
                        customdata=[carb_label(v) for v in vals],
                        hovertemplate=f"{who}, %{{x}}, {nb} bus(es): %{{customdata}} per million<extra></extra>")
        fig.update_layout(barmode="group", height=320, margin=dict(t=20, b=10), legend_title="Idling buses",
                          yaxis_title="Potential cancer risk (per million)", xaxis_title="Distance from the idling buses",
                          yaxis_range=[0, max(3.0, max(carb_high(v) for nb in (1, 5, 10) for v in CARB[idle][nb][who])) * 1.35])
        st.plotly_chart(fig, width="stretch")

        def below_one(nb):
            for d, v in zip(DIST, CARB[idle][nb][who]):
                if v == "<1":
                    return d
            return None

        lines = []
        for nb in (1, 5, 10):
            d = below_one(nb)
            lines.append(f"{nb} bus{'es' if nb > 1 else ''}: " + (f"below 1 per million from {d} m" if d else "not below 1 per million even at 60 m"))
        st.info(f"**{who}, {idle} min idling per bus per day** — " + " · ".join(lines))
        st.caption("Source: CARB, Table 3, potential cancer risks per million from idling diesel school buses (Sacramento meteorological data). "
                   "Up to 20 min/day, 180 days/year; exposure of 9 years for children, 40 for teachers, 70 for residents. "
                   "Bars show the upper end of each range; “<1” is drawn at 0.5.")
        st.markdown("**KPI 4 over time**")
        zz = zones.dropna().copy()
        zz["Coverage"] = np.where(zz["Targeted"] > 0, zz["With zone"] / zz["Targeted"].replace(0, np.nan) * 100, np.nan)
        if zz["Coverage"].notna().any():
            fig = go.Figure(go.Bar(x=zz.Term, y=zz.Coverage.round(1), marker_color=ACCENT,
                                   text=zz.Coverage.round(0).astype("Int64").astype(str) + "%", textposition="outside",
                                   hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
            fig.update_layout(height=260, margin=dict(t=20, b=10), yaxis_title="Schools with a zone ÷ targeted schools (%)",
                              yaxis_range=[0, 110])
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No zone recorded yet. Enter, for each term, the targeted schools and the schools with a zone "
                    "in the sidebar (KPI 4 – zones recorded). The chart appears here.")
    with b:
        notes("It shows whether the chosen action is really being applied, and lets us adjust the plan each school term. "
              "The CARB table tells us how large the zone must be.",
              "The share of targeted schools where a Clean School Zone is in place, with the zone distance taken from the risk table.",
              "For children, the risk is at its highest within 20 m of idling buses and falls below 1 per million at 40 m; "
              "residents and teachers still see a risk at 40 m, so 60 m gives a margin. We must agree the distance and a target "
              "with schools and local authorities, then raise coverage term after term and check that KPI 1 to KPI 3 improve.")

st.divider()
st.caption("Sources: WRI Electric School Bus Adoption Dataset v10 (doi 10.46830/writn.21.00134.v10); Dataset of U.S. School Bus Fleets, State Summary v3 (2024-05-10); "
           "California Air Resources Board (CARB), Table 3, potential cancer risks from idling diesel school buses, Sacramento meteorological data "
           "(document title and year to complete); WHO global air quality guidelines 2021. KPI 4 is entered by the team. "
           "Charts and text prepared with the help of an AI assistant; the team must check and declare this use.")
