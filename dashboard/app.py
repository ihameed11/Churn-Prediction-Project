"""Retention decision dashboard (Stage 6).

Run from the project folder:
    python dashboard/export_data.py      # once, after the notebooks
    streamlit run dashboard/app.py

The app only reads the files in dashboard/data (no model is loaded).
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
SEGMENTS = ["Low-value", "New high-value", "Established high-value"]
SEG_COLOURS = ["#2a78d6", "#1baf7a", "#4a3aa7"]
SEG_SCALE = alt.Scale(domain=SEGMENTS, range=SEG_COLOURS)
SEG_ICON = {"Low-value": "\U0001F7E6", "New high-value": "\U0001F7E9", "Established high-value": "\U0001F7EA"}

st.set_page_config(page_title="Churn Retention Dashboard", page_icon="\U0001F4CA", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2.2rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricValue"] {font-size: 1.3rem;}
[data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #57606a;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    cust = pd.read_csv(DATA / "customers.csv")
    kpis = pd.read_csv(DATA / "segment_kpis.csv", index_col="segment")
    val = pd.read_csv(DATA / "kpi_validation.csv", index_col="segment")
    play = pd.read_csv(DATA / "playbook.csv")
    play_b = pd.read_csv(DATA / "playbook_plan_b.csv")
    assumptions = json.load(open(DATA / "assumptions.json"))
    return cust, kpis, val, play, play_b, assumptions


cust, kpis, val, play, play_b, A = load()


def money(x):
    return f"{x:,.0f}"


def cohort_note():
    st.caption("ⓘ Historical test cohort: 1,407 customers the model never saw during training. "
               "Values are in the units of the monthly charge; revenue is annual (12 months).")


# ------------------------------------------------------------- page 1
def page_overview():
    st.title("Segment KPIs")
    st.markdown("Where is revenue at risk, whom should we contact, and what will the campaign earn?")
    cohort_note()

    tot = kpis.loc["Total"]
    c = st.columns(5)
    c[0].metric("Revenue at risk", money(tot["RaR estimated"]), help="Expected annual revenue lost to churn: sum of churn probability × annual value.")
    c[1].metric(f"Customers to contact ({tot['contacted'] / tot['customers']:.0%})", f"{int(tot['contacted']):,}")
    c[2].metric("Campaign budget", money(tot["budget"]), help="Worst case: every contacted customer takes the offer (offer + contact cost).")
    c[3].metric("Expected campaign profit", money(tot["profit estimated"]))
    c[4].metric("Return on budget", f"{tot['profit estimated'] / tot['budget']:.1f}×")

    st.subheader("By segment")
    cols = st.columns(3)
    for col, s in zip(cols, SEGMENTS):
        r = kpis.loc[s]
        with col.container(border=True):
            st.markdown(f"#### {SEG_ICON[s]} {s}")
            a, b = st.columns(2)
            a.metric("Customers", f"{int(r['customers']):,}")
            b.metric("Avg. churn risk", f"{r['churners estimated'] / r['customers']:.0%}")
            a.metric("Revenue at risk", money(r["RaR estimated"]))
            b.metric("Share of total", f"{r['RaR estimated'] / tot['RaR estimated']:.0%}")
            a.metric("To contact", f"{int(r['contacted']):,}")
            b.metric("Contact threshold", f"{r['threshold']:.2f}")
            a.metric("Expected profit", money(r["profit estimated"]))
            b.metric("Return on budget", f"{r['profit estimated'] / r['budget']:.1f}×")

    st.subheader("Can these numbers be trusted?")
    df = pd.DataFrame({"segment": SEGMENTS * 2,
                       "type": ["Estimated by the model"] * 3 + ["Actually lost (observed)"] * 3,
                       "revenue": list(kpis.loc[SEGMENTS, "RaR estimated"]) + list(kpis.loc[SEGMENTS, "RaR realised"])})
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("segment:N", sort=SEGMENTS, title=None, axis=alt.Axis(labelAngle=0, labelLimit=250, labelFontSize=12)),
        xOffset=alt.XOffset("type:N", sort=["Estimated by the model", "Actually lost (observed)"]),
        y=alt.Y("revenue:Q", title="Annual revenue at risk"),
        color=alt.Color("type:N", scale=alt.Scale(domain=["Estimated by the model", "Actually lost (observed)"], range=["#eda100", "#3d3d3a"]),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
        tooltip=["segment", "type", alt.Tooltip("revenue:Q", format=",.0f")],
    ).properties(height=300)
    left, right = st.columns([1, 1], vertical_alignment="center")
    left.altair_chart(chart, use_container_width=True)
    with right:
        err = val.loc[SEGMENTS + ["Total"], ["realised RaR", "primary: estimated RaR", "primary: relative error %", "primary: 95% CI"]]
        err.columns = ["Observed", "Estimated", "Error %", "95% interval"]
        err.index.name = None
        st.dataframe(err, use_container_width=True,
                     column_config={"Observed": st.column_config.NumberColumn(format="%.0f"),
                                    "Estimated": st.column_config.NumberColumn(format="%.0f"),
                                    "Error %": st.column_config.NumberColumn(format="%+.1f%%")})
        st.info(f"Overall the estimate is within **{val.loc['Total', 'primary: relative error %']:+.1f}%** of the revenue "
                "actually lost. Every segment's 95% interval includes zero, but small segments (54 low-value churners) "
                "cannot detect errors below about ±25%.")


# ------------------------------------------------------------- page 2
def select(budget, ranking):
    if budget == "No limit":
        return cust["contact"].to_numpy()
    n = int(round(int(budget.rstrip("%")) / 100 * len(cust)))
    key = "expected_profit" if ranking.startswith("Expected") else "churn_probability"
    chosen = cust[key].rank(ascending=False, method="first") <= n
    return chosen.to_numpy()


def campaign(mask):
    g = cust[mask]
    f, d, gam = A["contact_cost_f"], A["incentive_d"], A["acceptance_rate"]
    realised = (g.churned * (gam * (g.annual_value - d) - f) - (1 - g.churned) * (d + f)).sum()
    return {"contacted": len(g), "budget": len(g) * (f + d), "expected": g.expected_profit.sum(),
            "realised": realised, "reached": g.churned.sum() / cust.churned.sum()}


def page_planner():
    st.title("Campaign planner")
    st.markdown("Choose how many customers the team can contact and how to rank them.")
    cohort_note()

    c1, c2 = st.columns([1, 1])
    budget = c1.select_slider("Contact budget (share of customers)", ["10%", "20%", "30%", "No limit"], value="20%")
    ranking = c2.radio("Rank customers by", ["Expected profit (recommended)", "Churn risk only"], horizontal=True,
                       disabled=budget == "No limit",
                       help="Expected profit uses each customer's value and the offer cost; churn risk ignores value.")
    if budget == "No limit":
        st.caption("No limit: everyone above their segment's contact threshold is contacted (the profit-optimal rule).")

    mask = select(budget, ranking)
    r = campaign(mask)
    m = st.columns(5)
    m[0].metric("Customers contacted", f"{r['contacted']:,}")
    m[1].metric("Campaign budget", money(r["budget"]))
    m[2].metric("Expected profit", money(r["expected"]))
    m[3].metric("Profit with observed churn", money(r["realised"]),
                help="Uses who actually churned, still assuming 30% of contacted churners accept the offer.")
    m[4].metric("Churners reached", f"{r['reached']:.0%}")

    st.subheader("How the budget is shared between segments")
    rows = []
    for rk in ["Expected profit (recommended)", "Churn risk only"]:
        mk = select(budget, rk)
        cc = campaign(mk)
        for s in SEGMENTS:
            rows.append({"ranking": rk.split(" (")[0], "segment": s,
                         "share": (cust.loc[mk, "segment"] == s).mean() if mk.sum() else 0,
                         "profit": cc["expected"]})
    df = pd.DataFrame(rows)
    left, right = st.columns([3, 2])
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y("ranking:N", title=None, axis=alt.Axis(labelFontSize=12, labelLimit=200)), x=alt.X("share:Q", axis=alt.Axis(format="%"), title="Share of contacted customers"),
        color=alt.Color("segment:N", scale=SEG_SCALE, sort=SEGMENTS, legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("segment:N"), tooltip=["ranking", "segment", alt.Tooltip("share:Q", format=".0%")],
    ).properties(height=alt.Step(42))
    left.altair_chart(chart, use_container_width=True)
    if budget != "No limit":
        comp = df.groupby("ranking")["profit"].first()
        gain = comp["Expected profit"] - comp["Churn risk only"]
        right.success(f"Ranking by **expected profit** earns **{money(gain)} more** than ranking by churn risk alone "
                      f"with a {budget} budget, by moving contacts towards high-value customers.")
    else:
        right.info("With no budget limit, the segment thresholds decide: contact a customer when expected profit is positive.")

    st.subheader("Target list")
    show = cust[mask].sort_values("expected_profit", ascending=False)[
        ["customerID", "segment", "churn_probability", "annual_value", "expected_profit", "action"]].copy()
    show["churn_probability"] = (100 * show["churn_probability"]).round(0)
    show["action"] = show["action"].fillna("– (no offer computed; see playbook)")
    st.dataframe(show, use_container_width=True, hide_index=True, height=320,
                 column_config={"customerID": "Customer", "segment": "Segment",
                                "churn_probability": st.column_config.ProgressColumn("Churn risk", format="%.0f%%", min_value=0, max_value=100),
                                "annual_value": st.column_config.NumberColumn("Annual value", format="%.0f"),
                                "expected_profit": st.column_config.NumberColumn("Expected profit", format="%.0f"),
                                "action": st.column_config.TextColumn("Suggested action", width="large")})
    st.download_button("⬇ Download target list (CSV)", show.to_csv(index=False), "target_list.csv", "text/csv")


# ------------------------------------------------------------- page 3
def page_customer():
    st.title("Customer view")
    st.markdown("Look up one customer: risk, decision and the suggested retention offer.")
    cohort_note()

    f1, f2 = st.columns(2)
    seg = f1.selectbox("Segment", ["All segments"] + SEGMENTS)
    only = f2.toggle("Only customers to contact", value=True)
    pool = cust if seg == "All segments" else cust[cust.segment == seg]
    if only:
        pool = pool[pool.contact]
    pool = pool.sort_values("expected_profit", ascending=False)
    has_offer = pool["action"].notna().to_numpy()
    start = int(has_offer.argmax()) if has_offer.any() else 0
    cid = st.selectbox(f"Customer ({len(pool):,} shown, highest expected profit first)", pool.customerID, index=start,
                       format_func=lambda c: f"{c}  –  risk {cust.set_index('customerID').loc[c, 'churn_probability']:.0%}")
    if cid is None:
        st.warning("No customers match the filters.")
        return
    r = cust.set_index("customerID").loc[cid]

    left, right = st.columns([1, 1])
    with left.container(border=True):
        st.markdown(f"#### {SEG_ICON[r.segment]} {cid}")
        st.markdown(f"**Segment:** {r.segment}  \n**Tenure:** {r.tenure} months  \n**Contract:** {r.contract}  \n"
                    f"**Internet:** {r.internet}  \n**Payment:** {r.payment}  \n"
                    f"**Monthly charge:** {r.monthly_charge:.2f}  ·  **Annual value:** {money(r.annual_value)}")
    with right.container(border=True):
        a, b = st.columns(2)
        a.metric("Churn risk", f"{r.churn_probability:.0%}")
        b.metric("Segment threshold", f"{r.threshold:.0%}")
        X = alt.Scale(domain=[0, 1])
        track = alt.Chart(pd.DataFrame({"v": [1.0]})).mark_bar(color="#eceef1", height=20).encode(
            x=alt.X("v:Q", scale=X, axis=alt.Axis(format="%", title=None)))
        fill = alt.Chart(pd.DataFrame({"v": [r.churn_probability]})).mark_bar(
            color="#d6402a" if r.contact else "#1baf7a", height=20).encode(x=alt.X("v:Q", scale=X))
        tick = alt.Chart(pd.DataFrame({"v": [r.threshold]})).mark_tick(color="black", thickness=3, size=34).encode(x=alt.X("v:Q", scale=X))
        lab = alt.Chart(pd.DataFrame({"v": [r.threshold], "t": ["threshold"]})).mark_text(
            dy=-24, fontSize=11, color="black").encode(x=alt.X("v:Q", scale=X), text="t:N")
        st.altair_chart((track + fill + tick + lab).properties(height=70), use_container_width=True)
        st.metric("Expected profit of contacting", money(r.expected_profit))
        if r.contact:
            st.error("**Decision: contact this customer** (risk above the segment threshold)")
        else:
            st.success("**Decision: no contact needed** (risk below the segment threshold)")

    st.subheader("Suggested retention offer")
    if isinstance(r.action, str) and r.action:
        with st.container(border=True):
            st.markdown(f"### ✅ {r.action}")
            a, b, c = st.columns(3)
            a.metric("Churn risk after the offer", f"{r.prob_after_action:.0%}", f"{(r.prob_after_action - r.churn_probability) * 100:+.0f} points", delta_color="inverse")
            b.metric("12-month cost of the offer", money(r.action_cost_12m))
            c.metric("Net value", money(r.action_net_value), help="Drop in churn risk × annual value − 12-month cost.")
            st.caption("Model-based suggestion, not a guaranteed effect: test it in an A/B campaign before rolling it out.")
    elif r.contact:
        st.info("No counterfactual was computed for this customer (offers were generated for a sample of 295 contacted "
                "customers). See the segment playbook for the usual offer in this segment.")
    else:
        st.write("No offer needed for customers below the threshold.")
    with st.expander("What actually happened (historical outcome)"):
        st.write("This customer **churned**." if r.churned else "This customer **stayed**.")


# ------------------------------------------------------------- page 4
def page_playbook():
    st.title("Segment playbook & assumptions")
    st.markdown("The offers the model suggests most often in each segment, and the assumptions behind every number.")
    cohort_note()

    tabs = st.tabs([f"{SEG_ICON[s]} {s}" for s in SEGMENTS])
    for tab, s in zip(tabs, SEGMENTS):
        with tab:
            a, b = st.container(), st.container()
            nice = lambda x: (x.replace("Contract", "Contract upgrade").replace("PaymentMethod", "Automatic payment")
                              .replace("TechSupport", "Free TechSupport").replace("OnlineSecurity", "Free OnlineSecurity")
                              .replace("OnlineBackup", "Free OnlineBackup").replace("DeviceProtection", "Free DeviceProtection"))
            pa = play[play.segment == s].drop(columns=["segment", "customers"])
            pb = play_b[play_b.segment == s].drop(columns=["segment", "customers"])
            for t_ in (pa, pb):
                t_["recommended action"] = t_["recommended action"].map(nice)
            cfg = {"recommended action": st.column_config.TextColumn("Offer", width="medium"),
                   "share of segment (%)": st.column_config.NumberColumn("Share %", format="%.0f"),
                   "mean drop in probability": st.column_config.NumberColumn("Risk drop", format="%.2f"),
                   "mean cost": st.column_config.NumberColumn("Cost", format="%.0f"),
                   "mean net value": st.column_config.NumberColumn("Net value", format="%.0f"),
                   "median discount (%)": st.column_config.NumberColumn("Discount %", format="%.0f"),
                   "negative net value (%)": st.column_config.NumberColumn("Loss-making %", format="%.0f")}
            with a:
                st.markdown("**Plan A – best offer**")
                st.dataframe(pa, hide_index=True, use_container_width=True, column_config=cfg)
            with b:
                st.markdown("**Plan B – if the customer refuses a longer contract**")
                st.dataframe(pb, hide_index=True, use_container_width=True, column_config=cfg)
            no_opt = pb.loc[pb["recommended action"] == "(no actionable option)", "share of segment (%)"]
            if len(no_opt):
                st.warning(f"Without a contract change, **{no_opt.iloc[0]:.0f}%** of contacted customers in this "
                           "segment have no affordable offer within the rules.")

    st.subheader("Assumptions")
    c = st.columns(4)
    c[0].metric("Offer cost (d)", f"{A['incentive_d']:.0f}")
    c[1].metric("Contact cost (f)", f"{A['contact_cost_f']:.0f}")
    c[2].metric("Offer acceptance (γ)", f"{A['acceptance_rate']:.0%}")
    c[3].metric("Value horizon", f"{A['value_horizon_months']} months")
    st.markdown(f"""
- **Model:** {A['model']}; contact thresholds {', '.join(f'{s} {t:.2f}' for s, t in A['thresholds'].items())}.
- **Customer value** = 12 × monthly charge (revenue, not margin).
- **Acceptance rate** of 30% is assumed from the literature; it was never observed because no campaign was run.
- **Data:** one snapshot of a fictional operator (IBM Telco sample); no personal data.
- **Offers** describe how the model reacts to a change, not proven causal effects; protected attributes are never changed.
- **Human in the loop:** the dashboard supports retention staff; it does not make automatic decisions about customers.
""")


pages = [st.Page(page_overview, title="Segment KPIs", icon="\U0001F4CA", default=True),
         st.Page(page_planner, title="Campaign planner", icon="\U0001F3AF"),
         st.Page(page_customer, title="Customer view", icon="\U0001F464"),
         st.Page(page_playbook, title="Playbook & assumptions", icon="\U0001F4D8")]
with st.sidebar:
    st.markdown("### \U0001F4C9 Churn Retention Dashboard")
    st.caption("Value-segmented churn prediction for telecom retention – MSc thesis prototype")
st.navigation(pages).run()
