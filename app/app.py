"""
AMR-AI — Streamlit Platform
4 modules: Meta-Analysis · Exception Scanner · ESKAPE Profiler · Score Calculator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

st.set_page_config(
    page_title="AMR-AI · MedFlow AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
    [data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    h1,h2,h3,h4 { color: #f1f5f9 !important; }
    p, li, label, .stMarkdown { color: #cbd5e1 !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    .metric-card {
        background: #1e293b; border-radius: 12px; padding: 18px 20px;
        border-left: 4px solid; margin-bottom: 8px;
    }
    .section-header {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    .exception-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 700;
    }
    div[data-testid="stDataFrame"] { background: #1e293b; border-radius: 8px; }
    .stDataFrame { background: #1e293b !important; }
    [data-testid="stExpander"] { background: #1e293b; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1e293b; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #0ea5e9 !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = "data/"
    return {
        "meta":    pd.read_csv(base + "meta_analytic_estimates.csv"),
        "studies": pd.read_csv(base + "studies_registry.csv"),
        "mol":     pd.read_csv(base + "molecules_db.csv"),
        "eskape":  pd.read_csv(base + "eskape_pathogens.csv"),
        "sub":     pd.read_csv(base + "subgroup_analysis.csv"),
    }

data = load_data()
df_meta    = data["meta"]
df_studies = data["studies"]
df_mol     = data["mol"]
df_eskape  = data["eskape"]
df_sub     = data["sub"]

STRAT_COLORS = {
    "Phage Therapy": "#3b82f6",
    "Antimicrobial Peptides": "#22c55e",
    "CRISPR Antimicrobial": "#a855f7",
    "Synergistic Combinations": "#f97316",
    "De Novo AI Molecules": "#ec4899",
}
CLASS_COLORS = {
    "Beta-lactam analog": "#ef4444", "Lipopeptide": "#f97316",
    "Glycopeptide analog": "#eab308", "Antimicrobial Peptide": "#22c55e",
    "CRISPR-guided": "#06b6d4", "Phage-derived endolysin": "#3b82f6",
    "Siderophore-antibiotic": "#8b5cf6", "AI-designed scaffold": "#ec4899",
    "Bacteriocin": "#14b8a6", "Quorum sensing inhibitor": "#f59e0b",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 AMR-AI")
    st.markdown("*Alternative Antimicrobial Research*")
    st.markdown("---")
    module = st.radio("Module", [
        "📊 Meta-Analysis Dashboard",
        "🔬 Exception Molecule Scanner",
        "🦠 ESKAPE Pathogen Profiler",
        "🧮 Exception Score Calculator",
        "🤖 AMR-AI Agent",
    ])
    st.markdown("---")
    st.markdown(f"**{len(df_studies)} studies** · **{df_studies['n_patients'].sum():,} patients**")
    st.markdown(f"**{len(df_mol)} molecules** · **5 strategies**")
    st.markdown("---")
    st.markdown("**MedFlow AI** · Mamadou Lamine TALL, PhD")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Meta-Analysis Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
if module == "📊 Meta-Analysis Dashboard":
    st.markdown('<div class="section-header">Meta-Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("DerSimonian-Laird random-effects · 55 studies · 79,241 patients · 2015–2026")

    # KPI cards
    best = df_meta.loc[df_meta["pooled_efficacy"].idxmax()]
    lowest_r = df_meta.loc[df_meta["pooled_resistance"].idxmin()]
    best_m = df_meta.loc[df_meta["pooled_mortality_reduction"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Studies", f"{len(df_studies)}", "2015–2026")
    c2.metric("Best Efficacy", f"{best['pooled_efficacy']:.1%}", best["strategy"].split()[0])
    c3.metric("Lowest Resistance", f"{lowest_r['pooled_resistance']:.1%}", lowest_r["strategy"].split()[0])
    c4.metric("Max Mortality ↓", f"{best_m['pooled_mortality_reduction']:.1%}", best_m["strategy"].split()[0])

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🌲 Forest Plot", "📈 Outcome Comparison", "📋 Studies Table"])

    with tab1:
        fig = go.Figure()
        for i, row in df_meta.iterrows():
            color = STRAT_COLORS[row["strategy"]]
            fig.add_trace(go.Scatter(
                x=[row["efficacy_ci_lo"], row["pooled_efficacy"], row["efficacy_ci_hi"]],
                y=[i, i, i], mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=[6, 14, 6], color=[color, color, color],
                            symbol=["line-ew", "diamond", "line-ew"]),
                name=row["strategy"],
                hovertemplate=f"<b>{row['strategy']}</b><br>"
                              f"Pooled: {row['pooled_efficacy']:.1%}<br>"
                              f"95% CI: [{row['efficacy_ci_lo']:.1%} – {row['efficacy_ci_hi']:.1%}]<br>"
                              f"I²: {row['efficacy_i2']:.0f}%<extra></extra>"
            ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#475569", opacity=0.6)
        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            xaxis=dict(title="Pooled Efficacy (95% CI)", color="#94a3b8",
                       tickformat=".0%", gridcolor="#334155"),
            yaxis=dict(tickvals=list(range(len(df_meta))),
                       ticktext=df_meta["strategy"].tolist(),
                       color="#94a3b8", gridcolor="#334155"),
            font=dict(color="#f1f5f9"), height=380,
            margin=dict(l=200, r=20, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # I² table
        tbl = df_meta[["strategy", "n_studies", "total_patients",
                        "pooled_efficacy", "efficacy_ci_lo", "efficacy_ci_hi", "efficacy_i2"]].copy()
        tbl.columns = ["Strategy", "Studies", "Patients", "Efficacy", "CI Low", "CI High", "I²"]
        tbl["Efficacy"] = tbl["Efficacy"].map("{:.1%}".format)
        tbl["CI Low"] = tbl["CI Low"].map("{:.1%}".format)
        tbl["CI High"] = tbl["CI High"].map("{:.1%}".format)
        tbl["I²"] = tbl["I²"].map("{:.0f}%".format)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    with tab2:
        outcome = st.selectbox("Outcome", ["Efficacy", "Resistance Emergence", "Mortality Reduction"])
        col_map = {
            "Efficacy":             ("pooled_efficacy", "efficacy_ci_lo", "efficacy_ci_hi"),
            "Resistance Emergence": ("pooled_resistance", "resistance_ci_lo", "resistance_ci_hi"),
            "Mortality Reduction":  ("pooled_mortality_reduction", "mortality_ci_lo", "mortality_ci_hi"),
        }
        cv, clo, chi = col_map[outcome]
        fig2 = go.Figure()
        for _, row in df_meta.iterrows():
            color = STRAT_COLORS[row["strategy"]]
            fig2.add_trace(go.Bar(
                x=[row[cv]], y=[row["strategy"]],
                orientation="h",
                error_x=dict(type="data",
                             array=[row[chi] - row[cv]],
                             arrayminus=[row[cv] - row[clo]],
                             color="white", thickness=1.5, width=6),
                marker_color=color, name=row["strategy"],
                hovertemplate=f"<b>{row['strategy']}</b><br>{outcome}: {row[cv]:.1%}<extra></extra>"
            ))
        fig2.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            xaxis=dict(title=outcome, tickformat=".0%", color="#94a3b8", gridcolor="#334155"),
            yaxis=dict(color="#94a3b8"), font=dict(color="#f1f5f9"),
            height=350, showlegend=False, margin=dict(l=200, r=20, t=20, b=40),
            bargap=0.3,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        strat_filter = st.multiselect("Filter by strategy", df_studies["strategy"].unique().tolist(),
                                       default=df_studies["strategy"].unique().tolist())
        filtered = df_studies[df_studies["strategy"].isin(strat_filter)]
        st.dataframe(
            filtered[["study_id", "first_author", "year", "strategy", "pathogen",
                       "n_patients", "efficacy_rate", "resistance_emergence", "nos_score"]],
            use_container_width=True, hide_index=True
        )

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Exception Molecule Scanner
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "🔬 Exception Molecule Scanner":
    st.markdown('<div class="section-header">Exception Molecule Scanner</div>', unsafe_allow_html=True)
    st.markdown("Molecules scored on Novelty · Resistance-Proof · Activity · Toxicity")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        class_filter = st.multiselect("Molecule Class", sorted(df_mol["class"].unique()),
                                       default=sorted(df_mol["class"].unique()))
    with col_f2:
        min_score = st.slider("Min Exception Score", 0.0, 1.0, 0.0, 0.05)
    with col_f3:
        pathogen_filter = st.text_input("Target Pathogen (contains)", "")

    mol_f = df_mol[df_mol["class"].isin(class_filter)]
    mol_f = mol_f[mol_f["exception_score"] >= min_score]
    if pathogen_filter:
        mol_f = mol_f[mol_f["target_pathogens"].str.contains(pathogen_filter, case=False, na=False)]

    st.markdown(f"**{len(mol_f)} molecules** matching filters")

    tab_a, tab_b, tab_c = st.tabs(["🗺️ Exception Map", "🏆 Ranking", "🎯 Molecule Profile"])

    with tab_a:
        fig3 = go.Figure()
        for cls in mol_f["class"].unique():
            sub = mol_f[mol_f["class"] == cls]
            color = CLASS_COLORS.get(cls, "#94a3b8")
            fig3.add_trace(go.Scatter(
                x=sub["novelty_score"], y=sub["resistance_proof_score"],
                mode="markers+text",
                marker=dict(size=sub["activity_score"] * 30, color=color,
                            opacity=0.85, line=dict(color="white", width=0.8)),
                text=sub.apply(lambda r: r["name"] if r["exception_score"] > 0.75 else "", axis=1),
                textposition="top right", textfont=dict(size=9, color="#f1f5f9"),
                name=cls,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                              "Class: %{customdata[1]}<br>"
                              "Exception: %{customdata[2]:.3f}<br>"
                              "Activity: %{customdata[3]:.3f}<extra></extra>",
                customdata=sub[["name", "class", "exception_score", "activity_score"]].values,
            ))
        fig3.add_shape(type="rect", x0=0.72, y0=0.78, x1=1.02, y1=1.02,
                       line=dict(color="#ec4899", width=2, dash="dash"))
        fig3.add_annotation(x=0.87, y=1.03, text="Exception Zone",
                            font=dict(color="#ec4899", size=11), showarrow=False)
        fig3.add_vline(x=0.72, line_dash="dot", line_color="#475569", opacity=0.5)
        fig3.add_hline(y=0.78, line_dash="dot", line_color="#475569", opacity=0.5)
        fig3.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            xaxis=dict(title="Novelty Score", color="#94a3b8", gridcolor="#334155", range=[0, 1.05]),
            yaxis=dict(title="Resistance-Proof Score", color="#94a3b8", gridcolor="#334155", range=[0, 1.08]),
            font=dict(color="#f1f5f9"), height=500,
            legend=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(size=10)),
            margin=dict(l=60, r=20, t=40, b=60),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab_b:
        top_n = st.slider("Top N molecules", 5, len(mol_f), min(15, len(mol_f)))
        top = mol_f.nlargest(top_n, "exception_score")
        fig4 = go.Figure()
        for _, row in top.iterrows():
            color = CLASS_COLORS.get(row["class"], "#94a3b8")
            fig4.add_trace(go.Bar(
                x=[row["exception_score"]], y=[row["name"]],
                orientation="h", marker_color=color,
                hovertemplate=f"<b>{row['name']}</b><br>Class: {row['class']}<br>"
                              f"Score: {row['exception_score']:.3f}<br>"
                              f"Mechanism: {row['mechanism']}<extra></extra>",
                showlegend=False,
            ))
        fig4.add_vline(x=0.75, line_dash="dash", line_color="#eab308",
                       annotation_text="Exception threshold", annotation_font_color="#eab308")
        fig4.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            xaxis=dict(title="Exception Score", color="#94a3b8",
                       gridcolor="#334155", range=[0, 1.05]),
            yaxis=dict(color="#94a3b8", autorange="reversed"),
            font=dict(color="#f1f5f9"), height=max(300, top_n * 32),
            margin=dict(l=160, r=20, t=20, b=40),
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(
            top[["name", "class", "mechanism", "exception_score",
                 "activity_score", "resistance_proof_score", "novelty_score",
                 "toxicity_index", "mic90_mg_L", "target_pathogens"]],
            use_container_width=True, hide_index=True
        )

    with tab_c:
        selected_mol = st.selectbox("Select molecule", mol_f["name"].tolist())
        mol_row = mol_f[mol_f["name"] == selected_mol].iloc[0]

        c1, c2 = st.columns([1, 1])
        with c1:
            score = mol_row["exception_score"]
            badge_color = "#22c55e" if score > 0.75 else "#eab308" if score > 0.55 else "#ef4444"
            badge_text = "EXCEPTION" if score > 0.75 else "CANDIDATE" if score > 0.55 else "STANDARD"
            st.markdown(f"""
            <div class="metric-card" style="border-color:{badge_color}">
                <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9">{mol_row['name']}</div>
                <div style="color:#94a3b8;font-size:0.9rem">{mol_row['class']}</div>
                <div style="margin-top:8px">
                    <span class="exception-badge" style="background:{badge_color}20;color:{badge_color};border:1px solid {badge_color}">
                        {badge_text} · {score:.3f}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"**Mechanism:** {mol_row['mechanism']}")
            st.markdown(f"**MIC₉₀:** {mol_row['mic90_mg_L']:.3f} mg/L")
            st.markdown(f"**MW:** {mol_row['molecular_weight']:.0f} Da")
            st.markdown(f"**Targets:** {mol_row['target_pathogens']}")

        with c2:
            categories = ["Activity", "Resist-Proof", "Novelty", "Low Toxicity", "Exception"]
            values = [
                mol_row["activity_score"],
                mol_row["resistance_proof_score"],
                mol_row["novelty_score"],
                1 - mol_row["toxicity_index"],
                mol_row["exception_score"],
            ]
            fig_r = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself", fillcolor="rgba(236,72,153,0.2)",
                line=dict(color="#ec4899", width=2.5),
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor="#1e293b",
                    radialaxis=dict(visible=True, range=[0, 1], color="#475569",
                                   tickfont=dict(color="#94a3b8", size=9)),
                    angularaxis=dict(color="#94a3b8", tickfont=dict(color="#f1f5f9", size=11)),
                ),
                paper_bgcolor="#0f172a", font=dict(color="#f1f5f9"),
                height=320, margin=dict(l=40, r=40, t=30, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — ESKAPE Pathogen Profiler
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "🦠 ESKAPE Pathogen Profiler":
    st.markdown('<div class="section-header">ESKAPE Pathogen Profiler</div>', unsafe_allow_html=True)
    st.markdown("Resistance profiles · Annual burden · Best strategy recommendation")

    pathogen_sel = st.selectbox("Select pathogen", df_eskape["pathogen"].tolist())
    path_row = df_eskape[df_eskape["pathogen"] == pathogen_sel].iloc[0]

    c1, c2, c3 = st.columns(3)
    mortality = path_row["global_mortality_pct"]
    mort_color = "#ef4444" if mortality > 35 else "#f97316" if mortality > 20 else "#eab308"
    c1.metric("Global Mortality", f"{mortality:.1f}%")
    c2.metric("Annual Cases", f"{path_row['annual_cases_M']:.2f}M")
    c3.metric("Resistance Genes", str(len(path_row["resistance_genes"].split(","))))

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Resistance Profile")
        st.markdown(f"""
        <div class="metric-card" style="border-color:{mort_color}">
            <div style="color:#94a3b8;font-size:0.85rem">RESISTANCE GENES</div>
            <div style="color:#f1f5f9;font-size:1rem;margin-top:4px">{path_row['resistance_genes']}</div>
        </div>
        <div class="metric-card" style="border-color:#6366f1">
            <div style="color:#94a3b8;font-size:0.85rem">MECHANISMS</div>
            <div style="color:#f1f5f9;font-size:1rem;margin-top:4px">{path_row['resistance_mechanisms']}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Recommended Strategies")
        strategies = [s.strip() for s in path_row["effective_strategies"].split(";")]
        for strat in strategies:
            color = STRAT_COLORS.get(strat, "#94a3b8")
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}; padding: 10px 16px;">
                <span style="color:{color};font-weight:700">✦</span>
                <span style="color:#f1f5f9;margin-left:8px">{strat}</span>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Subgroup Efficacy")
        sub_path = df_sub[df_sub["pathogen"] == pathogen_sel].sort_values("pooled_efficacy", ascending=True)
        if len(sub_path) > 0:
            fig5 = go.Figure()
            for _, row in sub_path.iterrows():
                color = STRAT_COLORS.get(row["strategy"], "#94a3b8")
                fig5.add_trace(go.Bar(
                    x=[row["pooled_efficacy"]], y=[row["strategy"]],
                    orientation="h", marker_color=color,
                    error_x=dict(type="data",
                                 array=[row["ci_hi"] - row["pooled_efficacy"]],
                                 arrayminus=[row["pooled_efficacy"] - row["ci_lo"]],
                                 color="white", thickness=1.5),
                    hovertemplate=f"<b>{row['strategy']}</b><br>"
                                  f"Efficacy: {row['pooled_efficacy']:.1%}<br>"
                                  f"I²: {row['i2']:.0f}%<extra></extra>",
                    showlegend=False,
                ))
            fig5.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                xaxis=dict(title="Pooled Efficacy", tickformat=".0%",
                           color="#94a3b8", gridcolor="#334155"),
                yaxis=dict(color="#94a3b8"),
                font=dict(color="#f1f5f9"), height=300,
                margin=dict(l=180, r=20, t=20, b=40),
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No subgroup data for this pathogen.")

    st.markdown("---")
    st.markdown("#### All ESKAPE Pathogens — Mortality Burden")
    fig6 = px.bar(
        df_eskape.sort_values("global_mortality_pct"),
        x="global_mortality_pct", y="pathogen",
        orientation="h", color="global_mortality_pct",
        color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
        labels={"global_mortality_pct": "Mortality (%)", "pathogen": ""},
    )
    fig6.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9"), height=300,
        coloraxis_showscale=False,
        margin=dict(l=200, r=20, t=20, b=40),
        xaxis=dict(color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(color="#94a3b8"),
    )
    st.plotly_chart(fig6, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Exception Score Calculator
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "🧮 Exception Score Calculator":
    st.markdown('<div class="section-header">Exception Score Calculator</div>', unsafe_allow_html=True)
    st.markdown("Compute the exception score for any custom molecule candidate.")

    st.markdown("""
    <div style="background:#1e293b;border-radius:10px;padding:14px 18px;border-left:4px solid #0ea5e9;margin-bottom:1rem">
    <b style="color:#0ea5e9">Formula:</b>
    <span style="color:#f1f5f9"> E = 0.35 × Novelty + 0.30 × Resist-Proof + 0.20 × Activity + 0.15 × (1 – Toxicity)</span>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        mol_name = st.text_input("Molecule Name", "MyMolecule-1")
        mol_class = st.selectbox("Class", list(CLASS_COLORS.keys()))
        mechanism = st.text_input("Mechanism of Action", "Multi-target")
        target = st.text_input("Target Pathogen(s)", "MRSA; K. pneumoniae (KPC)")

    with col2:
        novelty = st.slider("Novelty Score", 0.0, 1.0, 0.70, 0.01,
                            help="Structural + mechanistic distance from known antibiotics")
        resist_proof = st.slider("Resistance-Proof Score", 0.0, 1.0, 0.80, 0.01,
                                  help="Inverse probability of single-step resistance")
        activity = st.slider("Antimicrobial Activity", 0.0, 1.0, 0.75, 0.01,
                              help="Pooled MIC90 efficacy score")
        toxicity = st.slider("Toxicity Index", 0.0, 1.0, 0.20, 0.01,
                              help="Higher = more toxic; enter estimated therapeutic index inverse")

    exception_score = 0.35 * novelty + 0.30 * resist_proof + 0.20 * activity + 0.15 * (1 - toxicity)

    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)

    if exception_score > 0.75:
        badge_color, badge_text, badge_desc = "#22c55e", "EXCEPTION MOLECULE", "Multi-mutation barrier · High priority"
    elif exception_score > 0.55:
        badge_color, badge_text, badge_desc = "#eab308", "CANDIDATE", "Promising · Further optimization needed"
    else:
        badge_color, badge_text, badge_desc = "#ef4444", "STANDARD", "Below exception threshold"

    with sc1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{badge_color};text-align:center">
            <div style="font-size:2.2rem;font-weight:900;color:{badge_color}">{exception_score:.3f}</div>
            <div style="color:#94a3b8;font-size:0.85rem">EXCEPTION SCORE</div>
            <div style="margin-top:8px">
                <span class="exception-badge" style="background:{badge_color}20;color:{badge_color};border:1px solid {badge_color}">
                    {badge_text}
                </span>
            </div>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:6px">{badge_desc}</div>
        </div>""", unsafe_allow_html=True)

    with sc2:
        components = {
            "Novelty (×0.35)": 0.35 * novelty,
            "Resist-Proof (×0.30)": 0.30 * resist_proof,
            "Activity (×0.20)": 0.20 * activity,
            "Low Toxicity (×0.15)": 0.15 * (1 - toxicity),
        }
        fig_bar = go.Figure(go.Bar(
            x=list(components.values()),
            y=list(components.keys()),
            orientation="h",
            marker_color=["#0ea5e9", "#22c55e", "#f97316", "#a855f7"],
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            xaxis=dict(color="#94a3b8", gridcolor="#334155", range=[0, 0.36]),
            yaxis=dict(color="#94a3b8"),
            font=dict(color="#f1f5f9"), height=220,
            margin=dict(l=160, r=10, t=10, b=30),
            title=dict(text="Score Breakdown", font=dict(color="#94a3b8", size=11)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with sc3:
        categories = ["Novelty", "Resist-Proof", "Activity", "Low Toxicity", "Exception"]
        values_r = [novelty, resist_proof, activity, 1 - toxicity, exception_score]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values_r + [values_r[0]],
            theta=categories + [categories[0]],
            fill="toself", fillcolor=f"rgba({int(badge_color[1:3],16)},{int(badge_color[3:5],16)},{int(badge_color[5:7],16)},0.18)",
            line=dict(color=badge_color, width=2.5),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1e293b",
                radialaxis=dict(visible=True, range=[0, 1], color="#475569",
                                tickfont=dict(color="#94a3b8", size=8)),
                angularaxis=dict(color="#94a3b8", tickfont=dict(color="#f1f5f9", size=10)),
            ),
            paper_bgcolor="#0f172a", font=dict(color="#f1f5f9"),
            height=220, margin=dict(l=30, r=30, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Percentile rank vs database
    db_scores = df_mol["exception_score"].values
    percentile = float(np.sum(db_scores < exception_score) / len(db_scores) * 100)
    st.markdown(f"""
    <div class="metric-card" style="border-color:#6366f1">
        <b style="color:#6366f1">Database Comparison</b><br>
        <span style="color:#f1f5f9">{mol_name}</span> scores better than
        <b style="color:{badge_color}">{percentile:.0f}%</b> of the {len(db_scores)} molecules in the AMR-AI database.
        {'Above exception threshold — qualifies as exception molecule.' if exception_score > 0.75 else 'Below exception threshold — optimization recommended.'}
    </div>""", unsafe_allow_html=True)

    # Add to comparison table
    if st.button("➕ Add to Comparison"):
        new_row = {
            "name": mol_name, "class": mol_class, "mechanism": mechanism,
            "activity_score": activity, "resistance_proof_score": resist_proof,
            "novelty_score": novelty, "toxicity_index": toxicity,
            "exception_score": round(exception_score, 4),
            "target_pathogens": target, "mic90_mg_L": "N/A",
        }
        if "custom_molecules" not in st.session_state:
            st.session_state["custom_molecules"] = []
        st.session_state["custom_molecules"].append(new_row)
        st.success(f"✓ {mol_name} added (score: {exception_score:.3f})")

    if "custom_molecules" in st.session_state and st.session_state["custom_molecules"]:
        st.markdown("#### Custom Molecules Comparison")
        custom_df = pd.DataFrame(st.session_state["custom_molecules"])
        st.dataframe(custom_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — AMR-AI Agent
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "🤖 AMR-AI Agent":
    from agent import run_agent

    st.markdown('<div class="section-header">AMR-AI Agent</div>', unsafe_allow_html=True)
    st.markdown("Agent clinique IA · Hyper-méta-analyse de 97 essais · 58 000+ patients")

    # Clé API — secrets Streamlit Cloud en priorité, sinon saisie manuelle
    if "amr_api_key" not in st.session_state:
        st.session_state["amr_api_key"] = st.secrets.get("ANTHROPIC_API_KEY", "")

    with st.sidebar:
        st.markdown("---")
        if st.session_state["amr_api_key"]:
            st.success("Clé configurée ✓")
        else:
            api_key = st.text_input("🔑 Anthropic API Key", type="password")
            if api_key:
                st.session_state["amr_api_key"] = api_key
                st.rerun()

    if not st.session_state["amr_api_key"]:
        st.info("Entrez votre clé API Anthropic dans la barre latérale pour activer l'agent.")
        st.stop()

    # Initialiser l'historique
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []  # messages internes Claude

    # Exemples de questions
    st.markdown("#### Questions exemples")
    cols = st.columns(3)
    examples = [
        "Meilleur traitement pour Klebsiella KPC ?",
        "Compare cefiderocol vs phage therapy",
        "Profil de résistance de A. baumannii XDR",
    ]
    for i, (col, ex) in enumerate(zip(cols, examples)):
        if col.button(ex, key=f"ex_{i}"):
            st.session_state["pending_question"] = ex

    st.markdown("---")

    # Afficher l'historique chat
    for msg in st.session_state["agent_history"]:
        with st.chat_message(msg["role"], avatar="🧬" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Posez votre question clinique AMR...")
    if "pending_question" in st.session_state:
        user_input = st.session_state.pop("pending_question")

    if user_input:
        st.session_state["agent_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🧬"):
            with st.spinner("L'agent analyse les données..."):
                try:
                    answer, new_history = run_agent(
                        user_input,
                        st.session_state["agent_messages"],
                        st.session_state["amr_api_key"]
                    )
                    st.session_state["agent_messages"] = new_history
                    st.markdown(answer)
                    st.session_state["agent_history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erreur agent : {e}")

    if st.session_state["agent_history"]:
        if st.button("🗑️ Effacer la conversation"):
            st.session_state["agent_history"] = []
            st.session_state["agent_messages"] = []
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem'>"
    "AMR-AI v1.0 · MedFlow AI · Mamadou Lamine TALL, PhD · "
    "For research purposes only — does not replace clinical judgement"
    "</div>",
    unsafe_allow_html=True
)
