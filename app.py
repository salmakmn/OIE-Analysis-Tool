import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & STYLE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OIE Tracker — EM&S Finance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .section-header { font-size: 1.4rem; font-weight: 700; color: #1e3a5f;
                      border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;
                      margin-bottom: 1rem; }
    .a25-warning { background-color: #fff7ed; border-left: 4px solid #f97316;
                   padding: 0.6rem 0.9rem; border-radius: 4px; margin: 0.4rem 0;
                   font-size: 0.85rem; color: #7c2d12; }
    .a25-info { background-color: #eff6ff; border-left: 4px solid #3b82f6;
                padding: 0.6rem 0.9rem; border-radius: 4px; margin: 0.4rem 0;
                font-size: 0.85rem; color: #1e3a5f; }
    .data-warning { background-color: #fef9c3; border-left: 4px solid #eab308;
                    padding: 0.6rem 0.9rem; border-radius: 4px; margin: 0.4rem 0;
                    font-size: 0.85rem; color: #713f12; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════
A25_SEUIL_SIGNIFICATIF = 1.0
MOIS_YTD = 8

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════
def clean_numeric(val):
    if pd.isna(val) or str(val).strip() in ["-", "", "nan", "N/A"]:
        return np.nan
    s = str(val).strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "").replace(" ", "").replace("\xa0", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        result = float(s)
        return -result if negative else result
    except:
        return np.nan

def fmt_m(val, decimals=1):
    if pd.isna(val): return "—"
    return f"{val:+.{decimals}f}M€" if val != 0 else "0.0M€"

def fmt_pct(val, decimals=1):
    if pd.isna(val): return "—"
    return f"{val:+.{decimals}f}%"

def safe_pct(numerator, denominator, seuil=A25_SEUIL_SIGNIFICATIF):
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    if denominator == 0 or abs(denominator) < seuil:
        return np.nan
    return (numerator / abs(denominator)) * 100

def safe_pct_series(ecart_series, ref_series, seuil=A25_SEUIL_SIGNIFICATIF):
    result = ecart_series / ref_series.abs() * 100
    mask = ref_series.isna() | ecart_series.isna() | (ref_series.abs() < seuil)
    result[mask] = np.nan
    return result.round(1)

def a25_coverage(df_agg):
    if "A25" not in df_agg.columns:
        return {"n_total": len(df_agg), "n_with_a25": 0, "n_without": len(df_agg),
                "scopes_with": [], "scopes_without": df_agg["Scope"].tolist(), "coverage_pct": 0.0}
    has_a25 = df_agg["A25"].notna() & (df_agg["A25"].abs() >= A25_SEUIL_SIGNIFICATIF)
    scopes_with    = df_agg.loc[has_a25, "Scope"].tolist()
    scopes_without = df_agg.loc[~has_a25, "Scope"].tolist()
    n_total = len(df_agg)
    n_with  = len(scopes_with)
    return {"n_total": n_total, "n_with_a25": n_with, "n_without": n_total - n_with,
            "scopes_with": scopes_with, "scopes_without": scopes_without,
            "coverage_pct": round(n_with / n_total * 100, 1) if n_total > 0 else 0.0}

def render_a25_badge(coverage):
    n_with  = coverage["n_with_a25"]
    n_total = coverage["n_total"]
    pct     = coverage["coverage_pct"]
    missing = coverage["scopes_without"]
    if n_with == n_total:
        st.markdown(f'<div class="a25-info">✅ <strong>A25 complet</strong> — disponible pour tous les {n_total} scopes.</div>', unsafe_allow_html=True)
    elif n_with == 0:
        st.markdown('<div class="a25-warning">⚠️ <strong>A25 non disponible</strong> — aucun scope ne dispose de données A25. Les comparaisons YoY sont désactivées.</div>', unsafe_allow_html=True)
    else:
        missing_str = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        st.markdown(f'<div class="a25-warning">ℹ️ <strong>A25 partiel</strong> — disponible pour <strong>{n_with}/{n_total} scopes</strong> ({pct}%). Les % vs A25 sont masqués pour les scopes sans données.<br><em>Scopes sans A25 : {missing_str}</em></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTION CLÉ — AGRÉGATION INTELLIGENTE CORRIGÉE
# ═══════════════════════════════════════════════════════════════════════════════
def smart_aggregate_by_scope(df, numeric_cols):
    """
    Agrège par scope COLONNE PAR COLONNE.
    
    Pour chaque scope × colonne :
    - Si au moins 1 driver (non-total) a une valeur → sommer les détails
    - Sinon → utiliser la ligne total du scope (même si d'autres colonnes ont des détails)
    
    C'est le comportement correct pour Puran/MEA :
    YTD 08 → détails disponibles → sommer les drivers
    B26    → détails tous NaN   → utiliser la ligne total Puran
    """
    df = df.copy()
    if "_is_total" not in df.columns:
        df["_is_total"] = df["Drivers"].str.strip() == df["Scope"].str.strip()

    results = []
    for scope in df["Scope"].dropna().unique():
        df_scope  = df[df["Scope"] == scope]
        df_detail = df_scope[~df_scope["_is_total"]]
        df_total  = df_scope[df_scope["_is_total"]]
        row = {"Scope": scope}

        for col in numeric_cols:
            detail_values = df_detail[col].dropna()

            if len(detail_values) > 0:
                # Détails disponibles pour cette colonne → sommer
                row[col] = detail_values.sum()
            elif not df_total.empty and df_total[col].notna().any():
                # Pas de détail pour cette colonne → fallback sur le total
                row[col] = df_total[col].values[0]
            else:
                row[col] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def enrich_driver_detail(df_scope_detail, df_scope_total, numeric_cols):
    """
    ✅ NOUVEAU — Enrichit le tableau détail driver avec les valeurs du total
    pour les colonnes où TOUS les drivers sont NaN.
    
    Cas Puran : B26, RF3, RF1 sont NaN pour tous les drivers
    → On affiche la valeur du total Puran dans une ligne dédiée
    → On ne l'attribue PAS à un driver individuel (ce serait faux)
    → On l'affiche comme ligne "Total Puran (données agrégées)"
    
    Args:
        df_scope_detail : DataFrame des lignes détail du scope
        df_scope_total  : DataFrame de la ligne total du scope (1 ligne)
        numeric_cols    : colonnes numériques à vérifier
    
    Returns:
        df_enriched     : DataFrame avec ligne total ajoutée si nécessaire
        cols_from_total : liste des colonnes dont les valeurs viennent du total
    """
    if df_scope_total.empty:
        return df_scope_detail, []

    cols_from_total = []
    for col in numeric_cols:
        if col not in df_scope_detail.columns:
            continue
        detail_values = df_scope_detail[col].dropna()
        total_value   = df_scope_total[col].dropna()
        # Si aucun détail mais total disponible → colonne "vient du total"
        if len(detail_values) == 0 and len(total_value) > 0:
            cols_from_total.append(col)

    return df_scope_detail, cols_from_total


@st.cache_data
def load_data(file):
    for encoding in ["latin-1", "utf-8-sig", "cp1252", "iso-8859-1"]:
        try:
            df = pd.read_csv(file, sep=";", encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("❌ Impossible de décoder le fichier.")

    df.columns = df.columns.str.strip()
    numeric_cols = ["YTD 08", "YTG", "RF3", "RF1", "B26", "H27", "B27", "A25"]
    for col in [c for c in numeric_cols if c in df.columns]:
        df[col] = df[col].apply(clean_numeric)
    df["_is_total"] = df["Drivers"].str.strip() == df["Scope"].str.strip()
    return df

def compute_kpis(df_agg, coverage):
    kpis = {}
    if "YTD 08" in df_agg.columns: kpis["ytd_08"] = df_agg["YTD 08"].sum()
    if "B26"    in df_agg.columns: kpis["b26"]    = df_agg["B26"].sum()
    if "RF3"    in df_agg.columns: kpis["rf3"]    = df_agg["RF3"].sum()

    if "A25" in df_agg.columns and coverage["n_with_a25"] > 0:
        df_with_a25 = df_agg[df_agg["Scope"].isin(coverage["scopes_with"])]
        kpis["a25"] = df_with_a25["A25"].sum()
        kpis["a25_scope_count"] = coverage["n_with_a25"]
    else:
        kpis["a25"] = np.nan
        kpis["a25_scope_count"] = 0

    if "ytd_08" in kpis and "b26" in kpis:
        b26_ytd = kpis["b26"] * (MOIS_YTD / 12)
        kpis["ecart_vs_b26"]  = kpis["ytd_08"] - b26_ytd
        kpis["ecart_pct_b26"] = safe_pct(kpis["ecart_vs_b26"], b26_ytd)

    if "ytd_08" in kpis and not pd.isna(kpis.get("a25", np.nan)) and coverage["n_with_a25"] > 0:
        df_with_a25 = df_agg[df_agg["Scope"].isin(coverage["scopes_with"])]
        ytd_a25_scopes = df_with_a25["YTD 08"].sum() if "YTD 08" in df_with_a25.columns else np.nan
        a25_ytd = kpis["a25"] * (MOIS_YTD / 12)
        kpis["ecart_vs_a25"]  = ytd_a25_scopes - a25_ytd
        kpis["ecart_pct_a25"] = safe_pct(kpis["ecart_vs_a25"], a25_ytd)
    else:
        kpis["ecart_vs_a25"]  = np.nan
        kpis["ecart_pct_a25"] = np.nan

    return kpis

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Sanofi_logo.svg/320px-Sanofi_logo.svg.png", width=150)
    st.title("OIE Tracker")
    st.caption("EM&S Finance — 2026")
    st.divider()

    uploaded_file = st.file_uploader("📂 Charger le fichier OIE (CSV)", type=["csv"])

    if uploaded_file:
        df_raw = load_data(uploaded_file)
        st.success(f"✅ {len(df_raw)} lignes chargées")
        n_total_rows  = df_raw["_is_total"].sum()
        n_detail_rows = (~df_raw["_is_total"]).sum()
        st.info(f"📊 {n_total_rows} lignes total | {n_detail_rows} lignes détail")

        st.subheader("🎛️ Filtres")
        all_scopes      = sorted(df_raw["Scope"].dropna().unique().tolist())
        selected_scopes = st.multiselect("Scope(s)", all_scopes, default=all_scopes)
        driver_search   = st.text_input("🔍 Rechercher un driver", "")

        st.divider()
        st.subheader("⚙️ Paramètres alertes")
        alert_threshold = st.slider("Seuil d'alerte (%)", 5, 30, 10, 1)
        top_n           = st.slider("Top N drivers", 3, 15, 5, 1)

        df = df_raw[df_raw["Scope"].isin(selected_scopes)].copy()

        SCENARIO_COLS       = ["RF3", "RF1", "B26", "H27", "B27"]
        available_scenarios = [c for c in SCENARIO_COLS if c in df.columns]
        selected_scenarios  = st.multiselect("Scénarios budgétaires", available_scenarios, default=available_scenarios)
    else:
        st.info("👆 Chargez votre fichier CSV pour commencer")
        df = None

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════
if df is not None:
    numeric_cols      = ["YTD 08", "YTG", "RF3", "RF1", "B26", "H27", "B27", "A25"]
    available_numeric = [c for c in numeric_cols if c in df.columns]
    df_agg            = smart_aggregate_by_scope(df, available_numeric)
    coverage          = a25_coverage(df_agg)

    tabs = st.tabs([
        "📊 Vue d'ensemble", "📈 Comparaison Budgétaire", "🔍 Analyse par Driver",
        "📉 Variance", "🌊 Waterfall & Tendances", "🗺️ Heatmap", "⚠️ Alertes & R&O", "📥 Export"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — VUE D'ENSEMBLE
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Vue d\'ensemble globale</div>', unsafe_allow_html=True)
        kpis = compute_kpis(df_agg, coverage)
        render_a25_badge(coverage)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("YTD 08", fmt_m(kpis.get("ytd_08")),
                      delta=fmt_m(kpis.get("ecart_vs_b26")), delta_color="inverse")
        with col2:
            st.metric("Budget B26 (FY)", fmt_m(kpis.get("b26")))
        with col3:
            a25_label = "A25 (FY)"
            if 0 < coverage["n_with_a25"] < coverage["n_total"]:
                a25_label = f"A25 — {coverage['n_with_a25']}/{coverage['n_total']} scopes"
            ecart_pct_a25 = kpis.get("ecart_pct_a25")
            delta_a25 = ("N/A vs A25 YTD" if coverage["n_with_a25"] == 0
                         else ("% non significatif" if pd.isna(ecart_pct_a25)
                               else fmt_pct(ecart_pct_a25) + " vs A25 YTD"))
            st.metric(a25_label,
                      fmt_m(kpis.get("a25")) if not pd.isna(kpis.get("a25", np.nan)) else "N/A",
                      delta=delta_a25)
        with col4:
            st.metric("RF3 (FY)", fmt_m(kpis.get("rf3")))
        with col5:
            st.metric("Écart vs B26 YTD", fmt_pct(kpis.get("ecart_pct_b26")))

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            if "YTD 08" in df_agg.columns:
                scope_ytd = df_agg[["Scope", "YTD 08"]].dropna().sort_values("YTD 08")
                fig = px.bar(scope_ytd, x="YTD 08", y="Scope", orientation="h",
                             title="YTD 08 par Scope (M€)", color="YTD 08",
                             color_continuous_scale="RdYlGn_r", text="YTD 08")
                fig.update_traces(texttemplate="%{text:.1f}M€", textposition="outside")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with col_right:
            if "YTD 08" in df_agg.columns:
                scope_abs = df_agg.copy()
                scope_abs["Abs YTD 08"] = scope_abs["YTD 08"].abs()
                fig2 = px.pie(scope_abs, values="Abs YTD 08", names="Scope",
                              title="Concentration des dépenses OIE par Scope", hole=0.4)
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 Récapitulatif par Scope")
        summary_cols      = ["YTD 08", "B26", "A25", "RF3"]
        available_summary = [c for c in summary_cols if c in df_agg.columns]
        if available_summary:
            summary = df_agg[["Scope"] + available_summary].copy()
            if "YTD 08" in summary.columns and "B26" in summary.columns:
                b26_ytd_col = summary["B26"] * (MOIS_YTD / 12)
                summary["Écart YTD vs B26"] = summary["YTD 08"] - b26_ytd_col
                summary["Écart % vs B26"]   = safe_pct_series(summary["Écart YTD vs B26"], b26_ytd_col)
            if "A25" in summary.columns:
                summary["A25 dispo ?"] = summary["A25"].apply(
                    lambda x: "✅" if (pd.notna(x) and abs(x) >= A25_SEUIL_SIGNIFICATIF) else "⚠️ N/A")
                a25_ytd_col = summary["A25"] * (MOIS_YTD / 12)
                ecart_a25   = summary["YTD 08"] - a25_ytd_col if "YTD 08" in summary.columns else pd.Series(np.nan, index=summary.index)
                summary["Écart % vs A25"] = safe_pct_series(ecart_a25, a25_ytd_col)

            fmt_cols = {c: "{:.1f}" for c in summary.columns if c not in ["Scope", "A25 dispo ?"] and "%" not in c}
            fmt_cols.update({c: "{:.1f}%" for c in summary.columns if "%" in c})
            gradient_cols = [c for c in ["Écart % vs B26", "Écart % vs A25"] if c in summary.columns]
            styled = summary.style.format(fmt_cols, na_rep="N/A")
            if gradient_cols:
                styled = styled.background_gradient(subset=gradient_cols, cmap="RdYlGn_r")
            st.dataframe(styled, use_container_width=True)
            if coverage["n_without"] > 0:
                st.caption(f"⚠️ 'N/A' = données non disponibles (seuil : {A25_SEUIL_SIGNIFICATIF}M€)")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — COMPARAISON BUDGÉTAIRE
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="section-header">📈 Comparaison Budgétaire par Scénario</div>', unsafe_allow_html=True)
        scope_sel    = st.selectbox("Sélectionner un Scope", ["Tous"] + sorted(df_agg["Scope"].tolist()))
        df_scope_agg = df_agg if scope_sel == "Tous" else df_agg[df_agg["Scope"] == scope_sel]

        scenario_data = []
        for sc in selected_scenarios:
            if sc in df_scope_agg.columns:
                scenario_data.append({"Scénario": sc, "Valeur (M€)": df_scope_agg[sc].sum()})

        if scenario_data:
            df_scenarios = pd.DataFrame(scenario_data)
            if "YTD 08" in df_scope_agg.columns:
                df_ref = pd.DataFrame([{"Scénario": "YTD 08 (réel)", "Valeur (M€)": df_scope_agg["YTD 08"].sum()}])
                df_scenarios = pd.concat([df_ref, df_scenarios], ignore_index=True)
            colors = ["#3b82f6" if "réel" in s else "#94a3b8" for s in df_scenarios["Scénario"]]
            fig = go.Figure(go.Bar(
                x=df_scenarios["Scénario"], y=df_scenarios["Valeur (M€)"],
                marker_color=colors,
                text=df_scenarios["Valeur (M€)"].apply(lambda x: f"{x:.1f}M€"),
                textposition="outside"
            ))
            fig.update_layout(title=f"Comparaison des scénarios — {scope_sel}",
                              xaxis_title="Scénario", yaxis_title="M€", height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Tableau détaillé — tous scopes")
        if selected_scenarios:
            tbl = df_agg[["Scope"] + selected_scenarios].copy()
            st.dataframe(tbl.style.format("{:.1f}", subset=selected_scenarios, na_rep="N/A")
                        .background_gradient(subset=selected_scenarios, cmap="RdYlGn_r"),
                        use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — ANALYSE PAR DRIVER (CORRIGÉE)
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="section-header">🔍 Analyse par Driver OIE</div>', unsafe_allow_html=True)

        scope_driver = st.selectbox("Scope", sorted(df["Scope"].dropna().unique().tolist()), key="scope_driver")

        # Séparer détails et total pour ce scope
        df_scope_all    = df[df["Scope"] == scope_driver].copy()
        df_drv          = df_scope_all[~df_scope_all["_is_total"]].copy()
        df_drv_total    = df_scope_all[df_scope_all["_is_total"]].copy()

        # ✅ CORRECTION CLÉE : détecter les colonnes sans détail → fallback total
        cols_from_total = []
        for col in available_numeric:
            if col not in df_drv.columns:
                continue
            if df_drv[col].dropna().empty and not df_drv_total.empty and df_drv_total[col].notna().any():
                cols_from_total.append(col)

        # Badge A25
        scope_has_a25 = scope_driver in coverage["scopes_with"]
        if not scope_has_a25:
            st.markdown(f'<div class="a25-warning">⚠️ A25 non disponible pour <strong>{scope_driver}</strong>.</div>', unsafe_allow_html=True)

        # Badge colonnes depuis total
        if cols_from_total:
            st.markdown(
                f'<div class="data-warning">ℹ️ Pour <strong>{scope_driver}</strong>, les colonnes '
                f'<strong>{", ".join(cols_from_total)}</strong> ne sont disponibles qu\'au niveau total '
                f'(pas de détail par driver). La valeur totale est affichée dans le tableau ci-dessous.</div>',
                unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if "YTD 08" in df_drv.columns and len(df_drv) > 0:
                top_drivers = (df_drv.groupby("Drivers")["YTD 08"]
                               .sum().dropna().sort_values(key=abs, ascending=False)
                               .head(top_n).reset_index())
                top_drivers.columns = ["Driver", "YTD 08 (M€)"]
                fig = px.bar(top_drivers, x="YTD 08 (M€)", y="Driver", orientation="h",
                             title=f"Top {top_n} Drivers — {scope_driver} (YTD 08)",
                             color="YTD 08 (M€)", color_continuous_scale="RdYlGn_r", text="YTD 08 (M€)")
                fig.update_traces(texttemplate="%{text:.2f}M€", textposition="outside")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            elif len(df_drv) == 0 and not df_drv_total.empty:
                # Scope avec seulement une ligne total (pas de détail du tout)
                st.info(f"ℹ️ Seule la ligne total est disponible pour {scope_driver} — pas de détail par driver.")
                if "YTD 08" in df_drv_total.columns:
                    st.metric("YTD 08 (total)", fmt_m(df_drv_total["YTD 08"].values[0]))

        with col_b:
            if "YTD 08" in df_drv.columns and len(df_drv) > 0:
                pareto = (df_drv.groupby("Drivers")["YTD 08"]
                          .sum().dropna().abs().sort_values(ascending=False).reset_index())
                pareto.columns = ["Driver", "Abs YTD 08"]
                if pareto["Abs YTD 08"].sum() > 0:
                    pareto["Cumul %"] = pareto["Abs YTD 08"].cumsum() / pareto["Abs YTD 08"].sum() * 100
                    pareto["Rang"]    = range(1, len(pareto) + 1)
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig2.add_trace(go.Bar(x=pareto["Rang"], y=pareto["Abs YTD 08"],
                                          name="Valeur abs.", marker_color="#667eea"), secondary_y=False)
                    fig2.add_trace(go.Scatter(x=pareto["Rang"], y=pareto["Cumul %"],
                                              name="Cumul %", line=dict(color="red", width=2)), secondary_y=True)
                    fig2.add_hline(y=80, line_dash="dash", line_color="orange",
                                   annotation_text="80%", secondary_y=True)
                    fig2.update_layout(title=f"Analyse de Pareto — {scope_driver}", height=400)
                    st.plotly_chart(fig2, use_container_width=True)

        # ✅ TABLEAU DÉTAIL CORRIGÉ
        st.subheader(f"📋 Détail complet — {scope_driver}")
        detail_cols = ["Drivers"] + [c for c in ["YTD 08", "B26", "RF3", "RF1", "A25"] if c in df_drv.columns]
        df_detail   = df_drv[detail_cols].copy()

        # Calcul des écarts sur les colonnes disponibles au niveau détail
        if "YTD 08" in df_detail.columns and "B26" in df_detail.columns:
            b26_ytd_drv = df_detail["B26"] * (MOIS_YTD / 12)
            df_detail["Écart YTD vs B26"] = df_detail["YTD 08"] - b26_ytd_drv
            df_detail["Écart % vs B26"]   = safe_pct_series(df_detail["Écart YTD vs B26"], b26_ytd_drv)

        if "A25" in df_detail.columns and scope_has_a25:
            a25_ytd_drv = df_detail["A25"] * (MOIS_YTD / 12)
            ecart_a25_drv = df_detail["YTD 08"] - a25_ytd_drv if "YTD 08" in df_detail.columns else pd.Series(np.nan, index=df_detail.index)
            df_detail["Écart % vs A25"] = safe_pct_series(ecart_a25_drv, a25_ytd_drv)
        elif "A25" in df_detail.columns:
            df_detail = df_detail.drop(columns=["A25"])

        fmt_d = {c: "{:.2f}" for c in df_detail.columns if c != "Drivers" and "%" not in c}
        fmt_d.update({c: "{:.1f}%" for c in df_detail.columns if "%" in c})
        st.dataframe(df_detail.set_index("Drivers").style.format(fmt_d, na_rep="N/A"),
                     use_container_width=True)

        # ✅ LIGNE TOTAL SÉPARÉE pour les colonnes sans détail
        if cols_from_total and not df_drv_total.empty:
            st.markdown("**📌 Valeurs disponibles uniquement au niveau total (pas de détail par driver) :**")
            total_display = df_drv_total[["Scope"] + [c for c in cols_from_total if c in df_drv_total.columns]].copy()
            total_display = total_display.rename(columns={"Scope": "Niveau"})
            total_display["Niveau"] = f"Total {scope_driver}"
            fmt_t = {c: "{:.2f}" for c in cols_from_total if c in total_display.columns}
            st.dataframe(total_display.set_index("Niveau").style.format(fmt_t, na_rep="N/A"),
                         use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — VARIANCE
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="section-header">📉 Analyse de Variance</div>', unsafe_allow_html=True)

        variance_options = ["YTD 08 vs B26 (Actuel vs Budget)", "RF3 vs B26", "RF1 vs B26"]
        if coverage["n_with_a25"] > 0:
            variance_options.insert(1, "YTD 08 vs A25 (YoY)")
        else:
            st.markdown('<div class="a25-warning">ℹ️ Analyse YoY désactivée — aucune donnée A25.</div>', unsafe_allow_html=True)

        variance_type = st.radio("Type de variance", variance_options, horizontal=True)
        mapping = {
            "YTD 08 vs B26 (Actuel vs Budget)": ("B26", "YTD 08", "YTD 08 vs B26", True),
            "YTD 08 vs A25 (YoY)":              ("A25", "YTD 08", "YTD 08 vs A25", True),
            "RF3 vs B26":                        ("B26", "RF3",    "RF3 vs B26",    False),
            "RF1 vs B26":                        ("B26", "RF1",    "RF1 vs B26",    False),
        }
        col_ref, col_act, label, is_ytd = mapping[variance_type]

        if col_ref in df_agg.columns and col_act in df_agg.columns:
            var_df = df_agg[["Scope", col_act, col_ref]].copy()

            if variance_type == "YTD 08 vs A25 (YoY)" and coverage["n_with_a25"] < coverage["n_total"]:
                var_df = var_df[var_df["Scope"].isin(coverage["scopes_with"])].copy()
                st.markdown(f'<div class="a25-info">ℹ️ Analyse YoY restreinte aux <strong>{coverage["n_with_a25"]} scopes</strong> avec A25.</div>', unsafe_allow_html=True)

            if is_ytd:
                var_df[f"{col_ref}_prorata"] = var_df[col_ref] * (MOIS_YTD / 12)
                ref_col = f"{col_ref}_prorata"
            else:
                ref_col = col_ref

            var_df["Écart"]   = var_df[col_act] - var_df[ref_col]
            var_df["Écart %"] = safe_pct_series(var_df["Écart"], var_df[ref_col])

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig = px.bar(var_df, x="Scope", y="Écart", title=f"Écart absolu — {label} (M€)",
                             color="Écart", color_continuous_scale="RdYlGn_r", text="Écart")
                fig.update_traces(texttemplate="%{text:.1f}M€", textposition="outside")
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col_v2:
                fig2 = px.bar(var_df.dropna(subset=["Écart %"]), x="Scope", y="Écart %",
                              title=f"Écart % — {label}",
                              color="Écart %", color_continuous_scale="RdYlGn_r", text="Écart %")
                fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig2.add_hline(y=0, line_dash="dash", line_color="black")
                fig2.add_hline(y=alert_threshold,  line_dash="dot", line_color="orange", annotation_text=f"+{alert_threshold}%")
                fig2.add_hline(y=-alert_threshold, line_dash="dot", line_color="orange", annotation_text=f"-{alert_threshold}%")
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
                n_na = var_df["Écart %"].isna().sum()
                if n_na > 0:
                    st.caption(f"ℹ️ {n_na} scope(s) exclus du graphique % (référence < {A25_SEUIL_SIGNIFICATIF}M€ ou N/A)")

            display_cols = ["Scope", col_act, ref_col, "Écart", "Écart %"]
            fmt_v = {col_act: "{:.1f}", ref_col: "{:.1f}", "Écart": "{:.1f}", "Écart %": "{:.1f}%"}
            st.dataframe(var_df[display_cols].style.format(fmt_v, na_rep="N/A"), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5 — WATERFALL & TENDANCES
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-header">🌊 Waterfall & Tendances YTD/YTG</div>', unsafe_allow_html=True)

        scope_wf = st.selectbox("Scope pour Waterfall", ["Tous"] + sorted(df_agg["Scope"].tolist()), key="scope_wf")
        df_wf    = df_agg if scope_wf == "Tous" else df_agg[df_agg["Scope"] == scope_wf]

        wf_cols_candidates = {"A25 (FY)": "A25", "YTD 08": "YTD 08", "B26 (FY)": "B26", "RF3 (FY)": "RF3"}
        wf_vals = {}
        for label_wf, col_wf in wf_cols_candidates.items():
            if col_wf not in df_wf.columns:
                continue
            if col_wf == "A25":
                if scope_wf == "Tous":
                    if coverage["n_with_a25"] == 0: continue
                    val = df_agg[df_agg["Scope"].isin(coverage["scopes_with"])]["A25"].sum()
                else:
                    if scope_wf not in coverage["scopes_with"]: continue
                    val = df_wf["A25"].sum()
            else:
                val = df_wf[col_wf].sum()
            if not pd.isna(val):
                wf_vals[label_wf] = val

        if len(wf_vals) >= 2:
            keys     = list(wf_vals.keys())
            vals     = list(wf_vals.values())
            measures = ["absolute"] + ["relative"] * (len(vals) - 1)
            deltas   = [vals[0]] + [vals[i] - vals[i-1] for i in range(1, len(vals))]
            title_wf = f"Waterfall OIE — {scope_wf} (M€)"
            if "A25 (FY)" not in wf_vals:
                title_wf += " (A25 exclu — données non disponibles)"
            fig_wf = go.Figure(go.Waterfall(
                name="OIE", orientation="v", measure=measures, x=keys, y=deltas,
                connector={"line": {"color": "rgb(63,63,63)"}},
                increasing={"marker": {"color": "#ef4444"}},
                decreasing={"marker": {"color": "#22c55e"}},
                totals={"marker": {"color": "#3b82f6"}}
            ))
            fig_wf.update_layout(title=title_wf, height=450, showlegend=False)
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("ℹ️ Données insuffisantes pour le waterfall.")

        st.divider()
        st.subheader("📅 Analyse YTD vs YTG")
        ytd_ytg_cols = [c for c in ["YTD 08", "YTG"] if c in df_agg.columns]
        if ytd_ytg_cols:
            df_melt = df_agg[["Scope"] + ytd_ytg_cols].melt(id_vars="Scope", var_name="Métrique", value_name="M€")
            fig_ytd = px.bar(df_melt, x="Scope", y="M€", color="Métrique", barmode="group",
                             title="YTD vs YTG par Scope", color_discrete_sequence=["#3b82f6", "#f59e0b"])
            fig_ytd.update_layout(height=400)
            st.plotly_chart(fig_ytd, use_container_width=True)

        st.subheader("🎯 Projection Full Year (Run-Rate)")
        if "YTD 08" in df_agg.columns:
            proj_df = df_agg[["Scope", "YTD 08"]].copy()
            proj_df["Run-Rate mensuel"] = proj_df["YTD 08"] / MOIS_YTD
            proj_df["Projection FY"]   = proj_df["YTD 08"] * (12 / MOIS_YTD)
            if "B26" in df_agg.columns:
                proj_df = proj_df.merge(df_agg[["Scope", "B26"]], on="Scope", how="left")
                proj_df["Écart Proj vs B26"] = proj_df["Projection FY"] - proj_df["B26"]
            st.dataframe(proj_df.style.format({
                "YTD 08": "{:.1f}", "Run-Rate mensuel": "{:.2f}",
                "Projection FY": "{:.1f}", "B26": "{:.1f}", "Écart Proj vs B26": "{:.1f}"
            }, na_rep="N/A"), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6 — HEATMAP
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="section-header">🗺️ Heatmap de Performance</div>', unsafe_allow_html=True)
        heatmap_metric = st.radio("Métrique heatmap",
                                   ["Écart % vs B26 (recommandé)", "Valeur absolue (M€)"], horizontal=True)
        if selected_scenarios:
            hm_data = df_agg[["Scope"] + selected_scenarios].set_index("Scope")
            if heatmap_metric == "Écart % vs B26 (recommandé)" and "B26" in df_agg.columns:
                b26_by_scope = df_agg.set_index("Scope")["B26"]
                hm_data_pct  = pd.DataFrame(index=hm_data.index)
                for sc in selected_scenarios:
                    if sc in hm_data.columns:
                        ecart = hm_data[sc] - b26_by_scope
                        hm_data_pct[sc] = safe_pct_series(ecart, b26_by_scope)
                hm_data = hm_data_pct
                title = "Heatmap — Écart % vs B26 par Scope × Scénario"
            else:
                title = "Heatmap — Valeurs par Scope × Scénario (M€)"
            fig_hm = px.imshow(hm_data, text_auto=".1f", color_continuous_scale="RdYlGn_r",
                               title=title, aspect="auto")
            fig_hm.update_layout(height=500)
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("🔴 Dépassement | 🟢 Économies | ⬜ N/A ou non significatif")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 7 — ALERTES & R&O
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="section-header">⚠️ Alertes & Risks & Opportunities</div>', unsafe_allow_html=True)

        alerts = []
        if "YTD 08" in df.columns and "B26" in df.columns:
            # Alertes sur les détails quand disponibles, sinon sur les totaux
            for scope in df["Scope"].dropna().unique():
                df_s      = df[df["Scope"] == scope]
                df_s_det  = df_s[~df_s["_is_total"]]
                df_s_tot  = df_s[df_s["_is_total"]]

                # Utiliser les détails si disponibles, sinon le total
                rows_to_check = df_s_det if len(df_s_det) > 0 else df_s_tot

                for _, row in rows_to_check.iterrows():
                    ytd = row.get("YTD 08", np.nan)
                    b26 = row.get("B26", np.nan)
                    if pd.notna(ytd) and pd.notna(b26):
                        b26_ytd   = b26 * (MOIS_YTD / 12)
                        ecart_abs = ytd - b26_ytd
                        ecart_pct = safe_pct(ecart_abs, b26_ytd)
                        if pd.notna(ecart_pct) and abs(ecart_pct) > alert_threshold:
                            level = "🔴 CRITIQUE" if abs(ecart_pct) > alert_threshold * 2 else "🟡 ATTENTION"
                            alerts.append({
                                "Niveau":    level,
                                "Scope":     row["Scope"],
                                "Driver":    row["Drivers"],
                                "YTD 08":   round(ytd, 2),
                                "B26 YTD":  round(b26_ytd, 2),
                                "Écart M€": round(ecart_abs, 2),
                                "Écart %":  round(ecart_pct, 1)
                            })

        if alerts:
            df_alerts  = pd.DataFrame(alerts).sort_values("Écart %", key=abs, ascending=False)
            critiques  = df_alerts[df_alerts["Niveau"] == "🔴 CRITIQUE"]
            attentions = df_alerts[df_alerts["Niveau"] == "🟡 ATTENTION"]
            col_a1, col_a2, col_a3 = st.columns(3)
            col_a1.metric("🔴 Alertes critiques", len(critiques))
            col_a2.metric("🟡 Alertes attention", len(attentions))
            col_a3.metric("📊 Total alertes",      len(df_alerts))
            st.divider()
            niveau_filter      = st.multiselect("Filtrer par niveau", ["🔴 CRITIQUE", "🟡 ATTENTION"],
                                                 default=["🔴 CRITIQUE", "🟡 ATTENTION"])
            df_alerts_filtered = df_alerts[df_alerts["Niveau"].isin(niveau_filter)]
            st.dataframe(df_alerts_filtered.style.format({
                "YTD 08": "{:.2f}", "B26 YTD": "{:.2f}", "Écart M€": "{:.2f}", "Écart %": "{:.1f}%"
            }), use_container_width=True)
        else:
            st.success(f"✅ Aucune alerte — tous les écarts < {alert_threshold}%")

        st.divider()
        st.subheader("🎯 Risks & Opportunities (R&O = RF3 - B26)")
        if "RF3" in df_agg.columns and "B26" in df_agg.columns:
            ro_df = df_agg[["Scope", "RF3", "B26"]].copy()
            ro_df["R&O Delta"] = ro_df["RF3"] - ro_df["B26"]
            ro_df["Type"]      = ro_df["R&O Delta"].apply(lambda x: "🔴 Risk" if x > 0 else "🟢 Opportunity")
            ro_df = ro_df.sort_values("R&O Delta", ascending=False)
            fig_ro = px.bar(ro_df, x="Scope", y="R&O Delta", color="Type",
                            title="Deltas R&O par Scope (M€)",
                            color_discrete_map={"🔴 Risk": "#ef4444", "🟢 Opportunity": "#22c55e"},
                            text="R&O Delta")
            fig_ro.update_traces(texttemplate="%{text:.2f}M€", textposition="outside")
            fig_ro.add_hline(y=0, line_dash="dash", line_color="black")
            fig_ro.update_layout(height=400)
            st.plotly_chart(fig_ro, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 8 — EXPORT
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown('<div class="section-header">📥 Export des données</div>', unsafe_allow_html=True)
        st.subheader("📊 Export Excel multi-onglets")

        if st.button("🔄 Générer le fichier Excel", type="primary"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_agg.to_excel(writer, sheet_name="Vue d'ensemble", index=False)
                df[~df["_is_total"]].drop(columns=["_is_total"]).to_excel(writer, sheet_name="Données détail", index=False)
                if selected_scenarios:
                    df_agg[["Scope"] + selected_scenarios].to_excel(writer, sheet_name="Scénarios budgétaires", index=False)
                if alerts:
                    pd.DataFrame(alerts).to_excel(writer, sheet_name="Alertes", index=False)
                if "YTD 08" in df_agg.columns:
                    proj_exp = df_agg[["Scope", "YTD 08"]].copy()
                    proj_exp["Run-Rate mensuel"] = (proj_exp["YTD 08"] / MOIS_YTD).round(2)
                    proj_exp["Projection FY"]    = (proj_exp["YTD 08"] * 12 / MOIS_YTD).round(2)
                    if "B26" in df_agg.columns:
                        proj_exp = proj_exp.merge(df_agg[["Scope", "B26"]], on="Scope")
                        proj_exp["Écart Proj vs B26"] = (proj_exp["Projection FY"] - proj_exp["B26"]).round(2)
                    proj_exp.to_excel(writer, sheet_name="Projections FY", index=False)
                a25_cov = df_agg[["Scope"]].copy()
                a25_cov["A25 disponible"] = a25_cov["Scope"].apply(lambda s: "Oui" if s in coverage["scopes_with"] else "Non")
                if "A25" in df_agg.columns:
                    a25_cov["Valeur A25 (M€)"] = df_agg["A25"].values
                a25_cov.to_excel(writer, sheet_name="Couverture A25", index=False)

            output.seek(0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(label="⬇️ Télécharger le fichier Excel", data=output,
                               file_name=f"OIE_Tracker_Export_{timestamp}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("✅ Fichier Excel généré avec 6 onglets !")

        st.divider()
        st.subheader("📋 Export CSV rapide")
        csv_data = df[~df["_is_total"]].drop(columns=["_is_total"]).to_csv(index=False, sep=";", encoding="utf-8-sig")
        st.download_button(label="⬇️ Télécharger CSV (détails uniquement)", data=csv_data,
                           file_name=f"OIE_detail_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

else:
    st.title("📊 OIE Tracker — EM&S Finance")
    st.markdown(f"""
    ### Bienvenue dans l'OIE Tracker
    Chargez votre fichier CSV dans la **barre latérale gauche** pour commencer.

    | Section | Description |
    |---------|-------------|
    | 📊 Vue d'ensemble | KPIs globaux et répartition par scope |
    | 📈 Comparaison Budgétaire | RF3, RF1, B26, H27, B27 |
    | 🔍 Analyse par Driver | Drill-down et analyse de Pareto |
    | 📉 Variance | Écarts YTD vs B26, A25, RF3 vs B26, RF1 vs B26 |
    | 🌊 Waterfall & Tendances | Trajectoire et projections FY |
    | 🗺️ Heatmap | Matrice scope × scénario |
    | ⚠️ Alertes & R&O | Détection automatique des risques |
    | 📥 Export | Excel 6 onglets + CSV |

    **Gestion automatique totaux/détails** : colonne par colonne — détails si disponibles, sinon total.
    **Seuil safe_pct** : {A25_SEUIL_SIGNIFICATIF}M€ — % masqué si référence trop faible.
    """)