import os
import time
import math
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
from PyPDF2 import PdfReader
from google import genai


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard Mensual + IA Generativa", layout="wide")

MESES_ORDEN = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

MESES_VENTAS = {
    "Enero": "Ene_KG",
    "Febrero": "Feb_KG",
    "Marzo": "Mar_KG",
    "Abril": "Abr_KG",
    "Mayo": "May_KG",
    "Junio": "Jun_KG",
    "Julio": "Jul_KG",
    "Agosto": "Ago_KG",
    "Septiembre": "Sep_KG",
    "Octubre": "Oct_KG",
    "Noviembre": "Nov_KG",
    "Diciembre": "Dic_KG",
}

MESES_PRES = {
    "Enero": "ENE",
    "Febrero": "FEB",
    "Marzo": "MAR",
    "Abril": "ABR",
    "Mayo": "MAY",
    "Junio": "JUN",
    "Julio": "JUL",
    "Agosto": "AGO",
    "Septiembre": "SEP",
    "Octubre": "OCT",
    "Noviembre": "NOV",
    "Diciembre": "DIC",
}

MANUAL_DIR = "manual_tecnico"
MANUAL_FALLBACK_ROOT = "Manual_tecnico_preventa.pdf"

DEFAULT_MODEL_PREFS = [
    # Ojo: disponibilidad real depende de tu cuenta / API. El app intenta descubrir modelos con list().
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
]


# =========================
# GEMINI (GOOGLE AI STUDIO)
# =========================
def _get_gemini_client() -> genai.Client:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Falta `GEMINI_API_KEY` en Secrets (Streamlit Cloud → Settings → Secrets).")
        st.stop()
    return genai.Client(api_key=api_key)

@st.cache_data(show_spinner=False, ttl=3600)
def list_available_models() -> List[str]:
    """Intenta listar modelos disponibles en tu cuenta.
    Si falla (por permisos/red), regresamos lista vacía y usamos fallback."""
    try:
        client = _get_gemini_client()
        models = client.models.list()
        names = []
        for m in models:
            # m.name suele venir como "models/xxx"
            name = getattr(m, "name", "") or ""
            if name.startswith("models/"):
                name = name.replace("models/", "", 1)
            if name:
                names.append(name)
        # orden estable, sin duplicados
        seen = set()
        out = []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out
    except Exception:
        return []

def gemini_generate(prompt: str, model_preference: Optional[str] = None, max_retries: int = 2) -> str:
    client = _get_gemini_client()

    available = list_available_models()
    candidates: List[str] = []

    if model_preference and model_preference.strip():
        candidates.append(model_preference.strip())

    # Preferidos + disponibles
    for m in DEFAULT_MODEL_PREFS:
        candidates.append(m)

    # Si logramos listar, añadimos algunos de los disponibles
    # (sin reventar la lista)
    for m in available[:20]:
        candidates.append(m)

    # únicos
    seen = set()
    model_candidates = []
    for c in candidates:
        if c and c not in seen:
            model_candidates.append(c)
            seen.add(c)

    last_err = None
    for model_name in model_candidates:
        for attempt in range(max_retries + 1):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = getattr(resp, "text", None)
                if text:
                    return text
                # fallback por si viene en otro campo
                return str(resp)
            except Exception as e:
                last_err = e
                # si es 503 (alta demanda), esperamos y reintentamos
                msg = str(e).lower()
                if "503" in msg or "unavailable" in msg or "high demand" in msg:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                # si es not found / invalid model, probamos siguiente modelo
                break

    raise RuntimeError(f"No pude generar respuesta con Gemini. Último error: {last_err}")


# =========================
# MANUAL PDF (RAG LOCAL SIN VECTOR STORE)
# =========================
@st.cache_data(show_spinner=False)
def find_manual_pdf() -> Optional[str]:
    # 1) manual_tecnico/*.pdf
    if os.path.isdir(MANUAL_DIR):
        pdfs = [f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")]
        if pdfs:
            pdfs.sort()
            return os.path.join(MANUAL_DIR, pdfs[0])

    # 2) fallback root
    if os.path.isfile(MANUAL_FALLBACK_ROOT):
        return MANUAL_FALLBACK_ROOT

    # 3) buscar cualquier pdf en raíz
    try:
        pdfs_root = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
        if pdfs_root:
            pdfs_root.sort()
            return pdfs_root[0]
    except Exception:
        pass

    return None

@st.cache_data(show_spinner=False)
def load_manual_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts).strip()

def _clean_text(s: str) -> str:
    s = s.replace("\ufeff", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0:
            i = 0
        if i >= len(text):
            break
    return chunks

def score_chunk(query: str, chunk: str) -> float:
    # Scoring simple por overlap de palabras (robusto, sin sklearn)
    q = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+", query.lower())
    c = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+", chunk.lower())
    if not q or not c:
        return 0.0
    qset = set(q)
    cset = set(c)
    inter = len(qset.intersection(cset))
    # pondera un poquito por longitud
    return inter / math.sqrt(len(cset) + 1)

def retrieve_context(query: str, chunks: List[str], k: int = 4) -> Tuple[str, List[Tuple[float, str]]]:
    scored = [(score_chunk(query, ch), ch) for ch in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [x for x in scored[:k] if x[0] > 0]
    context = "\n\n---\n\n".join([_clean_text(t[1]) for t in top])
    return context, top


# =========================
# DATA PIPELINE (VENTAS + PRESUPUESTO)
# =========================
def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def normalize_ventas(df: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    # columnas base opcionales
    base_cols = {}
    for col in ["ItemCode", "ItemName", "SlpName", "CardCode", "CardName", "Cliente", "CodigoCliente"]:
        if col in df.columns:
            base_cols[col] = df[col]
    base = pd.DataFrame(base_cols) if base_cols else pd.DataFrame(index=df.index)

    for mes, col in MESES_VENTAS.items():
        if col not in df.columns:
            continue
        tmp = base.copy()
        tmp["anio"] = int(year)
        tmp["mes"] = mes
        tmp["actual_kg"] = _to_num(df[col])
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=list(base.columns) + ["anio","mes","actual_kg"])
    return pd.concat(rows, ignore_index=True)

def normalize_pres(df: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    base_cols = {}
    for col in ["ItemCode", "ItemName", "SlpName", "CardCode", "CardName", "Cliente", "CodigoCliente"]:
        if col in df.columns:
            base_cols[col] = df[col]
    base = pd.DataFrame(base_cols) if base_cols else pd.DataFrame(index=df.index)

    for mes, col in MESES_PRES.items():
        if col not in df.columns:
            continue
        tmp = base.copy()
        tmp["anio"] = int(year)
        tmp["mes"] = mes
        tmp["budget_kg"] = _to_num(df[col])
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=list(base.columns) + ["anio","mes","budget_kg"])
    return pd.concat(rows, ignore_index=True)

def build_model(dfv: pd.DataFrame, dfp: pd.DataFrame, year: int) -> pd.DataFrame:
    v = normalize_ventas(dfv, year)
    p = normalize_pres(dfp, year)

    # keys comunes (mínimo)
    keys = ["anio", "mes"]
    for k in ["ItemCode", "SlpName", "CardCode", "CardName", "Cliente", "CodigoCliente", "ItemName"]:
        if k in v.columns and k in p.columns:
            keys.append(k)

    df = v.merge(p, on=keys, how="left")
    if "budget_kg" not in df.columns:
        df["budget_kg"] = 0
    df["budget_kg"] = df["budget_kg"].fillna(0)

    df["var_kg"] = df["actual_kg"] - df["budget_kg"]
    df["cumpl_pct"] = df.apply(lambda r: (r["actual_kg"] / r["budget_kg"] * 100) if r["budget_kg"] > 0 else 0, axis=1)

    # orden de meses (categorical)
    df["mes"] = pd.Categorical(df["mes"], categories=MESES_ORDEN, ordered=True)
    return df

def months_with_real_sales(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    by = df.groupby("mes", observed=True)["actual_kg"].sum()
    real = [str(m) for m, v in by.items() if float(v) > 0]
    # mantener orden
    out = [m for m in MESES_ORDEN if m in real]
    return out

def filter_df_for_kpis(df: pd.DataFrame, meses_sel: List[str], vendedor_sel: List[str], only_real_months: bool) -> pd.DataFrame:
    d = df.copy()
    if meses_sel:
        d = d[d["mes"].isin(meses_sel)]
    if vendedor_sel and "SlpName" in d.columns:
        d = d[d["SlpName"].isin(vendedor_sel)]
    if only_real_months:
        # Solo meses donde actual>0 (dentro del filtro actual)
        real = months_with_real_sales(d)
        if real:
            d = d[d["mes"].isin(real)]
        else:
            # si no hay ventas reales, deja vacío para evitar penalizar
            d = d.iloc[0:0]
    return d

def kpis(df: pd.DataFrame) -> Dict[str, float]:
    a = float(df["actual_kg"].sum()) if not df.empty else 0.0
    b = float(df["budget_kg"].sum()) if not df.empty else 0.0
    v = a - b
    c = (a / b * 100) if b > 0 else 0.0
    return {"actual": a, "budget": b, "var": v, "cumpl": c}


# =========================
# FORECAST + RIESGO (MBA-LIKE)
# =========================
def simple_forecast_monthly(by_mes: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Forecast simple: tendencia lineal sobre meses con actual>0."""
    if by_mes.empty:
        return by_mes

    df = by_mes.copy()
    df = df.sort_values("mes")

    # índice temporal (0..n-1)
    df = df.reset_index(drop=True)
    df["t"] = range(len(df))

    # Solo puntos con actual>0 para aprender tendencia
    train = df[df["actual_kg"] > 0].copy()
    if len(train) < 2:
        # insuficiente para tendencia: repetir último valor
        last = float(train["actual_kg"].iloc[-1]) if len(train) == 1 else 0.0
        preds = []
        start_idx = len(df)
        for i in range(horizon):
            preds.append({"t": start_idx + i, "pred_kg": last})
        pred = pd.DataFrame(preds)
        return pred

    # regresión lineal manual (sin sklearn)
    x = train["t"].astype(float).values
    y = train["actual_kg"].astype(float).values
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    slope = (((x - x_mean) * (y - y_mean)).sum() / denom) if denom != 0 else 0.0
    intercept = y_mean - slope * x_mean

    preds = []
    start_idx = len(df)
    for i in range(horizon):
        t = start_idx + i
        pred = max(0.0, intercept + slope * t)
        preds.append({"t": t, "pred_kg": pred})
    return pd.DataFrame(preds)

def risk_signals(df: pd.DataFrame) -> List[Dict[str, str]]:
    signals = []
    if df.empty:
        return signals

    # 1) Data completeness
    real_months = months_with_real_sales(df)
    if len(real_months) <= 2:
        signals.append({
            "nivel": "MEDIO",
            "tema": "Cobertura de datos",
            "detalle": "El archivo mensual solo trae 1–2 meses con ventas reales. El KPI anual no debe compararse contra presupuesto anual completo.",
            "accion": "Usar KPIs solo sobre meses con ventas reales (radio en sidebar) y actualizar mensualmente."
        })

    # 2) Variance risk (top negative)
    by_mes = df.groupby("mes", observed=True)[["actual_kg","budget_kg","var_kg"]].sum().reset_index()
    worst = by_mes.sort_values("var_kg").head(1)
    if not worst.empty and float(worst["var_kg"].iloc[0]) < 0:
        signals.append({
            "nivel": "ALTO",
            "tema": "Brecha vs presupuesto",
            "detalle": f"Mes con mayor brecha negativa: {worst['mes'].iloc[0]} ({worst['var_kg'].iloc[0]:,.0f} kg).",
            "accion": "Revisar mix, capacidad y pipeline comercial; activar plan de recuperación para el mes siguiente."
        })

    # 3) Concentration risk (vendor)
    if "SlpName" in df.columns:
        by_v = df.groupby("SlpName")["actual_kg"].sum().sort_values(ascending=False)
        if len(by_v) >= 1:
            top_share = float(by_v.iloc[0] / max(1e-9, by_v.sum()) * 100)
            if top_share > 70:
                signals.append({
                    "nivel": "MEDIO",
                    "tema": "Concentración por vendedor",
                    "detalle": f"El vendedor top concentra {top_share:.1f}% del volumen filtrado.",
                    "accion": "Mitigar dependencia: cobertura cruzada, desarrollo de cuentas adicionales, backups."
                })

    return signals


# =========================
# UI THEME (SGA-STYLE HOME)
# =========================
def inject_css():
    st.markdown("""
    <style>
      .hero {
        padding: 28px 24px;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(255,255,255,0.04), rgba(255,255,255,0.0));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 18px;
      }
      .hero h1 { margin-bottom: 6px; }
      .subtle { color: rgba(255,255,255,0.70); }
      .card {
        padding: 16px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        height: 100%;
      }
      .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        color: rgba(255,255,255,0.80);
        font-size: 12px;
        margin-bottom: 8px;
      }
      .stepTitle { font-weight: 700; font-size: 14px; margin-bottom: 6px; }
      .stepText { color: rgba(255,255,255,0.75); font-size: 13px; line-height: 1.35; }
      .divider {
        height: 1px; background: rgba(255,255,255,0.10);
        margin: 18px 0;
      }
      .kpiHint { color: rgba(255,255,255,0.65); font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

def home_sga():
    inject_css()

    st.markdown("""
    <div class="hero">
      <h1>SGA — Sistema Generativo Aplicado (Empaque Flexible B2B)</h1>
      <div class="subtle">
        Metodología en 6 pasos para convertir datos de ventas + conocimiento técnico en decisiones comerciales medibles, repetibles y escalables.
      </div>
      <div class="divider"></div>
      <div class="subtle">
        <b>Entregable:</b> Dashboard mensual (KG) + IA ejecutiva + IA técnica preventa + Forecast & Riesgo + Exportación ejecutiva.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
          <div class="badge">PASO 1</div>
          <div class="stepTitle">OBJETIVO</div>
          <div class="stepText">Alinear dirección comercial: cumplimiento vs presupuesto (KG) y decisiones por mes/vendedor/cliente.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
          <div class="badge">PASO 2</div>
          <div class="stepTitle">INPUT</div>
          <div class="stepText">Excel mensual de ventas + presupuesto. Actualización mensual (no penaliza meses futuros sin ventas).</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
          <div class="badge">PASO 3</div>
          <div class="stepTitle">ESTRUCTURACIÓN</div>
          <div class="stepText">Normalización por mes (long format) + KPIs + señales de riesgo + base para IA.</div>
        </div>
        """, unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("""
        <div class="card">
          <div class="badge">PASO 4</div>
          <div class="stepTitle">WORKFLOW</div>
          <div class="stepText">Tabs: Carga → Dashboard → IA ejecutiva → Forecast/Riesgo → IA técnica preventa (manual PDF).</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown("""
        <div class="card">
          <div class="badge">PASO 5</div>
          <div class="stepTitle">OUTPUT</div>
          <div class="stepText">Informe ejecutivo (IA) + recomendaciones + propuestas técnicas basadas SOLO en el manual.</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown("""
        <div class="card">
          <div class="badge">PASO 6</div>
          <div class="stepTitle">MÉTRICA</div>
          <div class="stepText">ROI simulado: cumplimiento, brechas, riesgos, forecast vs presupuesto y exportación para comité.</div>
        </div>
        """, unsafe_allow_html=True)

    st.info("Siguiente paso: ve a **“Cargar archivos (mensual)”** para subir ventas + presupuesto y activar todo el sistema.")


# =========================
# APP
# =========================
st.title("📊 Dashboard Mensual + IA Generativa")

tabs = st.tabs([
    "Inicio (SGA)",
    "Cargar archivos (mensual)",
    "Dashboard",
    "IA del Dashboard",
    "Forecast & Riesgo",
    "IA Técnica Preventa",
])

# -------------
# TAB: HOME
# -------------
with tabs[0]:
    home_sga()

# -------------
# TAB: CARGA
# -------------
with tabs[1]:
    st.subheader("1) Cargar archivos (mensual)")

    colA, colB = st.columns(2)
    with colA:
        ventas_file = st.file_uploader("Reporte de Ventas mensual (.xlsx)", type=["xlsx"], key="ventas_file")
    with colB:
        pres_file = st.file_uploader("Presupuesto (.xlsx)", type=["xlsx"], key="pres_file")

    year = st.number_input("Año", min_value=2020, max_value=2100, value=2026, step=1)

    if st.button("Procesar", type="primary"):
        if not ventas_file or not pres_file:
            st.error("Sube ambos archivos: ventas + presupuesto.")
        else:
            dfv = pd.read_excel(ventas_file)
            dfp = pd.read_excel(pres_file)

            df = build_model(dfv, dfp, int(year))
            st.session_state["df_model"] = df

            real = months_with_real_sales(df)
            st.success("✅ Datos procesados correctamente.")
            st.info(f"Meses detectados con datos reales: {', '.join(real) if real else 'Ninguno'}")

            with st.expander("Ver muestra (primeras filas)"):
                st.dataframe(df.head(30), use_container_width=True)

            st.caption("Tip: si tu Excel mensual solo trae 1–2 meses con números, usa KPIs solo sobre meses con ventas reales (radio en sidebar).")

# -------------
# SIDEBAR FILTROS (para tabs 2..)
# -------------
df_global = st.session_state.get("df_model", None)
if isinstance(df_global, pd.DataFrame) and not df_global.empty:
    st.sidebar.header("Filtros")

    years = sorted([int(x) for x in df_global["anio"].dropna().unique().tolist()]) if "anio" in df_global.columns else [2026]
    anio_sel = st.sidebar.selectbox("Año", years, index=len(years)-1)
    df_f = df_global[df_global["anio"] == anio_sel].copy()

    meses_all = [m for m in MESES_ORDEN if m in df_f["mes"].astype(str).unique().tolist()]
    if not meses_all:
        meses_all = MESES_ORDEN

    meses_sel = st.sidebar.multiselect("Mes (análisis)", MESES_ORDEN, default=months_with_real_sales(df_f) or MESES_ORDEN)

    vendedores = sorted(df_f["SlpName"].dropna().unique().tolist()) if "SlpName" in df_f.columns else []
    vend_sel = st.sidebar.multiselect("Vendedor", vendedores, default=vendedores)

    only_real = st.sidebar.radio(
        "KPIs basados en:",
        ["Solo meses con ventas reales", "Todos los meses seleccionados"],
        index=0
    ) == "Solo meses con ventas reales"
else:
    anio_sel, meses_sel, vend_sel, only_real = None, [], [], True

# -------------
# TAB: DASHBOARD
# -------------
with tabs[2]:
    st.subheader("2) Dashboard (KG)")

    if not isinstance(df_global, pd.DataFrame) or df_global.empty:
        st.warning("Primero carga y procesa tus Excel en la pestaña **Cargar archivos (mensual)**.")
    else:
        df_f = df_global[df_global["anio"] == anio_sel].copy()
        df_kpi = filter_df_for_kpis(df_f, meses_sel, vend_sel, only_real)

        m = kpis(df_kpi)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual (KG)", f"{m['actual']:,.0f}")
        c2.metric("Budget (KG)", f"{m['budget']:,.0f}")
        c3.metric("Varianza (KG)", f"{m['var']:,.0f}")
        c4.metric("% Cumplimiento", f"{m['cumpl']:.1f}%")

        st.caption("Si solo hay 1–2 meses con ventas reales, el % se calcula SOLO sobre esos meses (cuando ese modo está activo).")

        # Tendencia mensual: mostramos siempre meses seleccionados, pero si only_real=True se recorta
        by_mes = df_kpi.groupby("mes", observed=True)[["actual_kg","budget_kg"]].sum().reset_index()

        if not by_mes.empty:
            # asegurar columna mes string para plot (por categorical)
            by_mes["mes"] = by_mes["mes"].astype(str)
            by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
            by_mes = by_mes.sort_values("mes")

            st.markdown("### Tendencia mensual (Actual vs Budget)")
            st.plotly_chart(
                px.line(by_mes, x="mes", y=["actual_kg", "budget_kg"], markers=True),
                use_container_width=True
            )
        else:
            st.info("No hay datos en el filtro actual (o no hay ventas reales aún).")

        # Cumplimiento por vendedor
        if "SlpName" in df_kpi.columns and not df_kpi.empty:
            st.markdown("### Cumplimiento por vendedor (%)")
            by_v = df_kpi.groupby("SlpName", as_index=False)[["actual_kg","budget_kg"]].sum()
            by_v["cumpl_pct"] = by_v.apply(lambda r: (r["actual_kg"]/r["budget_kg"]*100) if r["budget_kg"]>0 else 0, axis=1)
            by_v = by_v.sort_values("cumpl_pct", ascending=False)
            st.plotly_chart(px.bar(by_v, x="SlpName", y="cumpl_pct"), use_container_width=True)

        # Export base
        st.markdown("### Exportación rápida")
        colx1, colx2 = st.columns(2)
        with colx1:
            csv = df_kpi.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar datos filtrados (CSV)", data=csv, file_name="datos_filtrados.csv", mime="text/csv")
        with colx2:
            # Excel en memoria
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_kpi.to_excel(writer, index=False, sheet_name="datos_filtrados")
            st.download_button(
                "⬇️ Descargar datos filtrados (Excel)",
                data=output.getvalue(),
                file_name="datos_filtrados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# -------------
# TAB: IA DASHBOARD
# -------------
with tabs[3]:
    st.subheader("3) IA del Dashboard (análisis ejecutivo)")
    st.caption("Genera un informe (sin inventar cifras) con base en los datos agregados por mes.")

    if not isinstance(df_global, pd.DataFrame) or df_global.empty:
        st.warning("Primero carga y procesa tus Excel.")
    else:
        df_f = df_global[df_global["anio"] == anio_sel].copy()
        df_kpi = filter_df_for_kpis(df_f, meses_sel, vend_sel, only_real)

        # Modelo (auto)
        available = list_available_models()
        model_default = None
        for m in DEFAULT_MODEL_PREFS:
            if m in available:
                model_default = m
                break
        if not model_default:
            model_default = (available[0] if available else DEFAULT_MODEL_PREFS[0])

        model_name = st.text_input("Modelo Gemini (opcional)", value=model_default)

        by_mes = df_kpi.groupby("mes", observed=True)[["actual_kg","budget_kg","var_kg"]].sum().reset_index()
        if not by_mes.empty:
            by_mes["mes"] = by_mes["mes"].astype(str)
            by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
            by_mes = by_mes.sort_values("mes")
        resumen = by_mes.to_string(index=False)

        if st.button("Generar análisis con IA", type="primary"):
            if df_kpi.empty:
                st.warning("No hay datos para analizar con el filtro actual.")
            else:
                prompt = f"""
Eres un consultor ejecutivo (MBA) para una empresa de empaque plástico flexible B2B.
Debes analizar desempeño de volumen (KG) vs presupuesto.

REGLAS:
- USA SOLO los datos en la tabla.
- No inventes cifras.
- Si faltan meses porque el reporte es mensual, explícitalo como 'meses aún no cargados' (no como 'cero ventas').

TABLA (agregado por mes):
{resumen}

ENTREGA (en español, claro y ejecutivo):
1) Resumen ejecutivo (máx 6 líneas)
2) Hallazgos clave (bullets)
3) Recomendaciones (bullets accionables)
4) Riesgos / alertas (bullets)
5) Próximos pasos para el siguiente mes (bullets)
"""
                try:
                    ans = gemini_generate(prompt, model_preference=model_name)
                    st.markdown(ans)

                    # Export reporte md
                    st.markdown("### Exportación ejecutiva")
                    md = f"# Informe ejecutivo (IA)\n\n{ans}\n"
                    st.download_button("⬇️ Descargar informe (Markdown)", data=md.encode("utf-8"), file_name="informe_ejecutivo.md", mime="text/markdown")

                except Exception as e:
                    st.error("No se pudo generar el análisis. Revisa tu GEMINI_API_KEY y el modelo.")
                    st.exception(e)

# -------------
# TAB: FORECAST & RIESGO
# -------------
with tabs[4]:
    st.subheader("4) Forecast & Riesgo (MBA)")
    st.caption("Forecast simple + señales de riesgo para discusión ejecutiva.")

    if not isinstance(df_global, pd.DataFrame) or df_global.empty:
        st.warning("Primero carga y procesa tus Excel.")
    else:
        df_f = df_global[df_global["anio"] == anio_sel].copy()
        df_kpi = filter_df_for_kpis(df_f, meses_sel, vend_sel, only_real)

        if df_kpi.empty:
            st.info("No hay datos suficientes con el filtro actual.")
        else:
            by_mes = df_kpi.groupby("mes", observed=True)[["actual_kg","budget_kg","var_kg"]].sum().reset_index()
            by_mes["mes"] = by_mes["mes"].astype(str)
            by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
            by_mes = by_mes.sort_values("mes")

            st.markdown("### Serie mensual (base)")
            st.plotly_chart(px.line(by_mes, x="mes", y=["actual_kg","budget_kg"], markers=True), use_container_width=True)

            st.markdown("### Forecast (próximos 3 meses, simple)")
            pred = simple_forecast_monthly(by_mes, horizon=3)
            if not pred.empty:
                # map t->mes futuro aproximado (sólo para display bonito)
                # si hay n meses visibles, el siguiente mes es el siguiente en MESES_ORDEN relativo al último mes real
                observed_months = [m for m in MESES_ORDEN if m in by_mes["mes"].astype(str).tolist()]
                last_mes = None
                # mejor: último mes con actual>0
                tmp = by_mes[by_mes["actual_kg"] > 0]
                if not tmp.empty:
                    last_mes = str(tmp.iloc[-1]["mes"])
                else:
                    last_mes = str(by_mes.iloc[-1]["mes"])

                start_idx = MESES_ORDEN.index(last_mes) if last_mes in MESES_ORDEN else 0
                labels = []
                for i in range(len(pred)):
                    labels.append(MESES_ORDEN[(start_idx + 1 + i) % 12])
                pred_disp = pred.copy()
                pred_disp["mes"] = labels

                st.plotly_chart(px.bar(pred_disp, x="mes", y="pred_kg"), use_container_width=True)
            else:
                st.info("No se pudo calcular forecast (datos insuficientes).")

            st.markdown("### Señales de riesgo")
            signals = risk_signals(df_kpi)
            if not signals:
                st.success("Sin alertas relevantes con el filtro actual.")
            else:
                for s in signals:
                    st.warning(f"**{s['nivel']} — {s['tema']}**\n\n{s['detalle']}\n\n**Acción sugerida:** {s['accion']}")

# -------------
# TAB: IA TÉCNICA PREVENTA
# -------------
with tabs[5]:
    st.subheader("5) IA Técnica Preventa (basada SOLO en el manual)")
    st.caption("Responde únicamente con evidencia del manual PDF. Si falta info, pregunta (no inventa).")

    pdf_path = find_manual_pdf()
    if not pdf_path:
        st.error(
            "No encontré el PDF del manual.\n\n"
            "✅ Sube tu PDF al repo en la carpeta `manual_tecnico/` (recomendado) "
            "o déjalo en la raíz como `Manual_tecnico_preventa.pdf`."
        )
    else:
        st.caption(f"Manual detectado: `{pdf_path}`")
        manual_text = load_manual_text(pdf_path)

        if not manual_text:
            st.error("Pude abrir el PDF, pero no pude extraer texto. Si es escaneado, expórtalo como PDF con texto (no imagen).")
        else:
            chunks = chunk_text(manual_text, chunk_size=1100, overlap=180)

            # chat history
            if "chat_preventa" not in st.session_state:
                st.session_state["chat_preventa"] = []

            for role, content in st.session_state["chat_preventa"]:
                with st.chat_message(role):
                    st.markdown(content)

            pregunta = st.chat_input("Ej: Café 500g, VFFS, vida útil 12 meses, bolsa con válvula.")
            if pregunta:
                st.session_state["chat_preventa"].append(("user", pregunta))
                with st.chat_message("user"):
                    st.markdown(pregunta)

                context, top = retrieve_context(pregunta, chunks, k=5)
                if not context:
                    # si no hay overlap, igualmente pedimos datos faltantes y advertimos
                    context = _clean_text(manual_text[:2000])

                prompt = f"""
Eres un ingeniero preventa experto en empaque plástico flexible (bolsa y bobina) para clientes B2B.

REGLAS CRÍTICAS:
- Responde SOLO usando el CONTEXTO del manual.
- Si el contexto NO trae una especificación necesaria, NO inventes: pregunta por el dato faltante.
- Si el usuario dio datos incompletos (producto, proceso, vida útil, barrera, máquina, formato), solicita lo mínimo indispensable.

FORMATO DE SALIDA (obligatorio):
1) Preguntas faltantes (si aplica)
2) Opción A segura (máxima protección)
3) Opción B optimizada costo
4) Riesgos / trade-offs
5) Evidencia del manual (citas cortas o referencias textuales del contexto)

CONTEXTO (extractos del manual):
{context}

PREGUNTA DEL USUARIO:
{pregunta}
"""
                # modelo (auto)
                available = list_available_models()
                model_default = None
                for m in DEFAULT_MODEL_PREFS:
                    if m in available:
                        model_default = m
                        break
                if not model_default:
                    model_default = (available[0] if available else DEFAULT_MODEL_PREFS[0])

                try:
                    ans = gemini_generate(prompt, model_preference=model_default)
                except Exception as e:
                    ans = f"❌ No pude generar respuesta con Gemini. Error: {e}"

                st.session_state["chat_preventa"].append(("assistant", ans))
                with st.chat_message("assistant"):
                    st.markdown(ans)

        with st.expander("Diagnóstico (para depurar rutas en Streamlit Cloud)"):
            st.write("Contenido raíz:", os.listdir("."))
            if os.path.isdir(MANUAL_DIR):
                st.write("Contenido manual_tecnico:", os.listdir(MANUAL_DIR))
