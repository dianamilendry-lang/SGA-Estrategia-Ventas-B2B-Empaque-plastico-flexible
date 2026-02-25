import os
import re
import time
from typing import List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
from PyPDF2 import PdfReader
from google import genai


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Dashboard Mensual + IA Generativa", layout="wide")

APP_TITLE = "📊 Dashboard Mensual + IA Generativa"

MANUAL_DIR = "manual_tecnico"  # carpeta recomendada en repo
MANUAL_FALLBACK_ROOT = True    # si no existe en carpeta, busca en raíz

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
    "Enero": "ENE", "Febrero": "FEB", "Marzo": "MAR", "Abril": "ABR",
    "Mayo": "MAY", "Junio": "JUN", "Julio": "JUL", "Agosto": "AGO",
    "Septiembre": "SEP", "Octubre": "OCT", "Noviembre": "NOV", "Diciembre": "DIC",
}

# =========================================================
# UTILIDADES GENERALES
# =========================================================
def _norm_colname(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"\s+", " ", c)
    return c

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


# =========================================================
# GEMINI (Google AI Studio / Gemini API)
# =========================================================
def gemini_client() -> genai.Client:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.error("Falta `GEMINI_API_KEY` en Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

@st.cache_data(show_spinner=False)
def gemini_list_models_cached() -> List[str]:
    """
    Lista modelos disponibles para tu API Key.
    Si falla, devuelve una lista mínima sugerida.
    """
    try:
        client = gemini_client()
        models = []
        for m in client.models.list():
            # m.name suele venir como "models/gemini-..." -> guardamos tal cual o recortamos
            name = getattr(m, "name", "") or ""
            if name:
                models.append(name.replace("models/", ""))
        models = sorted(list(set(models)))
        if models:
            return models
    except Exception:
        pass

    # fallback (no garantizado, pero útil si list() no funciona)
    return [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

def gemini_generate(prompt: str, model_name: str, retries: int = 3, backoff_sec: float = 1.5) -> str:
    """
    Genera contenido con reintentos para manejar picos (503) o errores transitorios.
    """
    client = gemini_client()
    last_err = None
    for i in range(retries):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            # backoff simple
            time.sleep(backoff_sec * (i + 1))
    raise RuntimeError(f"No pude generar respuesta con Gemini. Último error: {last_err}")


# =========================================================
# MANUAL PDF (RAG light sin embeddings)
# =========================================================
@st.cache_data(show_spinner=False)
def find_manual_pdf() -> str | None:
    # 1) buscar en manual_tecnico/
    if os.path.isdir(MANUAL_DIR):
        pdfs = [f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")]
        pdfs.sort()
        if pdfs:
            return os.path.join(MANUAL_DIR, pdfs[0])

    # 2) fallback: buscar en raíz
    if MANUAL_FALLBACK_ROOT:
        pdfs = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
        pdfs.sort()
        if pdfs:
            return pdfs[0]

    return None

@st.cache_data(show_spinner=False)
def load_manual_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    chunks = []
    for p in reader.pages:
        chunks.append(p.extract_text() or "")
    return "\n".join(chunks).strip()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j])
        i = max(j - overlap, j)
    return out

def rank_chunks_by_keywords(chunks: List[str], query: str, top_k: int = 4) -> List[Tuple[int, float, str]]:
    """
    Ranking simple: coincidencias de tokens (sin librerías extra).
    """
    q = re.sub(r"[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ ]+", " ", query.lower())
    q_tokens = [t for t in q.split() if len(t) >= 3]
    if not q_tokens:
        return []

    scored = []
    for idx, ch in enumerate(chunks):
        t = ch.lower()
        score = 0.0
        for tok in q_tokens:
            score += t.count(tok)
        if score > 0:
            scored.append((idx, score, ch))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# =========================================================
# NORMALIZACIÓN (mensual)
# =========================================================
def normalizar_ventas(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    # Normaliza nombres de columnas por si vienen con espacios raros
    df = df.copy()
    df.columns = [_norm_colname(c) for c in df.columns]

    # columnas mínimas (si existen)
    possible_cols = [
        "SlpName",
        "Código de cliente/proveedor",
        "Nombre de cliente",
        "ItemCode",
        "ItemName",
        "UM",
    ]
    id_cols = [c for c in possible_cols if c in df.columns]
    if "ItemCode" not in id_cols and "ItemCode" in df.columns:
        id_cols.append("ItemCode")

    out = []
    for mes, col_kg in MESES_VENTAS.items():
        if col_kg not in df.columns:
            continue
        tmp = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
        tmp["anio"] = anio
        tmp["mes"] = mes
        tmp["actual_kg"] = pd.to_numeric(df[col_kg], errors="coerce").fillna(0)
        out.append(tmp)

    if not out:
        return pd.DataFrame(columns=(id_cols + ["anio","mes","actual_kg"]))

    return pd.concat(out, ignore_index=True)

def normalizar_presupuesto(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_colname(c) for c in df.columns]

    id_cols = [c for c in ["ItemCode"] if c in df.columns]
    if "ItemCode" not in id_cols:
        # si no hay ItemCode no podemos cruzar bien
        return pd.DataFrame(columns=["ItemCode","anio","mes","budget_kg"])

    out = []
    for mes, col_kg in MESES_PRES.items():
        if col_kg not in df.columns:
            continue
        tmp = df[id_cols].copy()
        tmp["anio"] = anio
        tmp["mes"] = mes
        tmp["budget_kg"] = pd.to_numeric(df[col_kg], errors="coerce").fillna(0)
        out.append(tmp)

    if not out:
        return pd.DataFrame(columns=["ItemCode","anio","mes","budget_kg"])

    return pd.concat(out, ignore_index=True)

def preparar_df_final(dfv: pd.DataFrame, dfp: pd.DataFrame, anio: int) -> pd.DataFrame:
    ventas_long = normalizar_ventas(dfv, anio=anio)
    pres_long = normalizar_presupuesto(dfp, anio=anio)

    # merge principal
    df = ventas_long.merge(pres_long, on=["ItemCode","anio","mes"], how="left")
    df["budget_kg"] = df["budget_kg"].fillna(0)

    df["var_kg"] = df["actual_kg"] - df["budget_kg"]
    df["cumpl_pct"] = (df["actual_kg"] / df["budget_kg"]).replace([float("inf")], 0).fillna(0) * 100

    # categorical orden meses
    if "mes" in df.columns:
        df["mes"] = pd.Categorical(df["mes"], categories=MESES_ORDEN, ordered=True)

    return df


# =========================================================
# KPI y TABLAS
# =========================================================
def meses_con_ventas_reales(df: pd.DataFrame) -> List[str]:
    if df.empty or "mes" not in df.columns:
        return []
    tmp = df.groupby("mes", as_index=False)["actual_kg"].sum()
    tmp = tmp[tmp["actual_kg"] > 0]
    meses = [str(m) for m in tmp["mes"].tolist()]
    # mantener orden
    return [m for m in MESES_ORDEN if m in meses]

def compute_kpis(df: pd.DataFrame, modo_kpi: str) -> dict:
    """
    modo_kpi:
      - "solo_reales": solo meses donde actual_kg > 0
      - "todos": todos los meses filtrados
    """
    if df.empty:
        return {"actual": 0.0, "budget": 0.0, "var": 0.0, "cumpl": 0.0}

    dfx = df.copy()
    if modo_kpi == "solo_reales":
        by = dfx.groupby("mes", as_index=False)["actual_kg"].sum()
        reales = set(by[by["actual_kg"] > 0]["mes"].tolist())
        dfx = dfx[dfx["mes"].isin(reales)]

    actual = float(dfx["actual_kg"].sum())
    budget = float(dfx["budget_kg"].sum())
    var_ = actual - budget
    cumpl = (actual / budget * 100) if budget > 0 else 0.0
    return {"actual": actual, "budget": budget, "var": var_, "cumpl": cumpl}

def by_mes_table(df: pd.DataFrame, modo_kpi: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["mes","actual_kg","budget_kg","var_kg","cumpl_pct"])

    dfx = df.copy()
    if modo_kpi == "solo_reales":
        reales = set(meses_con_ventas_reales(dfx))
        dfx = dfx[dfx["mes"].isin(reales)]

    t = dfx.groupby("mes", as_index=False)[["actual_kg","budget_kg","var_kg"]].sum()
    t["cumpl_pct"] = t.apply(lambda r: (r["actual_kg"] / r["budget_kg"] * 100) if r["budget_kg"] > 0 else 0.0, axis=1)
    t["mes"] = pd.Categorical(t["mes"], categories=MESES_ORDEN, ordered=True)
    t = t.sort_values("mes")
    return t

def forecast_next_months(by_mes: pd.DataFrame, n_ahead: int = 3) -> pd.DataFrame:
    """
    Forecast simple MBA: promedio móvil de los últimos 2 meses con datos reales.
    """
    if by_mes.empty:
        return pd.DataFrame(columns=["mes","forecast_actual_kg"])

    hist = by_mes.copy()
    hist = hist[hist["actual_kg"] > 0].sort_values("mes")
    if hist.empty:
        return pd.DataFrame(columns=["mes","forecast_actual_kg"])

    # valor forecast base
    last_vals = hist["actual_kg"].tail(2).tolist()
    base = sum(last_vals) / len(last_vals)

    # determinar siguientes meses
    meses_hist = [str(m) for m in hist["mes"].tolist()]
    last_mes = meses_hist[-1]
    start_idx = MESES_ORDEN.index(last_mes) if last_mes in MESES_ORDEN else 0

    future = []
    for i in range(1, n_ahead + 1):
        idx = start_idx + i
        if idx >= len(MESES_ORDEN):
            break
        future.append({"mes": MESES_ORDEN[idx], "forecast_actual_kg": base})

    return pd.DataFrame(future)


# =========================================================
# UI / ESTILO (tipo landing + secciones)
# =========================================================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; }
      .kpi-box { border: 1px solid #eee; border-radius: 14px; padding: 14px 16px; }
      .section-title { font-size: 1.25rem; font-weight: 700; margin: 0.25rem 0 0.75rem 0; }
      .muted { color: #6b7280; }
      .card {
        border: 1px solid #eee; border-radius: 16px; padding: 16px;
        background: #fff;
      }
      .card h3 { margin: 0 0 6px 0; }
      .card p { margin: 0; color: #6b7280; }
      .pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        border: 1px solid #eee; font-size: 0.85rem; color: #374151;
        background: #fafafa;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(APP_TITLE)

tabs = st.tabs([
    "Cargar archivos (mensual)",
    "Dashboard",
    "IA del Dashboard",
    "Forecast & Riesgo",
    "IA Técnica Preventa",
])

# =========================================================
# TAB 1: CARGA
# =========================================================
with tabs[0]:
    st.markdown('<div class="section-title">1) Cargar archivos (mensual)</div>', unsafe_allow_html=True)

    cA, cB = st.columns(2)
    with cA:
        ventas_file = st.file_uploader("Reporte de Ventas mensual (.xlsx)", type=["xlsx"], key="ventas_file")
    with cB:
        pres_file = st.file_uploader("Presupuesto (.xlsx)", type=["xlsx"], key="pres_file")

    anio = st.number_input("Año", min_value=2020, max_value=2100, value=2026, step=1)

    if st.button("Procesar", type="primary"):
        if not ventas_file or not pres_file:
            st.error("Sube ambos archivos (ventas y presupuesto).")
        else:
            dfv = pd.read_excel(ventas_file)
            dfp = pd.read_excel(pres_file)

            df_final = preparar_df_final(dfv, dfp, anio=int(anio))
            st.session_state["df_final"] = df_final

            meses_reales = meses_con_ventas_reales(df_final)
            st.success("✅ Datos procesados correctamente.")
            if meses_reales:
                st.info("Meses detectados con datos reales: " + ", ".join(meses_reales))
            else:
                st.warning("No detecté meses con ventas > 0. Revisa columnas *_KG del Excel de ventas.")

            with st.expander("Ver muestra (primeras filas)"):
                st.dataframe(df_final.head(25), use_container_width=True)

    st.caption("Tip: si tu Excel mensual solo tiene 1–2 meses con números, este sistema puede calcular KPIs solo sobre esos meses (no penaliza meses futuros).")


# =========================================================
# SIDEBAR: FILTROS (si hay df)
# =========================================================
df_global = st.session_state.get("df_final", None)
if isinstance(df_global, pd.DataFrame) and not df_global.empty:
    st.sidebar.header("Filtros")

    anios = sorted([int(a) for a in df_global["anio"].dropna().unique().tolist()]) if "anio" in df_global.columns else []
    if anios:
        anio_sel = st.sidebar.selectbox("Año", anios, index=len(anios)-1)
    else:
        anio_sel = None

    df_f = df_global.copy()
    if anio_sel is not None and "anio" in df_f.columns:
        df_f = df_f[df_f["anio"] == anio_sel]

    meses_default = meses_con_ventas_reales(df_f) or MESES_ORDEN[:2]
    mes_sel = st.sidebar.multiselect("Mes (análisis)", MESES_ORDEN, default=meses_default)
    if mes_sel and "mes" in df_f.columns:
        df_f = df_f[df_f["mes"].isin(mes_sel)]

    if "SlpName" in df_f.columns:
        vendedores = sorted([v for v in df_f["SlpName"].dropna().unique().tolist()])
        vend_default = vendedores[:]
        vend_sel = st.sidebar.multiselect("Vendedor", vendedores, default=vend_default)
        if vend_sel:
            df_f = df_f[df_f["SlpName"].isin(vend_sel)]
    else:
        vend_sel = None

    st.sidebar.divider()
    st.sidebar.markdown("**KPIs basados en:**")
    modo_kpi = st.sidebar.radio(
        label="",
        options=["Solo meses con ventas reales", "Todos los meses seleccionados"],
        index=0,
        help="Para carga mensual: 'Solo meses con ventas reales' evita penalizar meses futuros con 0."
    )
    modo_kpi_key = "solo_reales" if modo_kpi.startswith("Solo") else "todos"

else:
    df_f = None
    modo_kpi_key = "solo_reales"


# =========================================================
# TAB 2: DASHBOARD
# =========================================================
with tabs[1]:
    st.markdown('<div class="section-title">2) Dashboard</div>', unsafe_allow_html=True)
    if df_f is None or df_f.empty:
        st.warning("Carga y procesa los archivos en la pestaña **Cargar archivos (mensual)**.")
    else:
        k = compute_kpis(df_f, modo_kpi=modo_kpi_key)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual (KG)", f"{k['actual']:,.0f}")
        c2.metric("Budget (KG)", f"{k['budget']:,.0f}")
        c3.metric("Varianza (KG)", f"{k['var']:,.0f}")
        c4.metric("% Cumplimiento", f"{k['cumpl']:.1f}%")

        st.markdown("### Tendencia mensual (Actual vs Budget)")
        t_mes = by_mes_table(df_f, modo_kpi=modo_kpi_key)

        if t_mes.empty:
            st.info("No hay datos suficientes para graficar.")
        else:
            fig = px.line(
                t_mes,
                x="mes",
                y=["actual_kg", "budget_kg"],
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tabla por mes")
        st.dataframe(t_mes, use_container_width=True)


# =========================================================
# TAB 3: IA DEL DASHBOARD
# =========================================================
with tabs[2]:
    st.markdown('<div class="section-title">3) IA del Dashboard (análisis ejecutivo)</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Genera un informe (sin inventar cifras) con base en datos agregados por mes.</div>', unsafe_allow_html=True)

    if df_f is None or df_f.empty:
        st.warning("Carga datos primero.")
    else:
        # modelo
        models = gemini_list_models_cached()
        default_model = "gemini-1.5-flash" if "gemini-1.5-flash" in models else (models[0] if models else "gemini-1.5-flash")
        model_name = st.text_input("Modelo Gemini", value=default_model, help="Si tienes dudas, presiona 'Ver modelos disponibles'.")
        with st.expander("Ver modelos disponibles (según tu API Key)"):
            st.write(models)

        t_mes = by_mes_table(df_f, modo_kpi=modo_kpi_key)
        resumen = t_mes.to_string(index=False)

        if st.button("Generar análisis con IA", type="primary"):
            prompt = f"""
Eres analista financiero-industrial B2B (empaque flexible).
Usa SOLO los datos siguientes (tabla mensual agregada). NO inventes cifras.

DATOS:
{resumen}

Entrega en español:
1) Resumen ejecutivo (3-6 bullets)
2) Conclusiones clave (máx 6)
3) Recomendaciones comerciales y de operación (máx 8)
4) Riesgos y alertas (máx 6)
5) Próximos pasos (3 bullets)
Regla: si detectas meses futuros con 0 porque el archivo es mensual, NO los trates como caída real;
en su lugar, explícitalo como "meses sin actualización aún".
"""
            try:
                ans = gemini_generate(prompt, model_name=model_name, retries=3)
                st.session_state["ia_dashboard_last"] = ans
                st.markdown(ans)
            except Exception as e:
                st.error("No se pudo generar el análisis. Revisa tu GEMINI_API_KEY y el modelo.")
                st.exception(e)

        # export ejecutivo simple (sin libs extra)
        last = st.session_state.get("ia_dashboard_last", "")
        if last:
            st.download_button(
                "⬇️ Descargar informe (TXT)",
                data=last.encode("utf-8"),
                file_name="informe_ejecutivo.txt",
                mime="text/plain"
            )


# =========================================================
# TAB 4: FORECAST & RIESGO
# =========================================================
with tabs[3]:
    st.markdown('<div class="section-title">4) Forecast & Riesgo</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Forecast simple y matriz de riesgo para consultoría ejecutiva (MBA).</div>', unsafe_allow_html=True)

    if df_f is None or df_f.empty:
        st.warning("Carga datos primero.")
    else:
        t_mes = by_mes_table(df_f, modo_kpi=modo_kpi_key)
        if t_mes.empty:
            st.info("No hay datos suficientes.")
        else:
            fc = forecast_next_months(t_mes, n_ahead=3)
            st.markdown("### Forecast (siguiente(s) mes(es))")
            if fc.empty:
                st.info("No puedo proyectar sin meses con ventas reales.")
            else:
                st.dataframe(fc, use_container_width=True)

                # gráfico combinado
                plot_hist = t_mes[["mes","actual_kg"]].copy()
                plot_hist["tipo"] = "Histórico"
                plot_fc = fc.rename(columns={"forecast_actual_kg": "actual_kg"}).copy()
                plot_fc["tipo"] = "Forecast"

                plot_all = pd.concat([plot_hist, plot_fc], ignore_index=True)
                plot_all["mes"] = pd.Categorical(plot_all["mes"], categories=MESES_ORDEN, ordered=True)
                plot_all = plot_all.sort_values("mes")

                st.plotly_chart(
                    px.line(plot_all, x="mes", y="actual_kg", color="tipo", markers=True),
                    use_container_width=True
                )

            st.markdown("### Riesgo (reglas simples)")
            # reglas: bajo/medio/alto
            k = compute_kpis(df_f, modo_kpi=modo_kpi_key)
            riesgos = []

            if k["cumpl"] < 80:
                riesgos.append(("Alto", "Cumplimiento bajo", f"Cumplimiento {k['cumpl']:.1f}% vs objetivo 80%"))
            elif k["cumpl"] < 95:
                riesgos.append(("Medio", "Cumplimiento moderado", f"Cumplimiento {k['cumpl']:.1f}%"))
            else:
                riesgos.append(("Bajo", "Cumplimiento saludable", f"Cumplimiento {k['cumpl']:.1f}%"))

            # volatilidad simple: si hay al menos 2 meses reales
            hist = t_mes[t_mes["actual_kg"] > 0].copy()
            if len(hist) >= 2:
                vol = float(hist["actual_kg"].pct_change().abs().mean())
                if vol > 0.35:
                    riesgos.append(("Medio", "Alta volatilidad mensual", f"Volatilidad promedio {vol:.2f}"))
            else:
                riesgos.append(("Bajo", "Serie corta", "Pocos meses con datos reales (carga mensual)."))

            risk_df = pd.DataFrame(riesgos, columns=["Nivel", "Riesgo", "Detalle"])
            st.dataframe(risk_df, use_container_width=True)

            st.markdown("### Exportación ejecutiva")
            export_xlsx = t_mes.copy()
            st.download_button(
                "⬇️ Descargar tabla mensual (Excel)",
                data=export_xlsx.to_excel(index=False, engine="openpyxl"),
                file_name="tabla_mensual.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# =========================================================
# TAB 5: IA TÉCNICA PREVENTA
# =========================================================
with tabs[4]:
    st.markdown('<div class="section-title">5) IA Técnica Preventa</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Responde SOLO con evidencia del manual. Si no hay evidencia, pide datos faltantes (no inventa).</div>', unsafe_allow_html=True)

    pdf_path = find_manual_pdf()
    if not pdf_path:
        st.warning("No encuentro un PDF del manual. Sube tu manual al repo en `manual_tecnico/` (recomendado) o en la raíz.")
        st.stop()

    st.caption(f"Manual detectado: `{pdf_path}`")

    manual_text = load_manual_text(pdf_path)
    if not manual_text:
        st.warning("Pude abrir el PDF, pero no pude extraer texto. Si es escaneado, expórtalo como PDF con texto.")
        st.stop()

    chunks = chunk_text(manual_text, chunk_size=1400, overlap=200)

    st.markdown("**Ejemplos de consulta:**")
    st.markdown("- Café 500g, VFFS, vida útil 12 meses\n- Detergente en polvo 1kg, HFFS, alta barrera\n- Snacks 30g, alta velocidad, metalizado")

    # modelo
    models = gemini_list_models_cached()
    default_model = "gemini-1.5-flash" if "gemini-1.5-flash" in models else (models[0] if models else "gemini-1.5-flash")
    model_name = st.text_input("Modelo Gemini (Preventa)", value=default_model, key="model_preventa")
    with st.expander("Ver modelos disponibles (según tu API Key)", expanded=False):
        st.write(models)

    if "chat_preventa" not in st.session_state:
        st.session_state["chat_preventa"] = []

    # historial
    for role, content in st.session_state["chat_preventa"]:
        with st.chat_message(role):
            st.markdown(content)

    pregunta = st.chat_input("Ej: Café 500g, VFFS, 12 meses.")
    if pregunta:
        st.session_state["chat_preventa"].append(("user", pregunta))
        with st.chat_message("user"):
            st.markdown(pregunta)

        ranked = rank_chunks_by_keywords(chunks, pregunta, top_k=4)
        context = "\n\n---\n\n".join([f"[Chunk {i}] {ch}" for i, _, ch in ranked]) if ranked else ""

        prompt = f"""
Eres ingeniero/a de preventa especializado/a en empaque plástico flexible (bolsa y bobina) B2B.
Regla crítica: RESPONDE SOLO usando el CONTEXTO del manual. Si algo no está en el contexto, di:
"Según el manual no hay evidencia suficiente" y pide los datos faltantes.

FORMATO DE RESPUESTA (obligatorio):
A) Información faltante (si aplica) en bullets
B) Opción A segura (máxima protección): estructura sugerida, micraje, barrera, nota de proceso
C) Opción B optimizada costo: estructura sugerida, micraje, barrera, nota de proceso
D) Riesgos (técnicos/comerciales) en bullets
E) Evidencia usada: cita los chunks (por ejemplo: "Chunk 2") y una frase corta

CONTEXTO (extractos del manual):
{context if context else "[Sin chunks relevantes encontrados. Debes pedir información faltante y NO inventar]."}

PREGUNTA:
{pregunta}
"""

        with st.chat_message("assistant"):
            try:
                ans = gemini_generate(prompt, model_name=model_name, retries=3)
                st.markdown(ans)
                st.session_state["chat_preventa"].append(("assistant", ans))
            except Exception as e:
                st.error("No se pudo generar respuesta con Gemini. Revisa tu GEMINI_API_KEY o el modelo (y reintenta).")
                st.exception(e)

    with st.expander("Debug (opcional)"):
        st.write("Contenido raíz:", os.listdir("."))
        if os.path.isdir(MANUAL_DIR):
            st.write("Contenido manual_tecnico:", os.listdir(MANUAL_DIR))
