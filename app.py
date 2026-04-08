"""
app.py — Interfaz Streamlit para predicción de clientes
Modelos: M1 Churn (¿comprará en 90 días?) | M2 Categoría (¿qué comprará?)
Artefacto: modelo_360.pkl
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Clientes",
    page_icon="🎯",
    layout="wide",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .churn-alto {
        background-color: #ffe0e0;
        border-left: 5px solid #e74c3c;
        padding: 18px 20px;
        border-radius: 8px;
    }
    .churn-bajo {
        background-color: #e0ffe0;
        border-left: 5px solid #27ae60;
        padding: 18px 20px;
        border-radius: 8px;
    }
    .card-cat {
        background-color: #e8f4fd;
        border-left: 5px solid #2980b9;
        padding: 18px 20px;
        border-radius: 8px;
    }
    .resultado-titulo { font-size: 15px; color: #555; margin-bottom: 4px; }
    .resultado-valor  { font-size: 28px; font-weight: bold; margin: 6px 0; }
    .resultado-conf   { font-size: 14px; color: #666; }
</style>
""", unsafe_allow_html=True)

# ── Carga del modelo ───────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo(path: str = "modelo_360.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    paquete      = cargar_modelo()
    modelo_churn = paquete["modelo_churn"]
    modelo_cat   = paquete["modelo_cat"]
    le_m2        = paquete["le_m2"]
    variables    = paquete["variables"]
except FileNotFoundError:
    st.error("⚠️ No se encontró **modelo_360.pkl**. Colócalo en el mismo directorio que esta app.")
    st.stop()

# ── Título ─────────────────────────────────────────────────────────────────────
st.title("🎯 Predicción de Comportamiento de Clientes")
st.markdown(
    "Ingresa el **perfil histórico de comportamiento** de un cliente para predecir "
    "si comprará en los próximos 90 días y qué categoría es más probable que compre."
)
st.divider()

# ── Listas de opciones ─────────────────────────────────────────────────────────
CATEGORIAS_DISPONIBLES = [
    "PINTUCO", "REVESTIMIENTO CORONA", "PLOMERIA GRIVAL",
    "ACCESORIOS GERFOR", "ICO", "OTROS", "DESCONOCIDO",
]
CIUDADES_DISPONIBLES = [
    "MEDELLÍN", "BOGOTÁ", "CALI", "BARRANQUILLA",
    "BUCARAMANGA", "PEREIRA", "MANIZALES", "OTRA",
]

# ── Panel de entrada ───────────────────────────────────────────────────────────
st.subheader("📋 Perfil del cliente")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📦 Volumen de compras**")
    total_transacciones = st.number_input(
        "Total de transacciones (documentos únicos)",
        min_value=1, max_value=5000, value=50, step=1,
        help="Número de pedidos/documentos distintos en el historial",
    )
    total_items = st.number_input(
        "Total de ítems comprados",
        min_value=1, max_value=50000, value=200, step=10,
        help="Número total de líneas de producto",
    )
    total_cantidad = st.number_input(
        "Cantidad total de unidades",
        min_value=1, max_value=500_000, value=1000, step=50,
        help="Suma de unidades físicas compradas",
    )

with col2:
    st.markdown("**💰 Valor monetario (COP)**")
    total_valor = st.number_input(
        "Valor bruto total ($)",
        min_value=0, max_value=500_000_000, value=20_000_000, step=500_000,
        help="Suma del valor bruto de todas las transacciones",
    )
    avg_valor = st.number_input(
        "Valor promedio por ítem ($)",
        min_value=0, max_value=50_000_000, value=400_000, step=10_000,
        help="Promedio de Valor_bruto por línea de producto",
    )
    max_valor = st.number_input(
        "Valor máximo en una transacción ($)",
        min_value=0, max_value=100_000_000, value=2_000_000, step=100_000,
        help="Mayor valor bruto registrado en el historial",
    )
    std_valor = st.number_input(
        "Desviación estándar del valor ($)",
        min_value=0, max_value=50_000_000, value=300_000, step=10_000,
        help="Variabilidad del gasto (0 = muy constante)",
    )

with col3:
    st.markdown("**📅 Comportamiento temporal**")
    recencia = st.number_input(
        "Recencia (días desde última compra)",
        min_value=0, max_value=730, value=30, step=1,
        help="Días desde la última compra hasta el corte (sep-2025). Valores altos = mayor riesgo de churn.",
    )
    dias_activo = st.number_input(
        "Días activo como cliente",
        min_value=1, max_value=730, value=365, step=1,
        help="Días entre primera y última compra",
    )
    frecuencia = st.number_input(
        "Frecuencia (compras/mes)",
        min_value=0.0, max_value=100.0, value=2.5, step=0.1, format="%.2f",
        help="Total transacciones ÷ (días_activo / 30)",
    )

st.divider()
col4, col5 = st.columns(2)

with col4:
    st.markdown("**🌐 Diversidad de comportamiento**")
    categorias_distintas = st.number_input(
        "Categorías distintas compradas",
        min_value=1, max_value=20, value=3, step=1,
    )
    subcategorias_distintas = st.number_input(
        "Subcategorías distintas compradas",
        min_value=1, max_value=50, value=8, step=1,
    )
    ciudades_distintas = st.number_input(
        "Ciudades de facturación distintas",
        min_value=1, max_value=30, value=2, step=1,
    )

with col5:
    st.markdown("**🏷️ Preferencias del cliente**")
    top_categoria = st.selectbox(
        "Categoría favorita (historial)",
        options=CATEGORIAS_DISPONIBLES,
        help="Categoría con mayor frecuencia de compra en el historial",
    )
    top_ciudad = st.selectbox(
        "Ciudad principal de facturación",
        options=CIUDADES_DISPONIBLES,
        help="Ciudad donde factura con mayor frecuencia",
    )

# ── Botón de predicción ────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔮 Generar predicciones", type="primary", use_container_width=True)

if predict_btn:

    # Encoding estable de variables categóricas (mismo orden que el entrenamiento)
    cat_map = {c: i for i, c in enumerate(sorted(CATEGORIAS_DISPONIBLES))}
    ciu_map = {c: i for i, c in enumerate(sorted(CIUDADES_DISPONIBLES))}

    top_categoria_enc = cat_map.get(top_categoria, 0)
    top_ciudad_enc    = ciu_map.get(top_ciudad, 0)

    # Vector de features — mismos nombres que X en el entrenamiento
    raw = {
        "total_transacciones":     total_transacciones,
        "total_items":             total_items,
        "total_valor":             total_valor,
        "avg_valor":               avg_valor,
        "std_valor":               std_valor,
        "max_valor":               max_valor,
        "total_cantidad":          total_cantidad,
        "ciudades_distintas":      ciudades_distintas,
        "categorias_distintas":    categorias_distintas,
        "subcategorias_distintas": subcategorias_distintas,
        "dias_activo":             dias_activo,
        "recencia":                recencia,
        "frecuencia":              frecuencia,
        "top_categoria_enc":       top_categoria_enc,
        "top_ciudad_enc":          top_ciudad_enc,
    }

    # Reindexar con las columnas exactas del modelo
    X_input = pd.DataFrame([raw]).reindex(columns=variables, fill_value=0)

    # ── M1 — Churn ────────────────────────────────────────────────────────────
    churn_pred  = modelo_churn.predict(X_input)[0]
    churn_proba = modelo_churn.predict_proba(X_input)[0]
    churn_pct   = churn_proba[churn_pred] * 100

    # ── M2 — Categoría ────────────────────────────────────────────────────────
    cat_pred_enc = modelo_cat.predict(X_input)[0]
    cat_pred     = le_m2.inverse_transform([cat_pred_enc])[0]
    cat_proba    = modelo_cat.predict_proba(X_input)[0]
    cat_classes  = le_m2.inverse_transform(modelo_cat.classes_)
    cat_conf_pct = cat_proba.max() * 100

    # ── Resultados ────────────────────────────────────────────────────────────
    st.subheader("📊 Resultados")
    res1, res2 = st.columns(2)

    with res1:
        churn_label = "🔴 CHURN" if churn_pred == 1 else "🟢 NO CHURN"
        churn_class = "churn-alto" if churn_pred == 1 else "churn-bajo"
        st.markdown(f"""
        <div class="{churn_class}">
            <p class="resultado-titulo">M1 — ¿Comprará en los próximos 90 días?</p>
            <p class="resultado-valor">{churn_label}</p>
            <p class="resultado-conf">Confianza: <strong>{churn_pct:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown(f"""
        <div class="card-cat">
            <p class="resultado-titulo">M2 — ¿Qué categoría comprará?</p>
            <p class="resultado-valor">🏷️ {cat_pred}</p>
            <p class="resultado-conf">Confianza: <strong>{cat_conf_pct:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Distribución de probabilidades ────────────────────────────────────────
    st.subheader("📈 Distribución de probabilidades")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**M1 — Churn**")
        churn_df = pd.DataFrame({
            "Clase":        ["No Churn", "Churn"],
            "Probabilidad": [churn_proba[0], churn_proba[1]],
        }).set_index("Clase")
        st.bar_chart(churn_df)

    with ch2:
        st.markdown("**M2 — Categoría**")
        cat_df = pd.DataFrame({
            "Categoría":    cat_classes,
            "Probabilidad": cat_proba,
        }).set_index("Categoría").sort_values("Probabilidad", ascending=False)
        st.bar_chart(cat_df)

    # ── Resumen ejecutivo ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🗒️ Resumen ejecutivo")

    accion_churn = (
        "⚠️ **Acción recomendada:** Alto riesgo de abandono. "
        "Considera una campaña de retención proactiva (descuentos, contacto comercial)."
        if churn_pred == 1
        else "✅ **Cliente activo.** Se espera que compre en los próximos 90 días. "
             "Enfoca esfuerzos en aumentar el valor del carrito."
    )

    st.info(f"""
**M1 — Churn:** {churn_label} ({churn_pct:.1f}% de confianza)  
{accion_churn}

**M2 — Categoría predicha:** {cat_pred} ({cat_conf_pct:.1f}% de confianza)  
📦 Foco de recomendación de producto: línea **{cat_pred}**.
""")

    # ── Datos de entrada usados ────────────────────────────────────────────────
    with st.expander("🔍 Ver datos de entrada enviados al modelo"):
        st.dataframe(X_input.T.rename(columns={0: "Valor"}), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Modelos: M1 Regresión Logística (Churn · Acc=0.97) · "
    "M2 Gradient Boosting (Categoría · Acc=0.63) | Corte historial: sep-2025"
)
