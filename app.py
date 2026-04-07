"""
app.py — Interfaz Streamlit para el sistema de predicción 360° de clientes
Modelos: M1 Churn | M2 Categoría | M3 Segmento de Valor
Artefacto: modelo_360.pkl
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción 360° de Clientes",
    page_icon="🎯",
    layout="wide",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .churn-alto   { background-color: #ffe0e0; border-left: 5px solid #e74c3c; padding: 10px; border-radius: 5px; }
    .churn-bajo   { background-color: #e0ffe0; border-left: 5px solid #27ae60; padding: 10px; border-radius: 5px; }
    .seg-alto     { color: #27ae60; font-weight: bold; }
    .seg-medio    { color: #f39c12; font-weight: bold; }
    .seg-bajo     { color: #e74c3c; font-weight: bold; }
    .seg-nocompra { color: #7f8c8d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Carga del modelo ───────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo(path: str = "modelo_360.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    paquete = cargar_modelo()
    modelo_churn = paquete["modelo_churn"]
    modelo_cat   = paquete["modelo_cat"]
    modelo_valor = paquete["modelo_valor"]
    le_m2        = paquete["le_m2"]
    le_m3        = paquete["le_m3"]
    variables    = paquete["variables"]
    modelo_ok    = True
except FileNotFoundError:
    modelo_ok = False
    st.error("⚠️ No se encontró el archivo **modelo_360.pkl**. Colócalo en el mismo directorio que esta app.")
    st.stop()

# ── Título ─────────────────────────────────────────────────────────────────────
st.title("🎯 Sistema de Predicción 360° de Clientes")
st.markdown(
    "Ingresa el **perfil histórico de comportamiento** de un cliente para obtener "
    "predicciones simultáneas de **Churn**, **Categoría de Compra** y **Segmento de Valor**."
)
st.divider()

# ── Sidebar — Categorías y ciudades disponibles ────────────────────────────────
CATEGORIAS_DISPONIBLES = [
    "PINTUCO", "REVESTIMIENTO CORONA", "PLOMERIA GRIVAL",
    "ACCESORIOS GERFOR", "ICO", "OTROS", "DESCONOCIDO"
]
CIUDADES_DISPONIBLES = [
    "MEDELLÍN", "BOGOTÁ", "CALI", "BARRANQUILLA",
    "BUCARAMANGA", "PEREIRA", "MANIZALES", "OTRA"
]

# ── Panel de entrada ───────────────────────────────────────────────────────────
st.subheader("📋 Perfil del cliente")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📦 Volumen de compras**")
    total_transacciones = st.number_input(
        "Total de transacciones (documentos únicos)",
        min_value=1, max_value=5000, value=50, step=1,
        help="Número de documentos/pedidos distintos en el historial"
    )
    total_items = st.number_input(
        "Total de ítems comprados",
        min_value=1, max_value=50000, value=200, step=10,
        help="Número total de líneas de producto"
    )
    total_cantidad = st.number_input(
        "Cantidad total de unidades",
        min_value=1, max_value=500000, value=1000, step=50,
        help="Suma de unidades físicas compradas"
    )

with col2:
    st.markdown("**💰 Valor monetario (COP)**")
    total_valor = st.number_input(
        "Valor bruto total ($)",
        min_value=0, max_value=500_000_000, value=20_000_000, step=500_000,
        help="Suma del valor bruto de todas las transacciones"
    )
    avg_valor = st.number_input(
        "Valor promedio por ítem ($)",
        min_value=0, max_value=50_000_000, value=400_000, step=10_000,
        help="Promedio de Valor_bruto por línea de producto"
    )
    max_valor = st.number_input(
        "Valor máximo en una transacción ($)",
        min_value=0, max_value=100_000_000, value=2_000_000, step=100_000,
        help="Mayor valor bruto registrado en el historial"
    )
    std_valor = st.number_input(
        "Desviación estándar del valor ($)",
        min_value=0, max_value=50_000_000, value=300_000, step=10_000,
        help="Variabilidad del gasto (0 = muy constante)"
    )

with col3:
    st.markdown("**📅 Comportamiento temporal**")
    recencia = st.number_input(
        "Recencia (días desde última compra)",
        min_value=0, max_value=730, value=30, step=1,
        help="Días transcurridos desde la última compra hasta el corte (sep-2025). Valores altos = mayor riesgo de churn."
    )
    dias_activo = st.number_input(
        "Días activo como cliente",
        min_value=1, max_value=730, value=365, step=1,
        help="Días entre primera y última compra"
    )
    frecuencia = st.number_input(
        "Frecuencia (compras/mes)",
        min_value=0.0, max_value=100.0, value=2.5, step=0.1, format="%.2f",
        help="Total de transacciones dividido por (días_activo / 30)"
    )

st.divider()
col4, col5 = st.columns(2)

with col4:
    st.markdown("**🌐 Diversidad de comportamiento**")
    categorias_distintas = st.number_input(
        "Categorías distintas compradas",
        min_value=1, max_value=20, value=3, step=1
    )
    subcategorias_distintas = st.number_input(
        "Subcategorías distintas compradas",
        min_value=1, max_value=50, value=8, step=1
    )
    ciudades_distintas = st.number_input(
        "Ciudades de facturación distintas",
        min_value=1, max_value=30, value=2, step=1
    )

with col5:
    st.markdown("**🏷️ Preferencias del cliente**")
    top_categoria = st.selectbox(
        "Categoría favorita (historial)",
        options=CATEGORIAS_DISPONIBLES,
        help="Categoría con mayor número de compras en el historial"
    )
    top_ciudad = st.selectbox(
        "Ciudad principal de facturación",
        options=CIUDADES_DISPONIBLES,
        help="Ciudad donde factura con mayor frecuencia"
    )

# ── Encoding de categóricas igual que en el entrenamiento ─────────────────────
def encode_label(encoder, value):
    """Encode con fallback a 0 si la clase es desconocida."""
    classes = list(encoder.classes_)
    if value in classes:
        return encoder.transform([value])[0]
    return 0  # clase desconocida → codificada como 0

# ── Botón de predicción ────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔮 Generar predicciones", type="primary", use_container_width=True)

if predict_btn:
    # Obtener los encoders de top_categoria y top_ciudad del paquete
    # (los modelos usan le_cat y le_ciu del feature engineering; usamos los de m2/m3 como proxy)
    # El modelo usa LabelEncoder sobre los campos del feats; reconstruimos manualmente.
    # Como los encoders guardados son le_m2 (categoría futura) y le_m3 (segmento),
    # para top_categoria_enc y top_ciudad_enc hacemos un hash numérico estable.
    cat_map  = {c: i for i, c in enumerate(sorted(CATEGORIAS_DISPONIBLES))}
    ciu_map  = {c: i for i, c in enumerate(sorted(CIUDADES_DISPONIBLES))}

    top_categoria_enc = cat_map.get(top_categoria, 0)
    top_ciudad_enc    = ciu_map.get(top_ciudad, 0)

    # Construir el vector de features con los mismos nombres que X en el entrenamiento
    raw = {
        "total_transacciones":      total_transacciones,
        "total_items":              total_items,
        "total_valor":              total_valor,
        "avg_valor":                avg_valor,
        "std_valor":                std_valor,
        "max_valor":                max_valor,
        "total_cantidad":           total_cantidad,
        "ciudades_distintas":       ciudades_distintas,
        "categorias_distintas":     categorias_distintas,
        "subcategorias_distintas":  subcategorias_distintas,
        "dias_activo":              dias_activo,
        "recencia":                 recencia,
        "frecuencia":               frecuencia,
        "top_categoria_enc":        top_categoria_enc,
        "top_ciudad_enc":           top_ciudad_enc,
    }

    # Reindexar con las columnas exactas del modelo, rellenar faltantes con 0
    X_input = pd.DataFrame([raw]).reindex(columns=variables, fill_value=0)

    # ── Predicciones ──────────────────────────────────────────────────────────
    churn_pred      = modelo_churn.predict(X_input)[0]
    churn_proba     = modelo_churn.predict_proba(X_input)[0]

    cat_pred_enc    = modelo_cat.predict(X_input)[0]
    cat_pred        = le_m2.inverse_transform([cat_pred_enc])[0]
    cat_proba       = modelo_cat.predict_proba(X_input)[0]
    cat_classes     = le_m2.inverse_transform(modelo_cat.classes_)

    valor_pred_enc  = modelo_valor.predict(X_input)[0]
    valor_pred      = le_m3.inverse_transform([valor_pred_enc])[0]
    valor_proba     = modelo_valor.predict_proba(X_input)[0]
    valor_classes   = le_m3.inverse_transform(modelo_valor.classes_)

    # ── Visualización de resultados ────────────────────────────────────────────
    st.subheader("📊 Resultados de la predicción")
    r1, r2, r3 = st.columns(3)

    # M1 — Churn
    with r1:
        churn_label   = "🔴 CHURN" if churn_pred == 1 else "🟢 NO CHURN"
        churn_pct     = churn_proba[churn_pred] * 100
        churn_class   = "churn-alto" if churn_pred == 1 else "churn-bajo"
        st.markdown(f"""
        <div class="{churn_class}">
            <h4>M1 — ¿El cliente comprará en los próximos 90 días?</h4>
            <h2>{churn_label}</h2>
            <p>Confianza: <strong>{churn_pct:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # M2 — Categoría
    with r2:
        max_cat_idx  = np.argmax(cat_proba)
        cat_conf_pct = cat_proba[max_cat_idx] * 100
        st.markdown(f"""
        <div style="background-color:#e8f4fd;border-left:5px solid #2980b9;padding:10px;border-radius:5px;">
            <h4>M2 — ¿Qué categoría comprará?</h4>
            <h2>🏷️ {cat_pred}</h2>
            <p>Confianza: <strong>{cat_conf_pct:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # M3 — Segmento de valor
    with r3:
        seg_colors = {"Alto": "#27ae60", "Medio": "#f39c12", "Bajo": "#e74c3c", "No_Compra": "#7f8c8d"}
        seg_emojis = {"Alto": "💎", "Medio": "⭐", "Bajo": "📉", "No_Compra": "❌"}
        seg_color  = seg_colors.get(valor_pred, "#7f8c8d")
        seg_emoji  = seg_emojis.get(valor_pred, "❓")
        max_seg_idx  = np.argmax(valor_proba)
        seg_conf_pct = valor_proba[max_seg_idx] * 100
        st.markdown(f"""
        <div style="background-color:#fef9e7;border-left:5px solid {seg_color};padding:10px;border-radius:5px;">
            <h4>M3 — Segmento de valor futuro</h4>
            <h2 style="color:{seg_color};">{seg_emoji} {valor_pred}</h2>
            <p>Confianza: <strong>{seg_conf_pct:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Distribución de probabilidades ────────────────────────────────────────
    st.subheader("📈 Distribución de probabilidades por modelo")
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        st.markdown("**M1 — Churn**")
        churn_df = pd.DataFrame({
            "Clase":       ["No Churn (0)", "Churn (1)"],
            "Probabilidad": [churn_proba[0], churn_proba[1]]
        }).set_index("Clase")
        st.bar_chart(churn_df)

    with pc2:
        st.markdown("**M2 — Categoría**")
        cat_df = pd.DataFrame({
            "Categoría":    cat_classes,
            "Probabilidad": cat_proba
        }).set_index("Categoría").sort_values("Probabilidad", ascending=False)
        st.bar_chart(cat_df)

    with pc3:
        st.markdown("**M3 — Segmento de valor**")
        seg_df = pd.DataFrame({
            "Segmento":     valor_classes,
            "Probabilidad": valor_proba
        }).set_index("Segmento").sort_values("Probabilidad", ascending=False)
        st.bar_chart(seg_df)

    # ── Resumen ejecutivo ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🗒️ Resumen ejecutivo")

    accion_churn = (
        "⚠️ **Acción recomendada:** Este cliente presenta alto riesgo de abandono. "
        "Considera una campaña de retención proactiva (descuentos, contacto comercial)."
        if churn_pred == 1
        else "✅ **Cliente activo.** Se espera que compre en los próximos 90 días. "
             "Enfoca esfuerzos en aumentar el valor del carrito."
    )

    seg_desc = {
        "Alto":      "Cliente VIP. Prioriza atención personalizada y programas de fidelización.",
        "Medio":     "Cliente con potencial de crecimiento. Campañas de up-selling.",
        "Bajo":      "Cliente de bajo valor futuro. Evalúa la rentabilidad antes de invertir en retención.",
        "No_Compra": "No se esperan compras. Considera campañas de reactivación o reasignación de recursos.",
    }

    st.info(f"""
**M1 — Churn:** {churn_label} ({churn_pct:.1f}% de confianza)  
{accion_churn}

**M2 — Categoría predicha:** {cat_pred} ({cat_conf_pct:.1f}% de confianza)  
📦 Foco de recomendación de producto: línea **{cat_pred}**.

**M3 — Segmento de valor:** {valor_pred} ({seg_conf_pct:.1f}% de confianza)  
💡 {seg_desc.get(valor_pred, "")}
""")

    # ── Datos de entrada usados ────────────────────────────────────────────────
    with st.expander("🔍 Ver datos de entrada enviados al modelo"):
        st.dataframe(X_input.T.rename(columns={0: "Valor"}), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Sistema 360° — Modelos: M1 Regresión Logística (Churn) · M2 Gradient Boosting (Categoría) · M3 Regresión Logística (Valor) | Corte historial: sep-2025")