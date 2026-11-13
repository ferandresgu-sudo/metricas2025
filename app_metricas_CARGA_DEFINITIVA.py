import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

st.set_page_config(page_title="Forecast Nov-Dic 2026", layout="wide")
st.markdown("## Forecast de ventas | Noviembre y diciembre 2026")
st.caption("Histórico mensual comparable y pronóstico conectado, con filtros de categoría, combo y producto.")

MESES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11,
    "DICIEMBRE": 12,
}

CAT_FEATURES = ["CODIGO", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]
LAGS = [1, 2, 3, 6, 12, 24]
NUM_FEATURES_BASE = (
    [f"lag{x}" for x in LAGS]
    + [
        "mean3", "std3", "mean6", "std6", "mean12", "std12",
        "sin_mes", "cos_mes", "MES_NUM", "ANO", "tendencia_3_12",
    ]
)
MACRO_FEATURES = ["ICC_BC", "INFLACION_LACTEOS_BC", "ITAEE_BC"]


def get_num_features(usar_macro=False):
    return NUM_FEATURES_BASE + (MACRO_FEATURES if usar_macro else [])


def get_features(usar_macro=False):
    return get_num_features(usar_macro) + CAT_FEATURES


# Aliases por compatibilidad (backtest y forecast sin macro).
NUM_FEATURES = NUM_FEATURES_BASE
FEATURES = NUM_FEATURES + CAT_FEATURES

# Configuración seleccionada en el backtest Nov-Dic 2025.
PESO_VENTA_DIRECTA = 0.85
PESO_ESTRUCTURAL = 0.15
PESO_FORECAST_ESTACIONAL = 0.85
UMBRAL_ESTACIONAL = 1.20

CALIBRACION_CATEGORIAS = {
    "LACTEOS CULINARIOS": 1.08, "CHOCOLATE MESA": 1.08,
    "NESTLE PROFESSIONAL": 1.08, "RTD": 1.05,
    "CHOCOLATE IMPULSO": 0.92, "CARNATION POLVO": 0.90,
    "CEREALES": 0.97, "AGUAS": 0.97,
}
FACTOR_CALIBRACION_MIN = 0.90
FACTOR_CALIBRACION_MAX = 1.10
APLICAR_CALIBRACION_A_COMBOS = False

CALIBRACION_COMBOS_CATEGORIA = {
    "LACTEOS CULINARIOS": 0.88, "CEREALES": 1.35, "CREMADORES": 1.02,
    "CAFE SOLUBLE": 1.40, "OTHER DAIRY": 1.01, "CULINARIOS": 1.40,
}
CALIBRACION_COMBOS_SKU = {
    "13802": 1.30, "13467": 1.30, "13134": 1.20,
    "13229": 0.80, "13123": 0.80,
}
FACTOR_COMBO_MIN = 0.75
FACTOR_COMBO_MAX = 1.40
RECUPERACION_COMBO_NOV = 0.20
RECUPERACION_COMBO_DIC = 0.30


# ============================================================
# Utilidades de normalización y carga
# ============================================================
def normalizar_codigo(serie):
    return serie.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _normalizar_encabezado(valor):
    import re, unicodedata
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("\n", " ").replace("\r", " ").strip().upper()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def _buscar_columna(columnas, aliases, prefijos=None):
    mapa = {_normalizar_encabezado(c): c for c in columnas}
    for alias in aliases:
        a = _normalizar_encabezado(alias)
        if a in mapa:
            return mapa[a]
    for prefijo in prefijos or []:
        p = _normalizar_encabezado(prefijo)
        for enc_norm, enc_orig in mapa.items():
            if enc_norm.startswith(p):
                return enc_orig
    return None


def _inferir_ano(nombre):
    import re
    m = re.search(r"(?:19|20)\d{2}", str(nombre))
    return int(m.group()) if m else None


def _inferir_mes(nombre_hoja):
    n = _normalizar_encabezado(nombre_hoja)
    for mes, num in MESES.items():
        if mes in n:
            return num
    return None


def cargar_archivos(archivos):
    partes = []
    diagnostico = []
    aliases = {
        "CODIGO": ["CODIGO", "CÓDIGO", "SKU", "COD PRODUCTO", "COD. PRODUCTO"],
        "VENTA": ["VENTA", "VENTAS", "MONTO VENTA", "VENTA NETA"],
        "UNIDADES": ["UNIDADES", "UNIDAD", "UNIDADES VENDIDAS", "CANTIDAD", "PIEZAS"],
        "MES": ["MES", "NOMBRE MES"],
        "DESCRIPCION": ["DESCRIPCION", "DESCRIPCIÓN", "PRODUCTO", "NOMBRE PRODUCTO"],
        "CATEGORIA": ["CATEGORIA", "CATEGORÍA"],
        "FAMILIA": ["FAMILIA"], "TIPO": ["TIPO"],
        "COMBOS": ["COMBOS", "COMBO", "ES COMBO"],
    }
    for archivo in archivos:
        nombre = archivo.name
        ano_archivo = _inferir_ano(nombre)
        try:
            libro = pd.ExcelFile(io.BytesIO(archivo.getvalue()), engine="openpyxl")
        except Exception as exc:
            diagnostico.append({"Archivo": nombre, "Hoja": "-", "Estado": "ERROR",
                                "Filas": 0, "Año asignado": ano_archivo,
                                "Detalle": f"No fue posible abrir el Excel: {exc}"})
            continue
        for hoja in libro.sheet_names:
            try:
                h = pd.read_excel(libro, sheet_name=hoja)
            except Exception as exc:
                diagnostico.append({"Archivo": nombre, "Hoja": hoja, "Estado": "ERROR",
                                    "Filas": 0, "Año asignado": ano_archivo,
                                    "Detalle": f"No fue posible leer la hoja: {exc}"})
                continue
            if h is None or h.empty:
                diagnostico.append({"Archivo": nombre, "Hoja": hoja, "Estado": "VACÍA",
                                    "Filas": 0, "Año asignado": ano_archivo,
                                    "Detalle": "Hoja sin registros"})
                continue
            ano_hoja = _inferir_ano(hoja)
            col_ano = _buscar_columna(h.columns, ["AÑO", "ANO", "YEAR"])
            anos_col = []
            if col_ano is not None:
                anos_col = (pd.to_numeric(h[col_ano], errors="coerce")
                            .dropna().astype(int)
                            .loc[lambda x: x.isin([2024, 2025, 2026])]
                            .unique().tolist())
            ano_res = ano_archivo or ano_hoja or (anos_col[0] if len(anos_col) == 1 else None)
            columnas = {d: _buscar_columna(h.columns, ops,
                                          prefijos=["UNIDADES"] if d == "UNIDADES" else None)
                        for d, ops in aliases.items()}
            faltantes = [c for c in ["CODIGO", "VENTA", "UNIDADES"] if columnas[c] is None]
            if ano_res not in {2024, 2025, 2026}:
                faltantes.append("AÑO")
            mes_hoja = _inferir_mes(hoja)
            if columnas["MES"] is None and mes_hoja is None:
                faltantes.append("MES")
            if faltantes:
                diagnostico.append({"Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                                    "Filas": len(h), "Año asignado": ano_res,
                                    "Detalle": "Faltan columnas: " + ", ".join(faltantes)})
                continue
            temp = pd.DataFrame(index=h.index)
            temp["CODIGO"] = h[columnas["CODIGO"]]
            temp["VENTA"] = h[columnas["VENTA"]]
            temp["UNIDADES"] = h[columnas["UNIDADES"]]
            temp["AÑO"] = int(ano_res)
            if columnas["MES"] is not None:
                mes_orig = h[columnas["MES"]]
                mes_txt = mes_orig.astype(str).map(_normalizar_encabezado)
                mes_num = pd.to_numeric(mes_orig, errors="coerce")
                temp["MES_NUM"] = mes_txt.map(MESES).fillna(mes_num)
            else:
                temp["MES_NUM"] = mes_hoja
            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = h[columnas[col]] if columnas[col] is not None else "SIN_DATO"
            temp["COMBOS"] = h[columnas["COMBOS"]] if columnas["COMBOS"] is not None else 0
            temp["CODIGO"] = normalizar_codigo(temp["CODIGO"])
            temp["VENTA"] = pd.to_numeric(temp["VENTA"], errors="coerce")
            temp["UNIDADES"] = pd.to_numeric(temp["UNIDADES"], errors="coerce")
            temp["MES_NUM"] = pd.to_numeric(temp["MES_NUM"], errors="coerce")
            temp["COMBOS"] = pd.to_numeric(temp["COMBOS"], errors="coerce").fillna(0)
            temp = temp.dropna(subset=["CODIGO", "VENTA", "UNIDADES", "MES_NUM"])
            temp = temp[temp["MES_NUM"].between(1, 12)]
            temp["MES_NUM"] = temp["MES_NUM"].astype(int)
            temp["VENTA"] = temp["VENTA"].clip(lower=0)
            temp["UNIDADES"] = temp["UNIDADES"].clip(lower=0)
            temp["COMBOS"] = temp["COMBOS"].clip(0, 1).astype(int)
            temp["Periodo"] = temp["AÑO"] * 100 + temp["MES_NUM"]
            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = temp[col].fillna("SIN_DATO").astype(str).str.strip()
            if temp.empty:
                diagnostico.append({"Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                                    "Filas": 0, "Año asignado": ano_res,
                                    "Detalle": "No quedaron filas válidas"})
                continue
            partes.append(temp)
            diagnostico.append({"Archivo": nombre, "Hoja": hoja, "Estado": "CARGADA",
                                "Filas": len(temp), "Año asignado": ano_res,
                                "Detalle": f"{temp['CODIGO'].nunique()} SKU | "
                                           f"meses {sorted(temp['MES_NUM'].unique().tolist())}"})
    if not partes:
        detalle = pd.DataFrame(diagnostico)
        resumen = "\n".join(f"{r.get('Archivo','')} | {r.get('Hoja','')}: {r.get('Detalle','')}"
                            for _, r in detalle.head(20).iterrows()) if not detalle.empty else "Sin archivos."
        raise ValueError("Ninguna hoja pudo cargarse. Diagnóstico:\n" + resumen)
    df = pd.concat(partes, ignore_index=True)
    agrupado = (df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False)
                .agg(VENTA=("VENTA", "sum"), UNIDADES=("UNIDADES", "sum"),
                     DESCRIPCION=("DESCRIPCION", "first"), CATEGORIA=("CATEGORIA", "first"),
                     FAMILIA=("FAMILIA", "first"), TIPO=("TIPO", "first"),
                     COMBOS=("COMBOS", "first")))
    agrupado["PRECIO_PROMEDIO"] = np.where(agrupado["UNIDADES"] > 0,
                                           agrupado["VENTA"] / agrupado["UNIDADES"], np.nan)
    return completar_calendario(agrupado), pd.DataFrame(diagnostico)


def completar_calendario(df):
    salida = []
    for codigo, grupo in df.groupby("CODIGO", sort=False):
        x = grupo.sort_values("Periodo").copy()
        fechas = pd.to_datetime(dict(year=x["AÑO"], month=x["MES_NUM"], day=1))
        x.index = fechas
        x = x.reindex(pd.date_range(fechas.min(), fechas.max(), freq="MS"))
        x["CODIGO"] = codigo
        x["AÑO"] = x.index.year
        x["MES_NUM"] = x.index.month
        x["Periodo"] = x["AÑO"] * 100 + x["MES_NUM"]
        x["VENTA"] = x["VENTA"].fillna(0.0)
        x["UNIDADES"] = x["UNIDADES"].fillna(0.0)
        for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO", "COMBOS"]:
            x[col] = x[col].ffill().bfill()
        x["PRECIO_PROMEDIO"] = np.where(x["UNIDADES"] > 0, x["VENTA"] / x["UNIDADES"], np.nan)
        x["PRECIO_PROMEDIO"] = x["PRECIO_PROMEDIO"].ffill()
        salida.append(x.reset_index(drop=True))
    z = pd.concat(salida, ignore_index=True)
    mediana_cat = z.groupby("CATEGORIA")["PRECIO_PROMEDIO"].transform("median")
    z["PRECIO_PROMEDIO"] = (z["PRECIO_PROMEDIO"].fillna(mediana_cat)
                            .fillna(z["PRECIO_PROMEDIO"].median()).clip(lower=0))
    z["COMBOS"] = pd.to_numeric(z["COMBOS"], errors="coerce").fillna(0).astype(int)
    return z


# ============================================================
# Carga robusta de variables macroeconómicas
# ============================================================
def cargar_variables_macro(archivo):
    """Lee CSV/Excel macro tolerando BOM, ; vs , y acentos en encabezados."""
    if archivo is None:
        return None

    contenido = archivo.getvalue() if hasattr(archivo, "getvalue") else archivo
    m = None
    intentos = []

    # 1) Intentar como texto con múltiples encodings y separadores
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                candidato = pd.read_csv(io.BytesIO(contenido), encoding=enc, sep=sep)
                if candidato.shape[1] >= 3 and len(candidato) > 0:
                    m = candidato
                    intentos.append(f"OK enc={enc} sep='{sep}' cols={list(candidato.columns)}")
                    break
            except Exception as e:
                intentos.append(f"fail enc={enc} sep='{sep}': {e}")
        if m is not None:
            break

    # 2) Fallback a Excel si no era CSV
    if m is None:
        try:
            m = pd.read_excel(io.BytesIO(contenido))
            intentos.append("OK como Excel")
        except Exception as e:
            intentos.append(f"fail Excel: {e}")

    if m is None:
        with st.expander("Diagnóstico de lectura macro"):
            st.code("\n".join(intentos))
        st.error(
            "No se pudo leer el archivo macro. Descarga la plantilla desde el botón "
            "de abajo y edítala sin cambiar los encabezados."
        )
        return None

    # 3) Normalizar encabezados (mayúsculas, sin acentos, sin BOM)
    m.columns = [_normalizar_encabezado(c).replace("\ufeff", "") for c in m.columns]

    # 4) Mapear alias a nombres canónicos
    alias = {
        "AÑO": ["AÑO", "ANO", "ANIO", "YEAR", "AÑO_MES"],
        "MES_NUM": ["MES_NUM", "MES", "MONTH", "NUM_MES"],
        "ICC_BC": ["ICC_BC", "ICC", "CONFIANZA_CONSUMIDOR", "ICC_BAJA_CALIFORNIA"],
        "INFLACION_LACTEOS_BC": [
            "INFLACION_LACTEOS_BC", "INFLACION_LACTEOS",
            "INFLACION_LACTEOS_BAJA_CALIFORNIA",
            "INFLACION", "INFLACION_BC",
        ],
        "ITAEE_BC": ["ITAEE_BC", "ITAEE", "ACTIVIDAD_ECONOMICA",
                     "ITAEE_BAJA_CALIFORNIA"],
    }
    mapeo = {}
    for canon, opciones in alias.items():
        for op in opciones:
            op_norm = _normalizar_encabezado(op)
            if op_norm in m.columns:
                mapeo[op_norm] = canon
                break
    m = m.rename(columns=mapeo)

    faltantes = [c for c in ["AÑO", "MES_NUM"] + MACRO_FEATURES if c not in m.columns]
    if faltantes:
        st.error(
            f"El archivo macro no tiene las columnas requeridas: {faltantes}.\n\n"
            f"Columnas detectadas: {list(m.columns)}\n\n"
            f"Esperado (cualquier alias de): {alias}"
        )
        with st.expander("Ver primeras filas leídas"):
            st.dataframe(m.head(5))
        return None

    # 5) Convertir tipos y limpiar
    m["AÑO"] = pd.to_numeric(m["AÑO"], errors="coerce")
    m["MES_NUM"] = pd.to_numeric(m["MES_NUM"], errors="coerce")
    for var in MACRO_FEATURES:
        m[var] = (m[var].astype(str).str.replace(",", ".", regex=False)
                  .pipe(pd.to_numeric, errors="coerce"))
    m = m.dropna(subset=["AÑO", "MES_NUM"]).copy()
    m["AÑO"] = m["AÑO"].astype(int)
    m["MES_NUM"] = m["MES_NUM"].astype(int)
    m = m.drop_duplicates(subset=["AÑO", "MES_NUM"], keep="last").sort_values(
        ["AÑO", "MES_NUM"]).reset_index(drop=True)

    st.success(
        f"Macro cargada: {len(m)} periodos | "
        f"{m['AÑO'].min()}-{m['MES_NUM'].min():02d} → "
        f"{m['AÑO'].max()}-{m['MES_NUM'].max():02d}"
    )
    return m


def merge_macro(df, df_macro):
    if df_macro is None or df_macro.empty:
        return df
    out = df.merge(df_macro, on=["AÑO", "MES_NUM"], how="left")
    out = out.sort_values(["AÑO", "MES_NUM"]).reset_index(drop=True)
    for var in MACRO_FEATURES:
        if var in out.columns:
            out[var] = out.groupby("CODIGO")[var].transform(lambda s: s.ffill().bfill())
            out[var] = out[var].fillna(out[var].median()).fillna(0)
    return out


def construir_macro_futuro(df_macro, periodos):
    if df_macro is None or df_macro.empty:
        return None
    ultimo = df_macro.sort_values(["AÑO", "MES_NUM"]).iloc[-1]
    resultado = {}
    for periodo in periodos:
        ano, mes = periodo // 100, periodo % 100
        fila = df_macro[(df_macro["AÑO"] == ano) & (df_macro["MES_NUM"] == mes)]
        if not fila.empty:
            resultado[(ano, mes)] = {v: float(fila[v].iloc[0]) for v in MACRO_FEATURES}
        else:
            resultado[(ano, mes)] = {v: float(ultimo[v]) for v in MACRO_FEATURES}
    return resultado


# ============================================================
# Features y entrenamiento
# ============================================================
def crear_features(df, objetivo, usar_macro=False):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grupo = z.groupby("CODIGO")[objetivo]
    for lag in LAGS:
        z[f"lag{lag}"] = grupo.shift(lag)
    for ventana in [3, 6, 12]:
        z[f"mean{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).mean())
        z[f"std{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).std()).fillna(0)
    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    z["ANO"] = z["AÑO"]
    z["tendencia_3_12"] = np.where(z["mean12"].abs() > 1e-9, z["mean3"] / z["mean12"], 1.0)
    for col in CAT_FEATURES:
        z[col] = z[col].fillna("SIN_DATO").astype(str)
    if usar_macro:
        for var in MACRO_FEATURES:
            if var not in z.columns:
                z[var] = 0.0
            z[var] = pd.to_numeric(z[var], errors="coerce").fillna(0)
    return z


def nuevo_modelo():
    if CatBoostRegressor is None:
        raise ImportError("Instala CatBoost con: python -m pip install catboost")
    return CatBoostRegressor(
        iterations=500, depth=7, learning_rate=0.05,
        loss_function="MAE", eval_metric="MAE",
        l2_leaf_reg=10, random_seed=42, verbose=False,
        allow_writing_files=False, thread_count=4,
    )


def entrenar_objetivo(df, objetivo, usar_macro=False):
    num_feats = get_num_features(usar_macro)
    feats = get_features(usar_macro)
    f = crear_features(df, objetivo, usar_macro).dropna(subset=["lag1", objetivo]).copy()
    medianas = f[num_feats].median().fillna(0)
    f[num_feats] = f[num_feats].fillna(medianas)
    pesos = np.where(f["MES_NUM"].isin([11, 12]), 2.5, 1.0)
    pesos *= 1 + 0.15 * (f["AÑO"] - f["AÑO"].min())
    modelo = nuevo_modelo()
    modelo.fit(f[feats], np.log1p(f[objetivo].clip(lower=0)),
               cat_features=CAT_FEATURES, sample_weight=pesos)
    return modelo, medianas


@st.cache_resource(show_spinner=False)
def entrenar_modelos(df, usar_macro=False):
    return {obj: entrenar_objetivo(df, obj, usar_macro)
            for obj in ["VENTA", "UNIDADES", "PRECIO_PROMEDIO"]}


def vector_futuro(historial, periodo, objetivo, medianas, usar_macro=False, macro_futuro=None):
    hist = historial.sort_values("Periodo")
    valores = hist[objetivo].astype(float).tolist()
    fila = {f"lag{lag}": valores[-lag] if len(valores) >= lag else np.nan for lag in LAGS}
    for ventana in [3, 6, 12]:
        arr = np.asarray(valores[-ventana:], dtype=float)
        fila[f"mean{ventana}"] = arr.mean() if len(arr) >= 2 else np.nan
        fila[f"std{ventana}"] = arr.std(ddof=1) if len(arr) >= 2 else 0.0
    mes, ano = periodo % 100, periodo // 100
    fila["sin_mes"] = np.sin(2 * np.pi * mes / 12)
    fila["cos_mes"] = np.cos(2 * np.pi * mes / 12)
    fila["MES_NUM"] = mes
    fila["ANO"] = ano
    fila["tendencia_3_12"] = (fila["mean3"] / fila["mean12"]
                              if pd.notna(fila["mean12"]) and fila["mean12"] > 0 else 1.0)
    ultimo = hist.iloc[-1]
    for col in CAT_FEATURES:
        fila[col] = str(ultimo.get(col, "SIN_DATO"))
    if usar_macro:
        valores_macro = (macro_futuro or {}).get((ano, mes), {})
        for var in MACRO_FEATURES:
            fila[var] = valores_macro.get(var, np.nan)
    feats = get_features(usar_macro)
    num_feats = get_num_features(usar_macro)
    x = pd.DataFrame([fila], columns=feats)
    x[num_feats] = x[num_feats].fillna(medianas)
    return x


def predecir_objetivo(modelo, medianas, historial, periodos, objetivo,
                      usar_macro=False, macro_futuro=None):
    hist = historial.copy()
    predicciones = []
    for periodo in periodos:
        pred = float(np.expm1(modelo.predict(
            vector_futuro(hist, periodo, objetivo, medianas, usar_macro, macro_futuro))[0]))
        pred = max(0.0, pred)
        predicciones.append(pred)
        nueva = hist.iloc[-1].copy()
        nueva["Periodo"] = periodo
        nueva["AÑO"] = periodo // 100
        nueva["MES_NUM"] = periodo % 100
        nueva[objetivo] = pred
        hist = pd.concat([hist, pd.DataFrame([nueva])], ignore_index=True)
    return predicciones


# ============================================================
# Estacionalidad y calibración
# ============================================================
def indice_estacional(historial, ano_objetivo):
    h = historial[historial["AÑO"] < ano_objetivo]
    temporada = h[h["MES_NUM"].isin([11, 12])]["VENTA"]
    resto = h[~h["MES_NUM"].isin([11, 12])]["VENTA"]
    if len(temporada) < 2 or len(resto) < 6 or resto.mean() <= 0:
        return 1.0
    return float(temporada.mean() / resto.mean())


def crecimiento_ytd(historial, ano_objetivo):
    actual = historial[(historial["AÑO"] == ano_objetivo) & (historial["MES_NUM"] <= 10)]
    previo = historial[(historial["AÑO"] == ano_objetivo - 1) & (historial["MES_NUM"] <= 10)]
    meses = sorted(set(actual["MES_NUM"]) & set(previo["MES_NUM"]))
    if len(meses) < 6:
        return 1.0
    va = actual[actual["MES_NUM"].isin(meses)]["VENTA"].sum()
    vp = previo[previo["MES_NUM"].isin(meses)]["VENTA"].sum()
    return float(np.clip(va / vp, 0.80, 1.40)) if vp > 0 else 1.0


def aplicar_ajuste_estacional(pred_base, historial, periodo):
    mes, ano = periodo % 100, periodo // 100
    indice = indice_estacional(historial, ano)
    mismos = historial[(historial["MES_NUM"] == mes) & (historial["AÑO"] < ano)] \
             .sort_values("AÑO").tail(2)
    diagnostico = {
        "Indice_estacional": float(indice), "Referencia_ultimo_ano": np.nan,
        "Referencia_historica": np.nan, "Crecimiento_YTD": 1.0,
        "Referencia_YTD": np.nan, "Prediccion_estacional": float(pred_base),
        "Referencia_suavizada": np.nan, "Prediccion_combinada": float(pred_base),
        "Piso_parcial": np.nan, "Porcentaje_recuperacion": 0.0,
        "Correccion_piso": 0.0, "Ajuste_aplicado": False,
    }
    if mismos.empty or indice < UMBRAL_ESTACIONAL:
        return max(0.0, float(pred_base)), diagnostico
    valores = mismos["VENTA"].astype(float).to_numpy()
    ultimo_ano = float(valores[-1])
    ref_hist = ultimo_ano if len(valores) == 1 else 0.65 * ultimo_ano + 0.35 * float(valores[-2])
    crec = crecimiento_ytd(historial, ano)
    ref_ytd = ref_hist * crec
    pred_est = PESO_FORECAST_ESTACIONAL * float(pred_base) + (1 - PESO_FORECAST_ESTACIONAL) * ref_ytd
    crec_suav = float(np.clip(1.0 + 0.50 * (crec - 1.0), 0.90, 1.20))
    ref_suav = ultimo_ano * crec_suav
    pred_comb = 0.90 * float(pred_est) + 0.10 * float(ref_suav)
    factor_piso = float(np.clip(1.0 + 0.35 * (crec - 1.0), 0.94, 1.15))
    piso = ultimo_ano * factor_piso
    if mes == 11:
        pct = 0.45
    elif mes == 12:
        pct = 0.60
    else:
        pct = 0.40
    if pred_comb < piso:
        corr = pct * (piso - pred_comb)
        pred_final = pred_comb + corr
    else:
        corr, pred_final = 0.0, pred_comb
    diagnostico.update({
        "Referencia_ultimo_ano": ultimo_ano, "Referencia_historica": float(ref_hist),
        "Crecimiento_YTD": float(crec), "Referencia_YTD": float(ref_ytd),
        "Prediccion_estacional": float(pred_est), "Referencia_suavizada": float(ref_suav),
        "Prediccion_combinada": float(pred_comb), "Piso_parcial": float(piso),
        "Porcentaje_recuperacion": float(pct), "Correccion_piso": float(corr),
        "Ajuste_aplicado": True,
    })
    return max(0.0, float(pred_final)), diagnostico


def normalizar_categoria_calibracion(categoria):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(categoria))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.upper().strip().split())


def aplicar_calibracion_selectiva(prediccion, categoria, es_combo=0):
    pred = max(0.0, float(prediccion))
    cat = normalizar_categoria_calibracion(categoria)
    if int(es_combo) == 1 and not APLICAR_CALIBRACION_A_COMBOS:
        return pred, 1.0, False, "Combo sin calibración"
    factor = float(np.clip(CALIBRACION_CATEGORIAS.get(cat, 1.0),
                           FACTOR_CALIBRACION_MIN, FACTOR_CALIBRACION_MAX))
    aplicada = not np.isclose(factor, 1.0)
    motivo = f"Factor selectivo de {cat}" if aplicada else "Sin corrección selectiva"
    return pred * factor, factor, aplicada, motivo


def aplicar_calibracion_combo(prediccion, codigo, categoria, es_combo=0):
    pred = max(0.0, float(prediccion))
    if int(es_combo) != 1:
        return pred, 1.0, False, "Producto regular sin calibración combo"
    cod = str(codigo).replace(".0", "").strip()
    cat = normalizar_categoria_calibracion(categoria)
    if cod in CALIBRACION_COMBOS_SKU:
        f_orig, nivel = CALIBRACION_COMBOS_SKU[cod], "SKU"
    elif cat in CALIBRACION_COMBOS_CATEGORIA:
        f_orig, nivel = CALIBRACION_COMBOS_CATEGORIA[cat], "CATEGORIA"
    else:
        f_orig, nivel = 1.0, "SIN_REGLA"
    factor = float(np.clip(f_orig, FACTOR_COMBO_MIN, FACTOR_COMBO_MAX))
    aplicada = not np.isclose(factor, 1.0)
    motivo = f"Factor combo por {nivel}: {factor:.2f}" if aplicada else "Combo sin corrección"
    return max(0.0, pred * factor), factor, aplicada, motivo


def aplicar_piso_combo_reducido(pred_comb, piso_parcial, mes):
    rec = (RECUPERACION_COMBO_NOV if int(mes) == 11
           else RECUPERACION_COMBO_DIC if int(mes) == 12 else 0.0)
    if pd.isna(piso_parcial) or pred_comb >= piso_parcial:
        return float(pred_comb), 0.0, rec
    corr = rec * (float(piso_parcial) - float(pred_comb))
    return float(pred_comb + corr), float(corr), rec


# ============================================================
# Métricas
# ============================================================
def calcular_metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, float), np.asarray(y_pred, float)
    den = float(np.abs(y_real).sum())
    wape = 100 * np.abs(y_real - y_pred).sum() / den if den > 0 else np.nan
    mae = float(np.mean(np.abs(y_real - y_pred)))
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    if len(y_real) > 1 and np.var(y_real) > 0:
        r2 = 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - y_real.mean()) ** 2)
    else:
        r2 = np.nan
    sesgo = 100 * (y_pred.sum() - y_real.sum()) / den if den > 0 else np.nan
    return {"WAPE": float(wape), "MAE": mae, "RMSE": rmse,
            "R2": float(r2) if pd.notna(r2) else np.nan,
            "SESGO": float(sesgo), "REAL": float(y_real.sum()), "PRED": float(y_pred.sum())}


# ============================================================
# Backtest
# ============================================================
@st.cache_data(show_spinner=False)
def backtest_suavizado_nov_dic_2025(df, usar_macro=False, version="v3-macro"):
    train = df[df["Periodo"] < 202511].copy()
    real = df[df["Periodo"].isin([202511, 202512])].copy()
    if train.empty or real.empty:
        return pd.DataFrame()

    macro_futuro = construir_macro_futuro(df, [202511, 202512]) if usar_macro else None
    modelos = entrenar_modelos(train, usar_macro)
    filas = []
    codigos = sorted(set(train["CODIGO"]) & set(real["CODIGO"]))

    for codigo in codigos:
        hist = train[train["CODIGO"] == codigo].sort_values("Periodo")
        obs = real[real["CODIGO"] == codigo].sort_values("Periodo")
        periodos = obs["Periodo"].astype(int).tolist()
        if hist.empty or not periodos:
            continue

        vd = predecir_objetivo(*modelos["VENTA"], hist, periodos, "VENTA", usar_macro, macro_futuro)
        unid = predecir_objetivo(*modelos["UNIDADES"], hist, periodos, "UNIDADES", usar_macro, macro_futuro)
        precio = predecir_objetivo(*modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO", usar_macro, macro_futuro)

        for (_, rr), v, u, p in zip(obs.iterrows(), vd, unid, precio):
            estructural = float(u) * float(p)
            base = PESO_VENTA_DIRECTA * float(v) + PESO_ESTRUCTURAL * estructural
            pred_sin_cal, diag = aplicar_ajuste_estacional(base, hist, int(rr["Periodo"]))
            es_combo = int(rr.get("COMBOS", 0))
            if es_combo == 1:
                pred_combo_piso, corr_combo_piso, rec_combo = aplicar_piso_combo_reducido(
                    diag["Prediccion_combinada"], diag["Piso_parcial"], int(rr["Periodo"]) % 100)
                pred_cal, fac, cal_app, motivo = aplicar_calibracion_combo(
                    pred_combo_piso, codigo, rr.get("CATEGORIA", "SIN_DATO"), es_combo)
            else:
                pred_combo_piso = pred_sin_cal
                corr_combo_piso = diag["Correccion_piso"]
                rec_combo = diag["Porcentaje_recuperacion"]
                pred_cal, fac, cal_app, motivo = aplicar_calibracion_selectiva(
                    pred_sin_cal, rr.get("CATEGORIA", "SIN_DATO"), es_combo)
            filas.append({
                "CODIGO": codigo, "DESCRIPCION": rr.get("DESCRIPCION", ""),
                "CATEGORIA": rr.get("CATEGORIA", "SIN_DATO"),
                "COMBOS": int(rr.get("COMBOS", 0)),
                "Periodo": int(rr["Periodo"]), "Actual": float(rr["VENTA"]),
                "Prediccion_base": float(base),
                "Referencia_ultimo_ano": diag["Referencia_ultimo_ano"],
                "Crecimiento_YTD": diag["Crecimiento_YTD"],
                "Prediccion_estacional": diag["Prediccion_estacional"],
                "Referencia_suavizada": diag["Referencia_suavizada"],
                "Prediccion_combinada": diag["Prediccion_combinada"],
                "Piso_parcial": diag["Piso_parcial"],
                "Porcentaje_recuperacion": diag["Porcentaje_recuperacion"],
                "Correccion_piso": diag["Correccion_piso"],
                "Ajuste_aplicado": diag["Ajuste_aplicado"],
                "Prediccion_piso_regular": float(pred_sin_cal),
                "Prediccion_sin_calibracion": float(pred_combo_piso),
                "Recuperacion_combo": float(rec_combo),
                "Correccion_piso_combo": float(corr_combo_piso),
                "Factor_calibracion_categoria": float(fac),
                "Calibracion_categoria_aplicada": bool(cal_app),
                "Motivo_calibracion": motivo,
                "Prediccion_calibrada": float(pred_cal),
                "Prediccion_final": float(pred_cal),
            })
    return pd.DataFrame(filas)


def comparar_modelos_calibracion(detalle):
    modelos = [("Combo piso reducido, sin calibración", "Prediccion_sin_calibracion"),
               ("Combo piso reducido + calibración", "Prediccion_calibrada")]
    filas = [{"Modelo": n, **calcular_metricas(detalle["Actual"], detalle[c])} for n, c in modelos]
    comp = pd.DataFrame(filas)
    base, cal = comp.iloc[0], comp.iloc[1]
    mejora = pd.DataFrame([{
        "Cambio_WAPE_pp": cal["WAPE"] - base["WAPE"],
        "Cambio_MAE": cal["MAE"] - base["MAE"],
        "Cambio_RMSE": cal["RMSE"] - base["RMSE"],
        "Cambio_R2": cal["R2"] - base["R2"],
        "Cambio_Sesgo_pp": cal["SESGO"] - base["SESGO"],
    }])
    return comp, mejora


def metricas_por_categoria_comparadas(detalle):
    filas = []
    for cat, g in detalle.groupby("CATEGORIA", dropna=False):
        b = calcular_metricas(g["Actual"], g["Prediccion_sin_calibracion"])
        c = calcular_metricas(g["Actual"], g["Prediccion_calibrada"])
        filas.append({"CATEGORIA": str(cat), "SKU": g["CODIGO"].nunique(),
                      "FACTOR": g["Factor_calibracion_categoria"].max(),
                      "WAPE_SIN": b["WAPE"], "WAPE_CAL": c["WAPE"],
                      "DELTA_WAPE": c["WAPE"] - b["WAPE"],
                      "RMSE_SIN": b["RMSE"], "RMSE_CAL": c["RMSE"],
                      "DELTA_RMSE": c["RMSE"] - b["RMSE"],
                      "SESGO_SIN": b["SESGO"], "SESGO_CAL": c["SESGO"],
                      "VENTA_REAL": b["REAL"], "PRED_SIN": b["PRED"], "PRED_CAL": c["PRED"]})
    return pd.DataFrame(filas).sort_values("VENTA_REAL", ascending=False)


def tabla_metricas_por_mes(detalle, col="Prediccion_final"):
    filas = [{"Periodo": int(p), "SKU": g["CODIGO"].nunique(),
              **calcular_metricas(g["Actual"], g[col])}
             for p, g in detalle.groupby("Periodo")]
    return pd.DataFrame(filas).sort_values("Periodo") if filas else pd.DataFrame()


def tabla_metricas_por_categoria(detalle, col="Prediccion_final"):
    filas = [{"CATEGORIA": str(c), "SKU": g["CODIGO"].nunique(),
              **calcular_metricas(g["Actual"], g[col])}
             for c, g in detalle.groupby("CATEGORIA", dropna=False)]
    return pd.DataFrame(filas).sort_values("REAL", ascending=False) if filas else pd.DataFrame()


# ============================================================
# Forecast 2026
# ============================================================
@st.cache_data(show_spinner=False)
def generar_forecast(df, usar_macro=False, version="piso-parcial-macro-v1"):
    macro_futuro = construir_macro_futuro(df, [202611, 202612]) if usar_macro else None
    modelos = entrenar_modelos(df, usar_macro)
    filas = []
    periodos = [202611, 202612]
    for codigo, historial in df.groupby("CODIGO", sort=False):
        hist = historial.sort_values("Periodo")
        if hist.empty or int(hist["Periodo"].max()) < 202608:
            continue
        meta = hist.iloc[-1]
        vd = predecir_objetivo(*modelos["VENTA"], hist, periodos, "VENTA", usar_macro, macro_futuro)
        unid = predecir_objetivo(*modelos["UNIDADES"], hist, periodos, "UNIDADES", usar_macro, macro_futuro)
        precio = predecir_objetivo(*modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO", usar_macro, macro_futuro)
        for periodo, v, u, p in zip(periodos, vd, unid, precio):
            estructural = u * p
            base = PESO_VENTA_DIRECTA * v + PESO_ESTRUCTURAL * estructural
            final, diag = aplicar_ajuste_estacional(base, hist, periodo)
            filas.append({
                "CODIGO": codigo, "DESCRIPCION": str(meta.get("DESCRIPCION", "")),
                "CATEGORIA": str(meta.get("CATEGORIA", "SIN_DATO")).strip(),
                "COMBOS": int(meta.get("COMBOS", 0)), "Periodo": periodo,
                "Venta_directa": float(v), "Unidades_pred": float(u),
                "Precio_pred": float(p), "Venta_estructural": float(estructural),
                "Prediccion_base": float(base),
                "Referencia_ultimo_ano": diag["Referencia_ultimo_ano"],
                "Crecimiento_YTD": diag["Crecimiento_YTD"],
                "Referencia_YTD": diag["Referencia_YTD"],
                "Prediccion_estacional": diag["Prediccion_estacional"],
                "Referencia_suavizada": diag["Referencia_suavizada"],
                "Prediccion_combinada": diag["Prediccion_combinada"],
                "Piso_parcial": diag["Piso_parcial"],
                "Porcentaje_recuperacion": diag["Porcentaje_recuperacion"],
                "Correccion_piso": diag["Correccion_piso"],
                "Indice_estacional": diag["Indice_estacional"],
                "Ajuste_aplicado": diag["Ajuste_aplicado"],
                "Prediccion_final": float(final),
            })
    return pd.DataFrame(filas)


def preparar_grafica(df_historico, forecast, categoria, combo, producto, fecha_corte_hist="2026-10-01"):
    futuro = forecast.copy()
    if categoria != "TODAS":
        futuro = futuro[futuro["CATEGORIA"] == categoria]
    if combo != "TODOS":
        vc = 1 if combo == "COMBOS" else 0
        futuro = futuro[futuro["COMBOS"] == vc]
    if producto != "TODOS":
        futuro = futuro[futuro["CODIGO"].astype(str) == producto]
    codigos = futuro["CODIGO"].astype(str).unique().tolist()
    hist = df_historico[df_historico["CODIGO"].astype(str).isin(codigos)].copy()
    if hist.empty or futuro.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    historico = hist.groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
    historico["Fecha"] = pd.to_datetime(dict(year=historico["AÑO"].astype(int),
                                             month=historico["MES_NUM"].astype(int), day=1))
    historico = historico[historico["Fecha"] <= pd.Timestamp(fecha_corte_hist)]
    historico = historico.rename(columns={"VENTA": "Valor"})
    historico["Serie"] = "Venta real"
    previsto = futuro.groupby("Periodo", as_index=False)["Prediccion_final"].sum()
    previsto["Fecha"] = pd.to_datetime(dict(year=(previsto["Periodo"] // 100).astype(int),
                                            month=(previsto["Periodo"] % 100).astype(int), day=1))
    previsto = previsto.rename(columns={"Prediccion_final": "Valor"})
    previsto["Serie"] = "Forecast"
    corte = historico[historico["Fecha"] == pd.Timestamp(fecha_corte_hist)][["Fecha", "Valor"]].copy()
    puente = corte if not corte.empty else historico.sort_values("Fecha").tail(1)[["Fecha", "Valor"]].copy()
    puente["Serie"] = "Forecast"
    fconectado = pd.concat([puente[["Fecha", "Valor", "Serie"]],
                            previsto[["Fecha", "Valor", "Serie"]]],
                           ignore_index=True).sort_values("Fecha")
    return (historico[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
            fconectado, previsto[["Fecha", "Valor", "Serie"]].sort_values("Fecha"))


# ============================================================
# UI principal
# ============================================================
st.markdown("### 1) Archivos de ventas")
archivos = st.file_uploader("Carga los Excel de ventas 2024, 2025 y 2026",
                            type=["xlsx"], accept_multiple_files=True)

st.markdown("### 2) Variables macroeconómicas (opcional)")
st.caption(
    "Sube un CSV con columnas AÑO, MES_NUM, ICC_BC, INFLACION_LACTEOS_BC, ITAEE_BC. "
    "Acepta separador `,` o `;`, y encoding UTF-8 o Latin-1."
)

PLANTILLA_MACRO = """AÑO,MES_NUM,ICC_BC,INFLACION_LACTEOS_BC,ITAEE_BC
2024,1,45.8,4.5,2.4
2024,2,46.1,4.6,2.2
2024,3,46.5,4.8,2.5
2024,4,46.2,4.9,2.3
2024,5,45.9,5.0,2.1
2024,6,45.5,5.1,1.9
2024,7,45.2,5.2,1.8
2024,8,44.9,5.3,1.6
2024,9,44.5,5.4,1.4
2024,10,44.1,5.5,1.2
2024,11,44.3,5.6,1.3
2024,12,44.6,5.7,1.5
2025,1,44.0,5.9,1.0
2025,2,43.8,6.0,0.8
2025,3,43.9,6.1,0.6
2025,4,43.5,6.2,0.4
2025,5,43.2,6.3,0.1
2025,6,43.0,6.2,-0.2
2025,7,42.9,6.3,-0.5
2025,8,42.8,6.4,-0.8
2025,9,42.9,6.4,-1.2
2025,10,42.7,6.4,-1.5
2025,11,42.5,6.5,-1.6
2025,12,43.0,6.6,-1.4
2026,1,42.4,6.6,-1.7
2026,2,42.3,6.7,-1.8
2026,3,42.2,6.8,-1.9
2026,4,42.1,6.8,-1.9
2026,5,42.0,6.9,-2.0
2026,6,42.0,6.9,-2.0
2026,7,41.9,7.0,-2.1
2026,8,41.8,7.0,-2.1
2026,9,41.7,7.1,-2.2
2026,10,41.6,7.1,-2.2
2026,11,41.5,7.1,-2.3
2026,12,41.7,7.2,-2.1
"""

col_upl, col_dl = st.columns([3, 1])
with col_dl:
    st.download_button(
        "⬇️ Descargar plantilla CSV",
        PLANTILLA_MACRO.encode("utf-8-sig"),
        "macro_bc_plantilla.csv",
        "text/csv",
        help="Descarga este archivo, edítalo en Excel o bloc de notas, y súbelo abajo.",
    )
with col_upl:
    macro_archivo = st.file_uploader(
        "CSV macro Baja California", type=["csv", "txt", "xlsx"], key="macro_csv"
    )

usar_macro = st.toggle(
    "Activar variables macroeconómicas en el modelo",
    value=False,
    help="Requiere CSV macro cargado arriba.",
)

df_macro = None
if usar_macro:
    df_macro = cargar_variables_macro(macro_archivo)
    if df_macro is None:
        st.warning("Toggle activo pero no hay CSV macro válido. Se usará modo sin macro.")
        usar_macro = False

st.markdown("### 3) Modo de ejecución")
modo = st.radio("Modo",
                ["FORECAST 2026", "BACKTEST 2025"],
                horizontal=True,
                help="BACKTEST entrena solo hasta Oct 2025 y valida contra Nov-Dic 2025 reales.")
modo_key = "backtest_2025" if modo == "BACKTEST 2025" else "forecast_2026"

if archivos:
    try:
        df_raw, diagnostico_carga = cargar_archivos(archivos)
        df = merge_macro(df_raw, df_macro) if usar_macro and df_macro is not None else df_raw
        resumen_anual = (df.groupby("AÑO", as_index=False)
                         .agg(SKU=("CODIGO", "nunique"), Registros=("Periodo", "size"),
                              Desde=("Periodo", "min"), Hasta=("Periodo", "max")))
        st.dataframe(resumen_anual, use_container_width=True, hide_index=True)
        with st.expander("Diagnóstico de carga"):
            st.dataframe(diagnostico_carga, use_container_width=True, hide_index=True)
        anos = set(resumen_anual["AÑO"].astype(int).tolist())
        requeridos = {2024, 2025} if modo_key == "backtest_2025" else {2024, 2025, 2026}
        faltan = sorted(requeridos - anos)
        if faltan:
            st.error(f"No se cargaron datos de {faltan}. Revisa el diagnóstico.")
            st.stop()

        st.info(f"Modo: **{modo}** | Macro: **{'ACTIVA' if usar_macro else 'INACTIVA'}**")

        # ---------- BACKTEST ----------
        if modo_key == "backtest_2025":
            st.markdown("### Backtest Nov-Dic 2025")
            segmento = st.segmented_control(
                "Población para métricas",
                ["TODOS", "SIN COMBOS", "SOLO COMBOS"],
                default="SIN COMBOS", key="seg_bt")
            segmento = segmento or "SIN COMBOS"

            if st.button("Ejecutar backtest", type="primary"):
                with st.spinner("Entrenando modelos (con y sin macro)..."):
                    detalle_sin = backtest_suavizado_nov_dic_2025(df_raw, usar_macro=False)
                    detalle_con = (backtest_suavizado_nov_dic_2025(df, usar_macro=True)
                                   if usar_macro else pd.DataFrame())
                    st.session_state["bt_sin"] = detalle_sin
                    st.session_state["bt_con"] = detalle_con
                st.rerun()

            if "bt_sin" in st.session_state:
                def filtrar_seg(d):
                    if d is None or d.empty:
                        return pd.DataFrame()
                    if segmento == "SOLO COMBOS":
                        return d[d["COMBOS"] == 1].copy()
                    if segmento == "SIN COMBOS":
                        return d[d["COMBOS"] == 0].copy()
                    return d.copy()

                sin = filtrar_seg(st.session_state["bt_sin"])
                con = filtrar_seg(st.session_state.get("bt_con"))

                if sin.empty:
                    st.warning("Sin registros para el segmento seleccionado.")
                else:
                    m_sin = calcular_metricas(sin["Actual"], sin["Prediccion_calibrada"])
                    filas_comp = [{"Escenario": "Sin macro", **m_sin}]
                    if not con.empty:
                        m_con = calcular_metricas(con["Actual"], con["Prediccion_calibrada"])
                        filas_comp.append({"Escenario": "Con macro", **m_con})
                    comp = pd.DataFrame(filas_comp)

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("WAPE sin macro", f"{m_sin['WAPE']:.2f}%")
                    c2.metric("MAE sin macro", f"${m_sin['MAE']:,.0f}")
                    c3.metric("RMSE sin macro", f"${m_sin['RMSE']:,.0f}")
                    c4.metric("R² sin macro", f"{m_sin['R2']:.4f}" if pd.notna(m_sin['R2']) else "N/D")
                    c5.metric("Sesgo sin macro", f"{m_sin['SESGO']:+.2f}%")

                    if not con.empty:
                        d = comp.iloc[1]
                        b = comp.iloc[0]
                        d1, d2, d3, d4, d5 = st.columns(5)
                        d1.metric("Δ WAPE", f"{d['WAPE']-b['WAPE']:+.2f} pp")
                        d2.metric("Δ MAE", f"${d['MAE']-b['MAE']:+,.0f}")
                        d3.metric("Δ RMSE", f"${d['RMSE']-b['RMSE']:+,.0f}")
                        d4.metric("Δ R²", f"{d['R2']-b['R2']:+.4f}")
                        d5.metric("Δ Sesgo", f"{d['SESGO']-b['SESGO']:+.2f} pp")

                    st.markdown("#### Comparación completa")
                    st.dataframe(comp.style.format({
                        "WAPE": "{:.2f}%", "MAE": "${:,.2f}",
                        "RMSE": "${:,.2f}", "R2": "{:.4f}",
                        "SESGO": "{:+.2f}%", "REAL": "${:,.2f}", "PRED": "${:,.2f}",
                    }), use_container_width=True, hide_index=True)

                    if not con.empty:
                        if (d['WAPE'] < b['WAPE']) and (d['SESGO'] > b['SESGO']):
                            st.success("El macro mejora WAPE y reduce el sesgo negativo. "
                                       "Recomendado activarlo en producción.")
                        elif d['WAPE'] < b['WAPE']:
                            st.warning("El macro mejora WAPE pero no corrige el sesgo. "
                                       "Revisa cobertura del CSV macro en Nov-Dic.")
                        else:
                            st.error("El macro no mejora el WAPE. No se recomienda activarlo.")

                    st.markdown("#### Detalle por categoría")
                    if not con.empty:
                        cat_sin = tabla_metricas_por_categoria(sin, "Prediccion_calibrada")
                        cat_con = tabla_metricas_por_categoria(con, "Prediccion_calibrada")
                        comp_cat = cat_sin.merge(cat_con, on="CATEGORIA",
                                                 suffixes=("_sin", "_con"), how="outer")
                        st.dataframe(comp_cat, use_container_width=True, hide_index=True)

                    with st.expander("Detalle SKU (sin macro)"):
                        st.dataframe(sin, use_container_width=True, hide_index=True)
                    if not con.empty:
                        with st.expander("Detalle SKU (con macro)"):
                            st.dataframe(con, use_container_width=True, hide_index=True)

        # ---------- FORECAST 2026 ----------
        else:
            st.markdown("### Forecast Nov-Dic 2026")
            if st.button("Generar forecast 2026", type="primary"):
                with st.spinner("Generando forecast..."):
                    st.session_state["fc_2026"] = generar_forecast(df, usar_macro=usar_macro)
                st.rerun()

            if "fc_2026" in st.session_state:
                forecast = st.session_state["fc_2026"]
                if forecast.empty:
                    st.error("No hay SKU activos con datos recientes.")
                    st.stop()
                categorias = ["TODAS"] + sorted(
                    forecast["CATEGORIA"].dropna().astype(str).str.strip().unique().tolist())
                c1, c2, c3 = st.columns([2, 1.25, 3])
                with c1:
                    cat = st.selectbox("Categoría", categorias)
                with c2:
                    combo = st.segmented_control("Tipo",
                                                 ["TODOS", "SIN COMBOS", "COMBOS"],
                                                 default="SIN COMBOS") or "SIN COMBOS"
                universo = forecast.copy()
                if cat != "TODAS":
                    universo = universo[universo["CATEGORIA"] == cat]
                if combo != "TODOS":
                    universo = universo[universo["COMBOS"] == (1 if combo == "COMBOS" else 0)]
                prods = universo[["CODIGO", "DESCRIPCION"]].drop_duplicates()
                prods["ETIQUETA"] = prods["CODIGO"].astype(str) + " | " + prods["DESCRIPCION"].astype(str)
                mp = dict(zip(prods["ETIQUETA"], prods["CODIGO"].astype(str)))
                with c3:
                    sel = st.selectbox("Producto", ["TODOS"] + sorted(mp))
                prod = "TODOS" if sel == "TODOS" else mp[sel]
                h, fl, fp = preparar_grafica(df, forecast, cat, combo, prod)
                if h.empty or fp.empty:
                    st.warning("Sin datos comparables para la selección.")
                    st.stop()
                nov = fp.loc[fp["Fecha"] == pd.Timestamp("2026-11-01"), "Valor"].sum()
                dic = fp.loc[fp["Fecha"] == pd.Timestamp("2026-12-01"), "Valor"].sum()
                oct_real = h.loc[h["Fecha"] == pd.Timestamp("2026-10-01"), "Valor"].sum()
                cr_nov = (nov / oct_real - 1) * 100 if oct_real > 0 else np.nan
                cr_dic = (dic / nov - 1) * 100 if nov > 0 else np.nan
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Octubre 2026 real", f"${oct_real:,.0f}")
                k2.metric("Noviembre 2026", f"${nov:,.0f}",
                          f"{cr_nov:+.1f}% vs octubre" if pd.notna(cr_nov) else None)
                k3.metric("Diciembre 2026", f"${dic:,.0f}",
                          f"{cr_dic:+.1f}% vs noviembre" if pd.notna(cr_dic) else None)
                k4.metric("SKU incluidos", f"{universo['CODIGO'].nunique():,}")

                eje_x = alt.X("Fecha:T", title=None,
                              scale=alt.Scale(domain=[pd.Timestamp("2024-01-01"),
                                                      pd.Timestamp("2026-12-01")]),
                              axis=alt.Axis(format="%b %Y", labelAngle=-35,
                                            tickCount={"interval": "month", "step": 2}))
                eje_y = alt.Y("Valor:Q", title="Venta mensual",
                              scale=alt.Scale(zero=False, nice=True),
                              axis=alt.Axis(format="$,.2s"))
                tt = [alt.Tooltip("Fecha:T", format="%B %Y"),
                      alt.Tooltip("Valor:Q", format="$,.2f")]
                chart = alt.layer(
                    alt.Chart(h).mark_line(color="#3B82F6", strokeWidth=3).encode(x=eje_x, y=eje_y, tooltip=tt),
                    alt.Chart(h).mark_circle(color="#3B82F6", size=44).encode(x=eje_x, y=eje_y, tooltip=tt),
                    alt.Chart(fl).mark_line(color="#F59E0B", strokeWidth=3.5, strokeDash=[8, 5]).encode(x=eje_x, y=eje_y, tooltip=tt),
                    alt.Chart(fp).mark_circle(color="#F59E0B", size=125, stroke="white", strokeWidth=1.5).encode(x=eje_x, y=eje_y, tooltip=tt),
                ).properties(height=500, title="Forecast Nov-Dic 2026")
                st.altair_chart(chart, use_container_width=True)
                with st.expander("Diagnóstico del forecast"):
                    st.dataframe(universo, use_container_width=True, hide_index=True)
                    st.download_button("Descargar CSV",
                                       universo.to_csv(index=False).encode("utf-8-sig"),
                                       "forecast_nov_dic_2026.csv", "text/csv")

    except Exception as error:
        st.exception(error)
else:
    st.info("Carga los Excel de ventas para comenzar.")
