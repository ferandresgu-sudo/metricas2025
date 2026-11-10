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
NUM_FEATURES = (
    [f"lag{x}" for x in LAGS]
    + [
        "mean3", "std3", "mean6", "std6", "mean12", "std12",
        "sin_mes", "cos_mes", "MES_NUM", "ANO", "tendencia_3_12",
    ]
)
FEATURES = NUM_FEATURES + CAT_FEATURES

# Configuración seleccionada en el backtest Nov-Dic 2025.
PESO_VENTA_DIRECTA = 0.85
PESO_ESTRUCTURAL = 0.15
PESO_FORECAST_ESTACIONAL = 0.85
UMBRAL_ESTACIONAL = 1.20


def normalizar_codigo(serie):
    return serie.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _normalizar_encabezado(valor):
    """Normaliza encabezados de Excel sin depender de espacios, acentos o saltos."""
    import re
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("\n", " ").replace("\r", " ").strip().upper()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def _buscar_columna(columnas, aliases, prefijos=None):
    """Busca encabezados exactos y también variantes como UNIDADES 2025."""
    mapa = {_normalizar_encabezado(c): c for c in columnas}
    for alias in aliases:
        alias_normalizado = _normalizar_encabezado(alias)
        if alias_normalizado in mapa:
            return mapa[alias_normalizado]
    for prefijo in prefijos or []:
        prefijo_normalizado = _normalizar_encabezado(prefijo)
        for encabezado_normalizado, encabezado_original in mapa.items():
            if encabezado_normalizado.startswith(prefijo_normalizado):
                return encabezado_original
    return None


def _inferir_ano(nombre_archivo):
    import re
    coincidencia = re.search(r"(?:19|20)\d{2}", str(nombre_archivo))
    return int(coincidencia.group()) if coincidencia else None


def _inferir_mes(nombre_hoja):
    nombre = _normalizar_encabezado(nombre_hoja)
    for mes, numero in MESES.items():
        if mes in nombre:
            return numero
    return None


def cargar_archivos(archivos):
    """Carga robusta de los tres Excel.

    El año se toma del nombre VENTAS 2024/2025/2026. La función acepta una hoja
    por mes o una tabla consolidada, y reconoce variaciones de encabezados.
    No usa caché para evitar reutilizar una lectura anterior incompleta.
    """
    partes = []
    diagnostico = []

    aliases = {
        "CODIGO": ["CODIGO", "CÓDIGO", "SKU", "COD PRODUCTO", "COD. PRODUCTO"],
        "VENTA": ["VENTA", "VENTAS", "MONTO VENTA", "VENTA NETA"],
        "UNIDADES": ["UNIDADES", "UNIDAD", "UNIDADES VENDIDAS", "CANTIDAD", "PIEZAS"],
        "MES": ["MES", "NOMBRE MES"],
        "DESCRIPCION": ["DESCRIPCION", "DESCRIPCIÓN", "PRODUCTO", "NOMBRE PRODUCTO"],
        "CATEGORIA": ["CATEGORIA", "CATEGORÍA"],
        "FAMILIA": ["FAMILIA"],
        "TIPO": ["TIPO"],
        "COMBOS": ["COMBOS", "COMBO", "ES COMBO"],
    }

    for archivo in archivos:
        nombre = archivo.name
        ano_archivo = _inferir_ano(nombre)

        try:
            contenido = archivo.getvalue()
            libro = pd.ExcelFile(io.BytesIO(contenido), engine="openpyxl")
        except Exception as exc:
            diagnostico.append({
                "Archivo": nombre, "Hoja": "-", "Estado": "ERROR",
                "Filas": 0, "Año asignado": ano_archivo,
                "Detalle": f"No fue posible abrir el Excel: {exc}",
            })
            continue

        for hoja in libro.sheet_names:
            try:
                h = pd.read_excel(libro, sheet_name=hoja)
            except Exception as exc:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "ERROR",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": f"No fue posible leer la hoja: {exc}",
                })
                continue

            if h is None or h.empty:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "VACÍA",
                    "Filas": 0, "Año asignado": ano_archivo,
                    "Detalle": "Hoja sin registros",
                })
                continue

            # El año se resuelve en este orden: nombre del archivo, nombre de hoja,
            # columna AÑO. Esto evita rechazar archivos renombrados por Streamlit/GitHub.
            ano_hoja = _inferir_ano(hoja)
            col_ano_original = _buscar_columna(h.columns, ["AÑO", "ANO", "YEAR"])
            anos_columna = []
            if col_ano_original is not None:
                anos_columna = (
                    pd.to_numeric(h[col_ano_original], errors="coerce")
                    .dropna().astype(int).loc[lambda x: x.isin([2024, 2025, 2026])]
                    .unique().tolist()
                )
            ano_resuelto = ano_archivo or ano_hoja or (anos_columna[0] if len(anos_columna) == 1 else None)

            columnas = {
                destino: _buscar_columna(
                    h.columns,
                    opciones,
                    prefijos=["UNIDADES"] if destino == "UNIDADES" else None,
                )
                for destino, opciones in aliases.items()
            }
            obligatorias = ["CODIGO", "VENTA", "UNIDADES"]
            faltantes = [c for c in obligatorias if columnas[c] is None]
            if ano_resuelto not in {2024, 2025, 2026}:
                faltantes.append("AÑO")
            mes_hoja = _inferir_mes(hoja)
            if columnas["MES"] is None and mes_hoja is None:
                faltantes.append("MES")

            if faltantes:
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                    "Filas": len(h), "Año asignado": ano_resuelto,
                    "Detalle": "Faltan columnas: " + ", ".join(faltantes),
                })
                continue

            temp = pd.DataFrame(index=h.index)
            temp["CODIGO"] = h[columnas["CODIGO"]]
            temp["VENTA"] = h[columnas["VENTA"]]
            temp["UNIDADES"] = h[columnas["UNIDADES"]]
            temp["AÑO"] = int(ano_resuelto)

            if columnas["MES"] is not None:
                mes_original = h[columnas["MES"]]
                mes_texto = mes_original.astype(str).map(_normalizar_encabezado)
                mes_numero = pd.to_numeric(mes_original, errors="coerce")
                temp["MES_NUM"] = mes_texto.map(MESES).fillna(mes_numero)
            else:
                temp["MES_NUM"] = mes_hoja

            for col in ["DESCRIPCION", "CATEGORIA", "FAMILIA", "TIPO"]:
                temp[col] = (
                    h[columnas[col]] if columnas[col] is not None else "SIN_DATO"
                )
            temp["COMBOS"] = (
                h[columnas["COMBOS"]] if columnas["COMBOS"] is not None else 0
            )

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
                diagnostico.append({
                    "Archivo": nombre, "Hoja": hoja, "Estado": "OMITIDA",
                    "Filas": 0, "Año asignado": ano_resuelto,
                    "Detalle": "No quedaron filas válidas después de limpiar",
                })
                continue

            partes.append(temp)
            diagnostico.append({
                "Archivo": nombre, "Hoja": hoja, "Estado": "CARGADA",
                "Filas": len(temp), "Año asignado": ano_resuelto,
                "Detalle": (
                    f"{temp['CODIGO'].nunique()} SKU | meses {sorted(temp['MES_NUM'].unique().tolist())} | "
                    f"unidades leídas desde: {columnas['UNIDADES']}"
                ),
            })

    if not partes:
        detalle_error = pd.DataFrame(diagnostico)
        if not detalle_error.empty:
            resumen_error = "\n".join(
                f"{r.get('Archivo', '')} | {r.get('Hoja', '')}: {r.get('Detalle', '')}"
                for _, r in detalle_error.head(20).iterrows()
            )
        else:
            resumen_error = "No se recibieron archivos o pandas no pudo abrirlos."
        raise ValueError(
            "Ninguna hoja pudo cargarse. Diagnóstico inicial:\n" + resumen_error
        )

    df = pd.concat(partes, ignore_index=True)
    agrupado = (
        df.groupby(["CODIGO", "Periodo", "MES_NUM", "AÑO"], as_index=False)
        .agg(
            VENTA=("VENTA", "sum"),
            UNIDADES=("UNIDADES", "sum"),
            DESCRIPCION=("DESCRIPCION", "first"),
            CATEGORIA=("CATEGORIA", "first"),
            FAMILIA=("FAMILIA", "first"),
            TIPO=("TIPO", "first"),
            COMBOS=("COMBOS", "first"),
        )
    )
    agrupado["PRECIO_PROMEDIO"] = np.where(
        agrupado["UNIDADES"] > 0,
        agrupado["VENTA"] / agrupado["UNIDADES"],
        np.nan,
    )
    salida = completar_calendario(agrupado)
    return salida, pd.DataFrame(diagnostico)


def completar_calendario(df):
    salida = []
    for codigo, grupo in df.groupby("CODIGO", sort=False):
        x = grupo.sort_values("Periodo").copy()
        fechas = pd.to_datetime(
            dict(year=x["AÑO"], month=x["MES_NUM"], day=1)
        )
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
        x["PRECIO_PROMEDIO"] = np.where(
            x["UNIDADES"] > 0, x["VENTA"] / x["UNIDADES"], np.nan
        )
        x["PRECIO_PROMEDIO"] = x["PRECIO_PROMEDIO"].ffill()
        salida.append(x.reset_index(drop=True))

    z = pd.concat(salida, ignore_index=True)
    mediana_categoria = z.groupby("CATEGORIA")["PRECIO_PROMEDIO"].transform("median")
    z["PRECIO_PROMEDIO"] = (
        z["PRECIO_PROMEDIO"]
        .fillna(mediana_categoria)
        .fillna(z["PRECIO_PROMEDIO"].median())
        .clip(lower=0)
    )
    z["COMBOS"] = pd.to_numeric(z["COMBOS"], errors="coerce").fillna(0).astype(int)
    return z


def crear_features(df, objetivo):
    z = df.sort_values(["CODIGO", "Periodo"]).copy()
    grupo = z.groupby("CODIGO")[objetivo]

    for lag in LAGS:
        z[f"lag{lag}"] = grupo.shift(lag)

    for ventana in [3, 6, 12]:
        z[f"mean{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).mean()
        )
        z[f"std{ventana}"] = grupo.transform(
            lambda s: s.shift(1).rolling(ventana, min_periods=2).std()
        ).fillna(0)

    z["sin_mes"] = np.sin(2 * np.pi * z["MES_NUM"] / 12)
    z["cos_mes"] = np.cos(2 * np.pi * z["MES_NUM"] / 12)
    z["ANO"] = z["AÑO"]
    z["tendencia_3_12"] = np.where(
        z["mean12"].abs() > 1e-9, z["mean3"] / z["mean12"], 1.0
    )

    for col in CAT_FEATURES:
        z[col] = z[col].fillna("SIN_DATO").astype(str)
    return z


def nuevo_modelo():
    if CatBoostRegressor is None:
        raise ImportError("Instala CatBoost con: python -m pip install catboost")
    return CatBoostRegressor(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=10,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def entrenar_objetivo(df, objetivo):
    f = crear_features(df, objetivo).dropna(subset=["lag1", objetivo]).copy()
    medianas = f[NUM_FEATURES].median().fillna(0)
    f[NUM_FEATURES] = f[NUM_FEATURES].fillna(medianas)
    pesos = np.where(f["MES_NUM"].isin([11, 12]), 2.5, 1.0)
    pesos *= 1 + 0.15 * (f["AÑO"] - f["AÑO"].min())

    modelo = nuevo_modelo()
    modelo.fit(
        f[FEATURES],
        np.log1p(f[objetivo].clip(lower=0)),
        cat_features=CAT_FEATURES,
        sample_weight=pesos,
    )
    return modelo, medianas


@st.cache_resource(show_spinner=False)
def entrenar_modelos(df):
    return {
        objetivo: entrenar_objetivo(df, objetivo)
        for objetivo in ["VENTA", "UNIDADES", "PRECIO_PROMEDIO"]
    }


def vector_futuro(historial, periodo, objetivo, medianas):
    hist = historial.sort_values("Periodo")
    valores = hist[objetivo].astype(float).tolist()
    fila = {
        f"lag{lag}": valores[-lag] if len(valores) >= lag else np.nan
        for lag in LAGS
    }

    for ventana in [3, 6, 12]:
        arreglo = np.asarray(valores[-ventana:], dtype=float)
        fila[f"mean{ventana}"] = arreglo.mean() if len(arreglo) >= 2 else np.nan
        fila[f"std{ventana}"] = arreglo.std(ddof=1) if len(arreglo) >= 2 else 0.0

    mes = periodo % 100
    ano = periodo // 100
    fila["sin_mes"] = np.sin(2 * np.pi * mes / 12)
    fila["cos_mes"] = np.cos(2 * np.pi * mes / 12)
    fila["MES_NUM"] = mes
    fila["ANO"] = ano
    fila["tendencia_3_12"] = (
        fila["mean3"] / fila["mean12"]
        if pd.notna(fila["mean12"]) and fila["mean12"] > 0
        else 1.0
    )

    ultimo = hist.iloc[-1]
    for col in CAT_FEATURES:
        fila[col] = str(ultimo.get(col, "SIN_DATO"))

    x = pd.DataFrame([fila], columns=FEATURES)
    x[NUM_FEATURES] = x[NUM_FEATURES].fillna(medianas)
    return x


def predecir_objetivo(modelo, medianas, historial, periodos, objetivo):
    hist = historial.copy()
    predicciones = []

    for periodo in periodos:
        pred = float(
            np.expm1(
                modelo.predict(vector_futuro(hist, periodo, objetivo, medianas))[0]
            )
        )
        pred = max(0.0, pred)
        predicciones.append(pred)

        nueva = hist.iloc[-1].copy()
        nueva["Periodo"] = periodo
        nueva["AÑO"] = periodo // 100
        nueva["MES_NUM"] = periodo % 100
        nueva[objetivo] = pred
        hist = pd.concat([hist, pd.DataFrame([nueva])], ignore_index=True)

    return predicciones


def indice_estacional(historial, ano_objetivo):
    h = historial[historial["AÑO"] < ano_objetivo]
    temporada = h[h["MES_NUM"].isin([11, 12])]["VENTA"]
    resto = h[~h["MES_NUM"].isin([11, 12])]["VENTA"]
    if len(temporada) < 2 or len(resto) < 6 or resto.mean() <= 0:
        return 1.0
    return float(temporada.mean() / resto.mean())


def crecimiento_ytd(historial, ano_objetivo):
    actual = historial[
        (historial["AÑO"] == ano_objetivo) & (historial["MES_NUM"] <= 10)
    ]
    previo = historial[
        (historial["AÑO"] == ano_objetivo - 1) & (historial["MES_NUM"] <= 10)
    ]
    meses = sorted(set(actual["MES_NUM"]) & set(previo["MES_NUM"]))
    if len(meses) < 6:
        return 1.0
    va = actual[actual["MES_NUM"].isin(meses)]["VENTA"].sum()
    vp = previo[previo["MES_NUM"].isin(meses)]["VENTA"].sum()
    return float(np.clip(va / vp, 0.80, 1.40)) if vp > 0 else 1.0


def aplicar_ajuste_estacional(pred_base, historial, periodo):
    """Ajuste estacional con piso parcial, no rígido.

    La referencia del año anterior protege contra subestimaciones extremas, pero
    nunca sustituye automáticamente la predicción. Si el modelo queda debajo del
    piso parcial, solo se recupera 40% de la distancia hacia dicho piso.
    """
    mes = periodo % 100
    ano = periodo // 100
    indice = indice_estacional(historial, ano)

    mismos = historial[
        (historial["MES_NUM"] == mes)
        & (historial["AÑO"] < ano)
    ].sort_values("AÑO").tail(2)

    diagnostico = {
        "Indice_estacional": float(indice),
        "Referencia_ultimo_ano": np.nan,
        "Referencia_historica": np.nan,
        "Crecimiento_YTD": 1.0,
        "Referencia_YTD": np.nan,
        "Prediccion_estacional": float(pred_base),
        "Referencia_suavizada": np.nan,
        "Prediccion_combinada": float(pred_base),
        "Piso_parcial": np.nan,
        "Porcentaje_recuperacion": 0.0,
        "Correccion_piso": 0.0,
        "Ajuste_aplicado": False,
    }

    if mismos.empty or indice < UMBRAL_ESTACIONAL:
        return max(0.0, float(pred_base)), diagnostico

    valores = mismos["VENTA"].astype(float).to_numpy()
    ultimo_ano = float(valores[-1])
    referencia_historica = (
        ultimo_ano
        if len(valores) == 1
        else 0.65 * ultimo_ano + 0.35 * float(valores[-2])
    )

    crec = crecimiento_ytd(historial, ano)
    referencia_ytd = referencia_historica * crec

    prediccion_estacional = (
        PESO_FORECAST_ESTACIONAL * float(pred_base)
        + (1.0 - PESO_FORECAST_ESTACIONAL) * referencia_ytd
    )

    # Referencia suavizada: transfiere 50% de la tendencia YTD y la limita.
    crecimiento_suavizado = float(np.clip(
        1.0 + 0.50 * (crec - 1.0),
        0.90,
        1.20,
    ))
    referencia_suavizada = ultimo_ano * crecimiento_suavizado

    # Ensamble principal: 90% señal del modelo estacional, 10% referencia suave.
    prediccion_combinada = (
        0.90 * float(prediccion_estacional)
        + 0.10 * float(referencia_suavizada)
    )

    # Piso PARCIAL: acepta disminuciones, pero controla caídas extremas.
    # El piso puede estar entre 94% y 115% del mismo mes del último año.
    factor_piso = float(np.clip(
        1.0 + 0.35 * (crec - 1.0),
        0.94,
        1.15,
    ))
    piso_parcial = ultimo_ano * factor_piso

    # Recuperación diferenciada por mes según el backtest:
    # noviembre requiere protección moderada y diciembre una protección mayor.
    if mes == 11:
        porcentaje_recuperacion = 0.45
    elif mes == 12:
        porcentaje_recuperacion = 0.60
    else:
        porcentaje_recuperacion = 0.40

    # Si queda debajo, recupera solo una parte de la distancia. No copia el piso.
    if prediccion_combinada < piso_parcial:
        correccion_piso = (
            porcentaje_recuperacion
            * (piso_parcial - prediccion_combinada)
        )
        prediccion_final = prediccion_combinada + correccion_piso
    else:
        correccion_piso = 0.0
        prediccion_final = prediccion_combinada

    diagnostico.update({
        "Referencia_ultimo_ano": ultimo_ano,
        "Referencia_historica": float(referencia_historica),
        "Crecimiento_YTD": float(crec),
        "Referencia_YTD": float(referencia_ytd),
        "Prediccion_estacional": float(prediccion_estacional),
        "Referencia_suavizada": float(referencia_suavizada),
        "Prediccion_combinada": float(prediccion_combinada),
        "Piso_parcial": float(piso_parcial),
        "Porcentaje_recuperacion": float(porcentaje_recuperacion),
        "Correccion_piso": float(correccion_piso),
        "Ajuste_aplicado": True,
    })
    return max(0.0, float(prediccion_final)), diagnostico


def calcular_metricas(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominador = float(np.abs(y_real).sum())
    wape = 100.0 * np.abs(y_real - y_pred).sum() / denominador if denominador > 0 else np.nan
    mae = float(np.mean(np.abs(y_real - y_pred)))
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    if len(y_real) > 1 and np.var(y_real) > 0:
        r2 = 1.0 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - y_real.mean()) ** 2)
    else:
        r2 = np.nan
    sesgo = 100.0 * (y_pred.sum() - y_real.sum()) / denominador if denominador > 0 else np.nan
    return {
        "WAPE": float(wape), "MAE": mae, "RMSE": rmse,
        "R2": float(r2) if pd.notna(r2) else np.nan,
        "SESGO": float(sesgo),
        "REAL": float(y_real.sum()), "PRED": float(y_pred.sum()),
    }


@st.cache_data(show_spinner=False)
def backtest_suavizado_nov_dic_2025(df, version="piso-parcial-mensual-backtest-v3"):
    """Entrena hasta octubre 2025 y predice Nov-Dic 2025 con la regla suavizada."""
    train = df[df["Periodo"] < 202511].copy()
    real = df[df["Periodo"].isin([202511, 202512])].copy()
    if train.empty or real.empty:
        return pd.DataFrame()

    modelos = entrenar_modelos(train)
    filas = []
    codigos = sorted(set(train["CODIGO"]) & set(real["CODIGO"]))

    for codigo in codigos:
        hist = train[train["CODIGO"] == codigo].sort_values("Periodo")
        observado = real[real["CODIGO"] == codigo].sort_values("Periodo")
        periodos = observado["Periodo"].astype(int).tolist()
        if hist.empty or not periodos:
            continue

        venta_directa = predecir_objetivo(*modelos["VENTA"], hist, periodos, "VENTA")
        unidades = predecir_objetivo(*modelos["UNIDADES"], hist, periodos, "UNIDADES")
        precio = predecir_objetivo(*modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO")

        for (_, rr), vd, unid, precio_prom in zip(
            observado.iterrows(), venta_directa, unidades, precio
        ):
            estructural = float(unid) * float(precio_prom)
            base = PESO_VENTA_DIRECTA * float(vd) + PESO_ESTRUCTURAL * estructural
            pred_final, diag = aplicar_ajuste_estacional(base, hist, int(rr["Periodo"]))
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": rr.get("DESCRIPCION", ""),
                "CATEGORIA": rr.get("CATEGORIA", "SIN_DATO"),
                "COMBOS": int(rr.get("COMBOS", 0)),
                "Periodo": int(rr["Periodo"]),
                "Actual": float(rr["VENTA"]),
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
                "Prediccion_final": float(pred_final),
            })
    return pd.DataFrame(filas)


def tabla_metricas_por_mes(detalle):
    filas = []
    for periodo, grupo in detalle.groupby("Periodo"):
        m = calcular_metricas(grupo["Actual"], grupo["Prediccion_final"])
        filas.append({"Periodo": int(periodo), "SKU": grupo["CODIGO"].nunique(), **m})
    return pd.DataFrame(filas).sort_values("Periodo") if filas else pd.DataFrame()


def tabla_metricas_por_categoria(detalle):
    filas = []
    for categoria, grupo in detalle.groupby("CATEGORIA", dropna=False):
        m = calcular_metricas(grupo["Actual"], grupo["Prediccion_final"])
        filas.append({"CATEGORIA": str(categoria), "SKU": grupo["CODIGO"].nunique(), **m})
    return pd.DataFrame(filas).sort_values("REAL", ascending=False) if filas else pd.DataFrame()


@st.cache_data(show_spinner=False)
def generar_forecast(df, version_modelo="piso-parcial-mensual-v3"):
    modelos = entrenar_modelos(df)
    filas = []
    periodos = [202611, 202612]

    for codigo, historial in df.groupby("CODIGO", sort=False):
        hist = historial.sort_values("Periodo")
        if hist.empty:
            continue
        # Solo pronosticar SKU activos con información reciente de 2026.
        # Evita saltar directamente de 2024/2025 a noviembre de 2026.
        ultimo_periodo = int(hist["Periodo"].max())
        if ultimo_periodo < 202608:
            continue
        meta = hist.iloc[-1]

        venta_directa = predecir_objetivo(
            *modelos["VENTA"], hist, periodos, "VENTA"
        )
        unidades = predecir_objetivo(
            *modelos["UNIDADES"], hist, periodos, "UNIDADES"
        )
        precio = predecir_objetivo(
            *modelos["PRECIO_PROMEDIO"], hist, periodos, "PRECIO_PROMEDIO"
        )

        for periodo, vd, unid, precio_prom in zip(
            periodos, venta_directa, unidades, precio
        ):
            estructural = unid * precio_prom
            base = (
                PESO_VENTA_DIRECTA * vd
                + PESO_ESTRUCTURAL * estructural
            )
            final, diag = aplicar_ajuste_estacional(base, hist, periodo)
            filas.append({
                "CODIGO": codigo,
                "DESCRIPCION": str(meta.get("DESCRIPCION", "")),
                "CATEGORIA": str(meta.get("CATEGORIA", "SIN_DATO")).strip(),
                "COMBOS": int(meta.get("COMBOS", 0)),
                "Periodo": periodo,
                "Venta_directa": float(vd),
                "Unidades_pred": float(unid),
                "Precio_pred": float(precio_prom),
                "Venta_estructural": float(estructural),
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


def preparar_grafica(df_historico, forecast, categoria, combo, producto):
    """Prepara histórico comparable y una línea forecast conectada a octubre 2026."""
    futuro = forecast.copy()

    if categoria != "TODAS":
        futuro = futuro[futuro["CATEGORIA"] == categoria]
    if combo != "TODOS":
        valor_combo = 1 if combo == "COMBOS" else 0
        futuro = futuro[futuro["COMBOS"] == valor_combo]
    if producto != "TODOS":
        futuro = futuro[futuro["CODIGO"].astype(str) == producto]

    # El histórico se restringe al mismo universo de SKU que sí tienen forecast.
    codigos_activos = futuro["CODIGO"].astype(str).unique().tolist()
    hist = df_historico[df_historico["CODIGO"].astype(str).isin(codigos_activos)].copy()
    if hist.empty or futuro.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    historico = hist.groupby(["AÑO", "MES_NUM"], as_index=False)["VENTA"].sum()
    historico["Fecha"] = pd.to_datetime(dict(
        year=historico["AÑO"].astype(int),
        month=historico["MES_NUM"].astype(int), day=1
    ))
    historico = historico[historico["Fecha"] <= pd.Timestamp("2026-10-01")]
    historico = historico.rename(columns={"VENTA": "Valor"})
    historico["Serie"] = "Venta real"

    previsto = futuro.groupby("Periodo", as_index=False)["Prediccion_final"].sum()
    previsto["Fecha"] = pd.to_datetime(dict(
        year=(previsto["Periodo"] // 100).astype(int),
        month=(previsto["Periodo"] % 100).astype(int), day=1
    ))
    previsto = previsto.rename(columns={"Prediccion_final": "Valor"})
    previsto["Serie"] = "Forecast"

    # Punto puente: octubre real se agrega al forecast para conectar las líneas.
    octubre = historico[historico["Fecha"] == pd.Timestamp("2026-10-01")][["Fecha", "Valor"]].copy()
    if octubre.empty:
        ultima = historico.sort_values("Fecha").tail(1)[["Fecha", "Valor"]].copy()
        puente = ultima
    else:
        puente = octubre
    puente["Serie"] = "Forecast"
    forecast_conectado = pd.concat(
        [puente[["Fecha", "Valor", "Serie"]], previsto[["Fecha", "Valor", "Serie"]]],
        ignore_index=True,
    ).sort_values("Fecha")

    return (
        historico[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
        forecast_conectado,
        previsto[["Fecha", "Valor", "Serie"]].sort_values("Fecha"),
    )


archivos = st.file_uploader(
    "Carga los archivos de ventas 2024, 2025 y 2026",
    type=["xlsx"],
    accept_multiple_files=True,
)

if archivos:
    try:
        # La carga no usa caché: cada ejecución procesa exactamente los archivos visibles.
        df, diagnostico_carga = cargar_archivos(archivos)
        resumen_anual = (
            df.groupby("AÑO", as_index=False)
            .agg(SKU=("CODIGO", "nunique"), Registros=("Periodo", "size"),
                 Desde=("Periodo", "min"), Hasta=("Periodo", "max"))
        )
        anos_cargados = set(resumen_anual["AÑO"].astype(int).tolist())
        st.success("Los archivos fueron leídos nuevamente sin utilizar caché.")
        st.dataframe(resumen_anual, use_container_width=True, hide_index=True)
        with st.expander("Diagnóstico detallado de cada hoja"):
            st.dataframe(diagnostico_carga, use_container_width=True, hide_index=True)
        faltantes = sorted({2024, 2025, 2026} - anos_cargados)
        if faltantes:
            st.error(
                f"No se cargaron datos válidos de {faltantes}. Revisa en el diagnóstico "
                "qué columna falta en las hojas correspondientes."
            )
            st.stop()

        st.markdown("### Validación del piso parcial diferenciado por mes")
        col_limpiar, col_version = st.columns([1, 2])
        with col_limpiar:
            if st.button("Limpiar resultados anteriores", type="secondary"):
                for clave in [
                    "detalle_backtest_suavizado",
                    "detalle_backtest_piso_mensual_v2",
                    "detalle_backtest_piso_mensual_v3",
                    "forecast_2026",
                ]:
                    st.session_state.pop(clave, None)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Resultados anteriores y caché eliminados.")
                st.rerun()
        with col_version:
            st.info("Configuración activa: noviembre 45% | diciembre 60% | versión v3")
        excluir_combos_bt = st.checkbox(
            "Excluir combos del backtest", value=True, key="excluir_combos_backtest"
        )
        if st.button("Calcular métricas 45%/60% Nov-Dic 2025", type="primary"):
            progreso_bt = st.progress(0, text="Entrenando únicamente hasta octubre de 2025...")
            try:
                detalle_bt = backtest_suavizado_nov_dic_2025(df)
                st.session_state["detalle_backtest_piso_mensual_v3"] = detalle_bt
                progreso_bt.progress(1.0, text="Backtest completado")
                st.rerun()
            except Exception as error_bt:
                progreso_bt.empty()
                st.exception(error_bt)

        if "detalle_backtest_piso_mensual_v3" in st.session_state:
            detalle_bt = st.session_state["detalle_backtest_piso_mensual_v3"].copy()
            evaluado_bt = (
                detalle_bt[detalle_bt["COMBOS"] == 0].copy()
                if excluir_combos_bt else detalle_bt.copy()
            )
            if evaluado_bt.empty:
                st.warning("No quedaron registros para evaluar con el filtro seleccionado.")
            else:
                comprobacion = (
                    evaluado_bt.groupby("Periodo", as_index=False)
                    .agg(
                        Recuperacion=("Porcentaje_recuperacion", "max"),
                        Correccion_total=("Correccion_piso", "sum"),
                        Prediccion_final=("Prediccion_final", "sum"),
                    )
                )
                st.markdown("#### Comprobación de la configuración aplicada")
                st.dataframe(
                    comprobacion.style.format({
                        "Recuperacion": "{:.0%}",
                        "Correccion_total": "${:,.2f}",
                        "Prediccion_final": "${:,.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                recuperaciones = dict(zip(
                    comprobacion["Periodo"].astype(int),
                    comprobacion["Recuperacion"].astype(float),
                ))
                if not (
                    abs(recuperaciones.get(202511, -1) - 0.45) < 1e-9
                    and abs(recuperaciones.get(202512, -1) - 0.60) < 1e-9
                ):
                    st.error(
                        "La configuración activa no corresponde a 45%/60%. "
                        "Pulsa Limpiar resultados anteriores y vuelve a ejecutar."
                    )
                    st.stop()

                metricas_bt = calcular_metricas(
                    evaluado_bt["Actual"], evaluado_bt["Prediccion_final"]
                )
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("WAPE", f"{metricas_bt['WAPE']:.2f}%")
                m2.metric("MAE", f"${metricas_bt['MAE']:,.2f}")
                m3.metric("RMSE", f"${metricas_bt['RMSE']:,.2f}")
                m4.metric("R²", f"{metricas_bt['R2']:.4f}" if pd.notna(metricas_bt['R2']) else "N/D")
                m5.metric("Sesgo", f"{metricas_bt['SESGO']:+.2f}%")
                st.caption(
                    f"Venta real: ${metricas_bt['REAL']:,.2f} | "
                    f"Pronóstico con piso mensual: ${metricas_bt['PRED']:,.2f} | "
                    f"{evaluado_bt['CODIGO'].nunique():,} SKU"
                )

                tab_mes, tab_categoria, tab_detalle = st.tabs([
                    "Por mes", "Por categoría", "Detalle SKU"
                ])
                formato = {
                    "WAPE": "{:.2f}%", "MAE": "${:,.2f}",
                    "RMSE": "${:,.2f}", "R2": "{:.4f}",
                    "SESGO": "{:+.2f}%", "REAL": "${:,.2f}",
                    "PRED": "${:,.2f}",
                }
                with tab_mes:
                    por_mes = tabla_metricas_por_mes(evaluado_bt)
                    st.dataframe(
                        por_mes.style.format(formato),
                        use_container_width=True, hide_index=True,
                    )
                with tab_categoria:
                    por_categoria = tabla_metricas_por_categoria(evaluado_bt)
                    st.dataframe(
                        por_categoria.style.format(formato),
                        use_container_width=True, hide_index=True,
                    )
                with tab_detalle:
                    st.dataframe(evaluado_bt, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Descargar backtest suavizado CSV",
                        evaluado_bt.to_csv(index=False).encode("utf-8-sig"),
                        "backtest_suavizado_nov_dic_2025.csv",
                        "text/csv",
                    )

        st.divider()
        if "forecast_2026" not in st.session_state:
            st.info("Genera el forecast para habilitar la gráfica de noviembre-diciembre de 2026.")

        if st.button("Generar forecast 2026 con piso 45%/60%", type="primary"):
            with st.spinner("Entrenando el modelo y generando el forecast..."):
                st.session_state["forecast_2026"] = generar_forecast(df)
            st.rerun()

        if "forecast_2026" in st.session_state:
            forecast = st.session_state["forecast_2026"]
            if forecast.empty:
                st.error("No se encontraron SKU activos con datos recientes de 2026.")
                st.stop()

            st.markdown("### Filtros")
            categorias = ["TODAS"] + sorted(
                forecast["CATEGORIA"].dropna().astype(str).str.strip().unique().tolist()
            )
            c1, c2, c3 = st.columns([2, 1.25, 3])
            with c1:
                categoria = st.selectbox("Categoría", categorias)
            with c2:
                combo = st.segmented_control(
                    "Tipo de producto",
                    options=["TODOS", "SIN COMBOS", "COMBOS"],
                    default="SIN COMBOS",
                )
                if combo is None:
                    combo = "SIN COMBOS"

            universo = forecast.copy()
            if categoria != "TODAS":
                universo = universo[universo["CATEGORIA"] == categoria]
            if combo != "TODOS":
                universo = universo[
                    universo["COMBOS"] == (1 if combo == "COMBOS" else 0)
                ]

            productos = universo[["CODIGO", "DESCRIPCION"]].drop_duplicates()
            productos["ETIQUETA"] = (
                productos["CODIGO"].astype(str) + " | "
                + productos["DESCRIPCION"].astype(str)
            )
            mapa_productos = dict(
                zip(productos["ETIQUETA"], productos["CODIGO"].astype(str))
            )
            with c3:
                seleccion_producto = st.selectbox(
                    "Producto", ["TODOS"] + sorted(mapa_productos)
                )
            producto = "TODOS" if seleccion_producto == "TODOS" else mapa_productos[seleccion_producto]

            historico, forecast_linea, forecast_puntos = preparar_grafica(
                df, forecast, categoria, combo, producto
            )
            if historico.empty or forecast_puntos.empty:
                st.warning("No hay datos comparables para la selección actual.")
                st.stop()

            nov = forecast_puntos.loc[
                forecast_puntos["Fecha"] == pd.Timestamp("2026-11-01"), "Valor"
            ].sum()
            dic = forecast_puntos.loc[
                forecast_puntos["Fecha"] == pd.Timestamp("2026-12-01"), "Valor"
            ].sum()
            oct_real = historico.loc[
                historico["Fecha"] == pd.Timestamp("2026-10-01"), "Valor"
            ].sum()
            crecimiento_nov = (nov / oct_real - 1) * 100 if oct_real > 0 else np.nan
            crecimiento_dic = (dic / nov - 1) * 100 if nov > 0 else np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Octubre 2026 real", f"${oct_real:,.0f}")
            k2.metric("Noviembre 2026", f"${nov:,.0f}",
                      f"{crecimiento_nov:+.1f}% vs octubre" if pd.notna(crecimiento_nov) else None)
            k3.metric("Diciembre 2026", f"${dic:,.0f}",
                      f"{crecimiento_dic:+.1f}% vs noviembre" if pd.notna(crecimiento_dic) else None)
            k4.metric("SKU incluidos", f"{universo['CODIGO'].nunique():,}")

            fecha_min = pd.Timestamp("2024-01-01")
            fecha_max = pd.Timestamp("2026-12-01")
            eje_x = alt.X(
                "Fecha:T", title=None,
                scale=alt.Scale(domain=[fecha_min, fecha_max]),
                axis=alt.Axis(
                    format="%b %Y", labelAngle=-35,
                    tickCount={"interval": "month", "step": 2},
                    grid=False, labelPadding=10,
                ),
            )
            eje_y = alt.Y(
                "Valor:Q", title="Venta mensual",
                scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(format="$,.2s", grid=True, gridOpacity=0.14),
            )
            tooltip = [
                alt.Tooltip("Fecha:T", title="Mes", format="%B %Y"),
                alt.Tooltip("Valor:Q", title="Venta", format="$,.2f"),
            ]

            linea_real = alt.Chart(historico).mark_line(
                color="#3B82F6", strokeWidth=3.2, interpolate="monotone"
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
            puntos_reales = alt.Chart(historico).mark_circle(
                color="#3B82F6", size=44, opacity=0.9
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)

            linea_forecast = alt.Chart(forecast_linea).mark_line(
                color="#F59E0B", strokeWidth=3.5,
                strokeDash=[8, 5], interpolate="monotone"
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)
            puntos_forecast = alt.Chart(forecast_puntos).mark_circle(
                color="#F59E0B", size=125, stroke="white", strokeWidth=1.5
            ).encode(x=eje_x, y=eje_y, tooltip=tooltip)

            limite = pd.DataFrame({"Fecha": [pd.Timestamp("2026-11-01")]})
            regla = alt.Chart(limite).mark_rule(
                color="#94A3B8", strokeDash=[4, 4], opacity=0.75
            ).encode(x="Fecha:T")
            etiqueta = alt.Chart(pd.DataFrame({
                "Fecha": [pd.Timestamp("2026-11-01")],
                "Texto": ["Inicio forecast"]
            })).mark_text(
                align="left", dx=7, dy=-205,
                color="#94A3B8", fontSize=12
            ).encode(x="Fecha:T", text="Texto:N")

            chart = alt.layer(
                linea_real, puntos_reales,
                linea_forecast, puntos_forecast,
                regla, etiqueta,
            ).properties(
                height=520,
                title=alt.TitleParams(
                    text="Evolución mensual y forecast Nov-Dic 2026",
                    subtitle="Línea azul: ventas reales | Línea naranja punteada: pronóstico",
                    anchor="start", fontSize=18, subtitleFontSize=12,
                    offset=16,
                ),
            ).configure_view(strokeWidth=0).configure_axis(
                labelFontSize=12, titleFontSize=13
            )

            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "El histórico utiliza exactamente los mismos SKU activos incluidos en el forecast. "
                "Los productos sin actividad reciente en 2026 se excluyen para evitar saltos artificiales."
            )
            with st.expander("Diagnóstico del ajuste estacional"):
                diagnostico_visible = forecast.copy()
                if categoria != "TODAS":
                    diagnostico_visible = diagnostico_visible[diagnostico_visible["CATEGORIA"] == categoria]
                if combo != "TODOS":
                    diagnostico_visible = diagnostico_visible[
                        diagnostico_visible["COMBOS"] == (1 if combo == "COMBOS" else 0)
                    ]
                if producto != "TODOS":
                    diagnostico_visible = diagnostico_visible[
                        diagnostico_visible["CODIGO"].astype(str) == producto
                    ]
                columnas_diag = [
                    "CODIGO", "DESCRIPCION", "CATEGORIA", "Periodo",
                    "Prediccion_base", "Referencia_ultimo_ano", "Crecimiento_YTD",
                    "Referencia_YTD", "Prediccion_estacional", "Referencia_suavizada",
                    "Prediccion_combinada", "Piso_parcial",
                    "Porcentaje_recuperacion", "Correccion_piso",
                    "Indice_estacional", "Ajuste_aplicado", "Prediccion_final",
                ]
                st.dataframe(
                    diagnostico_visible[columnas_diag],
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Descargar diagnóstico CSV",
                    diagnostico_visible[columnas_diag].to_csv(index=False).encode("utf-8-sig"),
                    "diagnostico_forecast_nov_dic_2026.csv",
                    "text/csv",
                )

    except Exception as error:
        st.exception(error)
else:
    st.info("Carga los tres archivos Excel para comenzar.")
