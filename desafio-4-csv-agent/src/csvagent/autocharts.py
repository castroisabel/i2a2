"""Gráficos genéricos sugeridos por heurística sobre o schema (sem chamar o LLM).

Rodam automaticamente ao carregar os dados na Interface A: olham os tipos das
colunas (numérica, categórica de baixa cardinalidade, data) e, se encontrarem
uma combinação útil, montam até 2 gráficos -- "total por categoria" e
"evolução no tempo". Não força um gráfico quando os dados não têm uma
combinação óbvia (ex: só colunas de texto livre, ou só IDs).
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_ID_LIKE = re.compile(r"(^id$|^id_|_id$|codigo|código)", re.IGNORECASE)
_MIN_CATEGORIES, _MAX_CATEGORIES = 2, 20


def _pick_value_column(df: pd.DataFrame) -> str | None:
    """Primeira coluna numérica que não parece ser um identificador pelo nome."""
    for col in df.select_dtypes(include="number").columns:
        if _ID_LIKE.search(str(col)):
            continue
        return col
    return None


def _pick_category_column(df: pd.DataFrame) -> str | None:
    """Primeira coluna categórica com cardinalidade baixa o suficiente para virar barra."""
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        n_unique = df[col].nunique(dropna=True)
        if _MIN_CATEGORIES <= n_unique <= _MAX_CATEGORIES:
            return col
    return None


def _pick_date_column(df: pd.DataFrame) -> pd.Series | None:
    """Retorna a coluna de data já convertida (datetime), se achar uma com >=50% de linhas válidas."""
    for col in df.select_dtypes(include="datetime").columns:
        return df[col]
    for col in df.columns:
        if not re.search(r"data|date", str(col), re.IGNORECASE):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() >= max(3, 0.5 * len(df)):
            return parsed
    return None


def suggest_default_charts(df: pd.DataFrame) -> list[tuple[str, go.Figure]]:
    """Até 2 (título, figura Plotly) genéricos: total por categoria e evolução no tempo."""
    charts: list[tuple[str, go.Figure]] = []
    value_col = _pick_value_column(df)
    if value_col is None:
        return charts

    category_col = _pick_category_column(df)
    if category_col is not None:
        agg = (
            df.groupby(category_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        title = f"{value_col} total por {category_col}"
        charts.append((title, px.bar(agg, x=category_col, y=value_col, title=title)))

    date_series = _pick_date_column(df)
    if date_series is not None:
        valid = date_series.notna()
        tmp = pd.DataFrame({"_date": date_series[valid], value_col: df.loc[valid, value_col]})
        span_days = (tmp["_date"].max() - tmp["_date"].min()).days
        freq = "D" if span_days <= 31 else "MS"
        grouped = tmp.set_index("_date").resample(freq)[value_col].sum().reset_index()
        title = f"Evolução de {value_col} ao longo do tempo"
        fig = px.line(
            grouped, x="_date", y=value_col, title=title, markers=True, labels={"_date": "Data"}
        )
        charts.append((title, fig))

    return charts[:2]


def suggest_example_questions(df: pd.DataFrame, table_name: str | None = None) -> list[str]:
    """Perguntas de exemplo derivadas do schema real (mesma heurística dos gráficos
    automáticos) -- substituem uma lista fixa que não tinha relação com o dado carregado.
    `table_name` só é usado (como prefixo) quando o catálogo tem mais de uma tabela."""
    value_col = _pick_value_column(df)
    category_col = _pick_category_column(df)
    date_series = _pick_date_column(df)

    templates: list[str] = []
    if value_col and category_col:
        templates.append(f"Qual {category_col} teve o maior total de {value_col}?")
        templates.append(f"Quais os 5 maiores {category_col} em {value_col}?")
    if value_col and date_series is not None:
        templates.append(f"Qual foi o total de {value_col} em cada mês?")
    if category_col:
        templates.append(f"Quais os valores mais frequentes em {category_col}?")
    if value_col and not templates:
        templates.append(f"Qual o total de {value_col}?")
    if not templates:
        templates.append("Quantas linhas tem esta tabela?")

    if table_name is None:
        return templates
    return [f"Na tabela `{table_name}`, " + t[0].lower() + t[1:] for t in templates]
