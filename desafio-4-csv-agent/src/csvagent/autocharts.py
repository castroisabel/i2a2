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


def build_chart_from_dataframe(df: pd.DataFrame, title: str = "") -> go.Figure | None:
    """Gera automaticamente uma figura Plotly a partir de um DataFrame retornado por uma consulta.
    Usa linhas com marcadores roxos para dados temporais e barras azuis para dados categóricos,
    seguindo o design system do projeto.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 2:
        return None

    try:
        # Se tiver apenas 1 coluna numérica e o índice tiver nomes úteis
        work_df = df.copy()
        if len(work_df.columns) == 1 and not isinstance(work_df.index, pd.RangeIndex):
            work_df = work_df.reset_index()

        cols = list(work_df.columns)
        if len(cols) < 2:
            return None

        # Converte qualquer coluna Period ou Timestamp em string para serialização segura em JSON
        for c in cols:
            if str(work_df[c].dtype).startswith("period") or isinstance(work_df[c].dtype, pd.PeriodDtype):
                work_df[c] = work_df[c].astype(str)
            elif work_df[c].dtype == object:
                work_df[c] = work_df[c].apply(lambda x: str(x) if isinstance(x, (pd.Period, pd.Timestamp)) else x)

        # Identifica colunas numéricas e não-numéricas
        num_cols = work_df.select_dtypes(include="number").columns.tolist()
        non_num_cols = [c for c in cols if c not in num_cols]

        if num_cols and non_num_cols:
            x_col = non_num_cols[0]
            y_col = num_cols[0]

            # Verifica se x_col parece ser temporal (data, ano, mês, etc)
            is_temporal = any(
                k in str(x_col).lower()
                for k in ["mes", "mês", "ano", "data", "date", "periodo", "período", "dia", "month", "year"]
            ) or any(
                re.match(r"^\d{4}[-/]\d{2}", str(v)) for v in work_df[x_col].dropna().head(3)
            )

            chart_title = title or f"{y_col} por {x_col}"

            if is_temporal:
                fig = px.line(
                    work_df,
                    x=x_col,
                    y=y_col,
                    title=f"📈 {chart_title}",
                    markers=True,
                )
                fig.update_traces(
                    line=dict(color="#7C3AED", width=3),
                    marker=dict(size=8, color="#7C3AED", symbol="circle")
                )
            else:
                # Ordena por valor se for categórico
                plot_df = work_df.sort_values(by=y_col, ascending=False).head(15)
                fig = px.bar(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    title=f"📊 {chart_title}",
                    color_discrete_sequence=["#3B82F6"]
                )
                fig.update_traces(marker_line_width=0, opacity=0.9)

            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode="x unified",
                font=dict(family="Inter, sans-serif")
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(226, 232, 240, 0.6)")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(226, 232, 240, 0.6)")
            return fig

        elif len(num_cols) >= 2:
            x_col, y_col = num_cols[0], num_cols[1]
            fig = px.line(
                work_df,
                x=x_col,
                y=y_col,
                title=f"📈 {title or f'{y_col} vs {x_col}'}",
                markers=True
            )
            fig.update_traces(line=dict(color="#7C3AED", width=3), marker=dict(size=8, color="#7C3AED"))
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode="x unified",
                font=dict(family="Inter, sans-serif")
            )
            return fig

    except Exception:  # noqa: BLE001
        return None

    return None

