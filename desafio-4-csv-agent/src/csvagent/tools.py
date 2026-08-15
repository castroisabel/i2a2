"""Tools que o agente LangChain pode chamar para responder perguntas sobre os CSVs.

Cada tool é deliberadamente pequena e de responsabilidade única -- isso é o que
permite ao agente compor uma investigação (ex: "descrever tabela" -> "ver
valores frequentes de uma coluna" -> "rodar pandas para agregar" -> "gerar
gráfico") e é o que fica registrado no trace exibido na Interface B para
explicar como o agente decidiu chegar na resposta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from langchain_core.tools import tool

from csvagent.catalog import DataCatalog, TableNotFoundError
from csvagent.sandbox import SandboxViolation, run_chart_snippet, run_pandas_snippet

_MAX_ROWS_IN_OBSERVATION = 50


@dataclass
class AgentSession:
    """Guarda os efeitos colaterais de uma rodada de perguntas para a UI renderizar
    (tabelas e gráficos gerados) e o trace de chamadas de tool para transparência."""

    catalog: DataCatalog
    generated_tables: list[pd.DataFrame] = field(default_factory=list)
    generated_charts: list[go.Figure] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def reset_round(self) -> None:
        self.generated_tables.clear()
        self.generated_charts.clear()
        self.trace.clear()

    def log(self, tool_name: str, tool_input: Any, output_preview: str) -> None:
        self.trace.append(
            {"tool": tool_name, "input": tool_input, "output_preview": output_preview[:800]}
        )


def build_tools(session: AgentSession) -> list:
    catalog = session.catalog

    @tool
    def listar_tabelas() -> str:
        """Lista todas as tabelas (arquivos CSV) disponíveis para consulta, com
        número de linhas e colunas. Use esta ferramenta primeiro se não souber
        o nome exato da tabela que precisa consultar."""
        lines = []
        for name, df in catalog.tables.items():
            lines.append(f"- {name}: {len(df)} linhas, colunas = {list(df.columns)}")
        output = "\n".join(lines) if lines else "Nenhuma tabela carregada."
        session.log("listar_tabelas", {}, output)
        return output

    @tool
    def descrever_tabela(nome_tabela: str) -> str:
        """Mostra o schema detalhado de uma tabela: nome e tipo de cada coluna,
        descrição vinda do dicionário de dados (quando disponível), estatísticas
        básicas de colunas numéricas e uma amostra de linhas. Use antes de montar
        uma consulta pandas para confirmar os nomes exatos das colunas."""
        try:
            df = catalog.get(nome_tabela)
        except TableNotFoundError as exc:
            session.log("descrever_tabela", nome_tabela, str(exc))
            return str(exc)

        table_dict = catalog.dictionary.get(nome_tabela, {})
        col_lines = []
        for col in df.columns:
            desc = table_dict.get(col, "")
            desc_part = f" -- {desc}" if desc else ""
            col_lines.append(f"  - {col} ({df[col].dtype}){desc_part}")

        numeric_stats = ""
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            numeric_stats = "\nEstatísticas (colunas numéricas):\n" + numeric_df.describe().to_markdown()

        sample = df.head(5).to_markdown(index=False)
        output = (
            f"Tabela `{nome_tabela}` -- {len(df)} linhas\n"
            + "\n".join(col_lines)
            + numeric_stats
            + f"\n\nAmostra de linhas:\n{sample}"
        )
        session.log("descrever_tabela", nome_tabela, output)
        return output

    @tool
    def valores_frequentes(nome_tabela: str, coluna: str, top_n: int = 10) -> str:
        """Retorna os valores mais frequentes de uma coluna categórica de uma tabela,
        com a contagem de ocorrências de cada um. Útil para entender que valores
        existem numa coluna (ex: categorias, fornecedores, estados) antes de filtrar
        ou agrupar por ela."""
        try:
            df = catalog.get(nome_tabela)
        except TableNotFoundError as exc:
            session.log("valores_frequentes", {"nome_tabela": nome_tabela}, str(exc))
            return str(exc)
        if coluna not in df.columns:
            output = f"Coluna '{coluna}' não existe em '{nome_tabela}'. Colunas disponíveis: {list(df.columns)}"
            session.log("valores_frequentes", {"nome_tabela": nome_tabela, "coluna": coluna}, output)
            return output

        counts = df[coluna].value_counts(dropna=False).head(top_n)
        output = counts.to_markdown()
        session.log(
            "valores_frequentes", {"nome_tabela": nome_tabela, "coluna": coluna, "top_n": top_n}, output
        )
        return output

    @tool
    def executar_pandas(codigo: str) -> str:
        """Executa uma consulta pandas sobre as tabelas carregadas e retorna o
        resultado real (não invente números). As tabelas disponíveis estão no
        dicionário `tables`, acessadas por `tables["nome_da_tabela"]` -- use
        exatamente os nomes retornados por `listar_tabelas`. O código DEVE
        atribuir o resultado final (DataFrame, Series ou valor escalar) à
        variável `resultado`. Não use import, funções, classes nem loops `while`.
        Exemplo: resultado = tables["notas"].groupby("fornecedor")["valor"].sum().sort_values(ascending=False).head(5)
        """
        try:
            result = run_pandas_snippet(codigo, catalog.tables)
        except (SandboxViolation, TableNotFoundError, KeyError, Exception) as exc:  # noqa: BLE001
            error_msg = f"Erro ao executar o código: {exc}"
            session.log("executar_pandas", codigo, error_msg)
            return error_msg

        if isinstance(result, pd.DataFrame):
            session.generated_tables.append(result)
            truncated = result.head(_MAX_ROWS_IN_OBSERVATION)
            note = "" if len(result) <= _MAX_ROWS_IN_OBSERVATION else f"\n(mostrando {_MAX_ROWS_IN_OBSERVATION} de {len(result)} linhas)"
            output = truncated.to_markdown() + note
        elif isinstance(result, pd.Series):
            session.generated_tables.append(result.to_frame())
            output = result.head(_MAX_ROWS_IN_OBSERVATION).to_markdown()
        else:
            output = str(result)

        session.log("executar_pandas", codigo, output)
        return output

    @tool
    def gerar_grafico(codigo: str) -> str:
        """Gera um gráfico Plotly a partir das tabelas carregadas. As tabelas estão
        em `tables["nome_da_tabela"]`, e as bibliotecas `px` (plotly.express) e `go`
        (plotly.graph_objects) já estão disponíveis. O código DEVE atribuir a figura
        final à variável `fig`. Use isso quando o usuário pedir explicitamente um
        gráfico, ou quando um gráfico ilustrar melhor a resposta do que só texto.
        Exemplo: fig = px.bar(tables["notas"].groupby("mes")["valor"].sum().reset_index(), x="mes", y="valor")
        """
        try:
            fig = run_chart_snippet(codigo, catalog.tables)
        except (SandboxViolation, TableNotFoundError, KeyError, Exception) as exc:  # noqa: BLE001
            error_msg = f"Erro ao gerar o gráfico: {exc}"
            session.log("gerar_grafico", codigo, error_msg)
            return error_msg

        session.generated_charts.append(fig)
        output = "Gráfico gerado com sucesso e será exibido para o usuário."
        session.log("gerar_grafico", codigo, output)
        return output

    return [listar_tabelas, descrever_tabela, valores_frequentes, executar_pandas, gerar_grafico]
