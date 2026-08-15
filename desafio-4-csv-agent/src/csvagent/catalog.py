"""Catálogo em memória dos dados carregados: DataFrames + dicionário de dados.

É este catálogo que alimenta o prompt do agente (resumo do schema) e que as
tools consultam para executar as perguntas do usuário.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class TableNotFoundError(Exception):
    """Levantado quando o agente pede uma tabela que não existe -- mensagem já vem
    com a lista de tabelas disponíveis para o LLM se auto-corrigir na próxima chamada."""


@dataclass
class DataCatalog:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    # tabela -> {coluna: descrição}, vindo do dicionário de dados do ZIP (se houver)
    dictionary: dict[str, dict[str, str]] = field(default_factory=dict)

    def add_table(self, name: str, df: pd.DataFrame) -> None:
        self.tables[name] = df

    def set_dictionary(self, dictionary: dict[str, dict[str, str]]) -> None:
        self.dictionary = dictionary

    @property
    def is_empty(self) -> bool:
        return len(self.tables) == 0

    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    def get(self, name: str) -> pd.DataFrame:
        if name in self.tables:
            return self.tables[name]
        # tolera diferença de maiúsculas/minúsculas -- o LLM às vezes normaliza o nome
        for key in self.tables:
            if key.lower() == name.lower():
                return self.tables[key]
        available = ", ".join(self.table_names())
        raise TableNotFoundError(
            f"Tabela '{name}' não encontrada. Tabelas disponíveis: {available}"
        )

    def context_summary(self, max_sample_rows: int = 3) -> str:
        """Resumo textual do catálogo (schema + amostra + dicionário) para o system prompt."""
        parts: list[str] = []
        for name, df in self.tables.items():
            parts.append(f"### Tabela `{name}` ({len(df)} linhas, {len(df.columns)} colunas)")
            col_lines = []
            table_dict = self.dictionary.get(name, {})
            for col in df.columns:
                dtype = str(df[col].dtype)
                desc = table_dict.get(col, "")
                desc_part = f" -- {desc}" if desc else ""
                col_lines.append(f"  - {col} ({dtype}){desc_part}")
            parts.append("\n".join(col_lines))
            if max_sample_rows > 0 and len(df) > 0:
                sample = df.head(max_sample_rows).to_markdown(index=False)
                parts.append(f"Amostra:\n{sample}")
            parts.append("")
        return "\n".join(parts)
