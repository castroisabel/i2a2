"""Interface A: descompacta o ZIP enviado pelo usuário e carrega os CSVs.

Responsabilidades:
1. Extrair o .zip para uma pasta temporária, ignorando lixo de SO (__MACOSX, .DS_Store).
2. Para cada arquivo tabular (.csv/.txt), detectar encoding e separador automaticamente
   -- os datasets de exemplo do curso (notas fiscais) costumam vir em latin-1 com ';'.
3. Distinguir arquivos de DADOS de um eventual arquivo de DICIONÁRIO DE DADOS
   (heurística por nome do arquivo e, na falta disso, pelo formato do cabeçalho).
4. Devolver um DataCatalog pronto para o agente consultar.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import chardet
import pandas as pd

from csvagent.catalog import DataCatalog

# Palavras que indicam "isto é o dicionário de dados", não uma tabela de fatos.
_DICTIONARY_NAME_HINTS = (
    "dicionario",
    "dicionário",
    "dictionary",
    "dic_",
    "_dic",
    "layout",
    "metadado",
    "metadata",
    "glossario",
    "glossário",
)

# Cabeçalhos típicos de um dicionário de dados (coluna -> descrição).
_DICTIONARY_HEADER_HINTS = (
    "coluna",
    "campo",
    "descricao",
    "descrição",
    "description",
    "column",
    "field",
    "tipo_dado",
    "data_type",
)

_CANDIDATE_SEPARATORS = [",", ";", "\t", "|"]
_IGNORED_DIR_NAMES = {"__MACOSX"}


@dataclass
class RawFile:
    name: str
    raw_bytes: bytes


def _iter_zip_members(zip_path: str | Path) -> list[RawFile]:
    """Lê todos os arquivos tabulares (csv/txt) de dentro do zip, ignorando lixo de SO."""
    files: list[RawFile] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename)
            if any(part in _IGNORED_DIR_NAMES or part.startswith(".") for part in path.parts):
                continue
            if path.suffix.lower() not in (".csv", ".txt"):
                continue
            files.append(RawFile(name=path.name, raw_bytes=zf.read(info)))
    return files


def _detect_encoding(raw: bytes) -> str:
    guess = chardet.detect(raw[:200_000])
    encoding = (guess.get("encoding") or "utf-8").lower()
    # chardet às vezes acerta ascii onde na verdade é latin-1 com acentos raros no fim do arquivo.
    if encoding in ("ascii", "iso-8859-1"):
        encoding = "latin-1"
    return encoding


def _detect_separator(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters="".join(_CANDIDATE_SEPARATORS))
        return dialect.delimiter
    except csv.Error:
        pass
    # Fallback manual: separador que produz mais de uma coluna consistente na 1ª linha.
    first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    best_sep, best_count = ",", 1
    for sep in _CANDIDATE_SEPARATORS:
        count = first_line.count(sep)
        if count > best_count:
            best_sep, best_count = sep, count
    return best_sep


def _looks_like_dictionary(name: str, df: pd.DataFrame) -> bool:
    lowered_name = name.lower()
    if any(hint in lowered_name for hint in _DICTIONARY_NAME_HINTS):
        return True
    # Poucas colunas + cabeçalho com cara de "coluna/descrição" -> provavelmente é o dicionário.
    lowered_cols = [str(c).lower() for c in df.columns]
    if len(df.columns) <= 5:
        hits = sum(1 for c in lowered_cols for hint in _DICTIONARY_HEADER_HINTS if hint in c)
        if hits >= 2:
            return True
    return False


def _read_dataframe(raw: RawFile) -> pd.DataFrame:
    encoding = _detect_encoding(raw.raw_bytes)
    text_sample = raw.raw_bytes[:20_000].decode(encoding, errors="replace")
    sep = _detect_separator(text_sample)
    return pd.read_csv(
        io.BytesIO(raw.raw_bytes),
        sep=sep,
        encoding=encoding,
        engine="python",
        on_bad_lines="warn",
    )


def _parse_dictionary(df: pd.DataFrame) -> dict[str, str]:
    """Tenta mapear coluna -> descrição a partir de um DataFrame de dicionário de dados."""
    cols_lower = {str(c).lower(): c for c in df.columns}

    def find_col(hints: tuple[str, ...]) -> str | None:
        for hint in hints:
            for lowered, original in cols_lower.items():
                if hint in lowered:
                    return original
        return None

    col_name_field = find_col(("coluna", "campo", "column", "field", "nome"))
    col_desc_field = find_col(("descricao", "descrição", "description"))
    if not col_name_field or not col_desc_field:
        return {}

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        key = str(row[col_name_field]).strip()
        val = str(row[col_desc_field]).strip()
        if key and key.lower() != "nan":
            mapping[key] = val
    return mapping


def load_zip_into_catalog(zip_path: str | Path) -> tuple[DataCatalog, list[str]]:
    """Extrai o zip e monta o DataCatalog. Retorna (catálogo, avisos)."""
    raw_files = _iter_zip_members(zip_path)
    if not raw_files:
        raise ValueError(
            "Nenhum arquivo .csv ou .txt encontrado dentro do ZIP. "
            "Confira se o arquivo enviado realmente contém os dados compactados."
        )
    return _build_catalog_from_raw_files(raw_files)


def load_uploads_into_catalog(uploaded_files: list) -> tuple[DataCatalog, list[str]]:
    """Monta o catálogo a partir de um ou mais arquivos enviados pelo usuário.

    Cada item pode ser um .zip (descompactado automaticamente) ou um .csv/.txt solto,
    usado diretamente -- assim o usuário não é obrigado a compactar um único CSV.
    Espera objetos com `.name` e `.getvalue()` (ex: `st.file_uploader`).
    """
    raw_files: list[RawFile] = []
    for uploaded in uploaded_files:
        name = uploaded.name
        data = uploaded.getvalue()
        suffix = Path(name).suffix.lower()
        if suffix == ".zip":
            raw_files.extend(_iter_zip_members(io.BytesIO(data)))
        elif suffix in (".csv", ".txt"):
            raw_files.append(RawFile(name=name, raw_bytes=data))

    if not raw_files:
        raise ValueError(
            "Nenhum arquivo .csv ou .txt encontrado. Envie um .zip contendo os dados "
            "ou arquivos .csv diretamente."
        )
    return _build_catalog_from_raw_files(raw_files)


def _build_catalog_from_raw_files(raw_files: list[RawFile]) -> tuple[DataCatalog, list[str]]:
    warnings: list[str] = []
    catalog = DataCatalog()
    dictionary_candidates: list[tuple[str, pd.DataFrame]] = []
    data_candidates: list[tuple[str, pd.DataFrame]] = []

    for raw in raw_files:
        try:
            df = _read_dataframe(raw)
        except Exception as exc:  # noqa: BLE001 - queremos seguir carregando os demais arquivos
            warnings.append(f"Não foi possível ler '{raw.name}': {exc}")
            continue

        table_key = Path(raw.name).stem
        if _looks_like_dictionary(raw.name, df):
            dictionary_candidates.append((table_key, df))
        else:
            data_candidates.append((table_key, df))

    if not data_candidates:
        # Nada bateu a heurística de "dados" -- trata tudo como dados (fallback seguro).
        data_candidates, dictionary_candidates = dictionary_candidates, []
        warnings.append(
            "Nenhum arquivo teve cara de dicionário de dados; todos foram carregados como tabelas."
        )

    for table_key, df in data_candidates:
        catalog.add_table(table_key, df)

    combined_dictionary: dict[str, dict[str, str]] = {}
    for _, dict_df in dictionary_candidates:
        parsed = _parse_dictionary(dict_df)
        if not parsed:
            warnings.append(
                "Um arquivo parecia ser um dicionário de dados, mas não foi possível "
                "identificar as colunas de nome/descrição automaticamente."
            )
            continue
        # Dicionário único aplicado a todas as tabelas (cenário mais comum: um dicionário
        # descreve as colunas compartilhadas entre cabeçalho/itens da NF-e, por exemplo).
        for table_key, _ in data_candidates:
            combined_dictionary.setdefault(table_key, {}).update(parsed)

    catalog.set_dictionary(combined_dictionary)
    return catalog, warnings
