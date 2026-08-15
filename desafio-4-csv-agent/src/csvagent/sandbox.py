"""Execução restrita de código Python gerado pelo LLM.

O agente escreve pandas/plotly "de verdade" em vez de responder de memória --
é isso que garante que a resposta reflita os dados carregados. Mas código
gerado por um LLM não deve rodar livre: este módulo valida a árvore sintática
(AST) ANTES de executar, bloqueando import, acesso a atributos dunder, I/O de
arquivo, rede, e chamadas a builtins perigosos (eval/exec/open/...).

Não é um sandbox de produção (não há isolamento de processo/SO) -- para o
escopo do desafio, a defesa em profundidade é: (1) whitelist de AST, (2)
builtins restritos, (3) só variáveis com nomes conhecidos ficam expostas ao
código (as tabelas + pd/px/go), (4) sem loops `while` para limitar loops
infinitos triviais. Ver README para essa ressalva.
"""

from __future__ import annotations

import ast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "exit",
    "quit",
    "help",
    "breakpoint",
}

_ALLOWED_BUILTINS = {
    "len",
    "range",
    "sum",
    "min",
    "max",
    "sorted",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "round",
    "abs",
    "enumerate",
    "zip",
    "map",
    "filter",
    "any",
    "all",
    "print",
    "isinstance",
    "type",
}


class SandboxViolation(Exception):
    """Código recusado pela validação estática antes mesmo de tentar executar."""


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxViolation("Import não é permitido dentro do código gerado.")
        if isinstance(node, (ast.While,)):
            raise SandboxViolation("Loops `while` não são permitidos (risco de loop infinito).")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raise SandboxViolation("Definição de função/classe não é permitida.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SandboxViolation("global/nonlocal não são permitidos.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxViolation(f"Acesso a atributo '{node.attr}' não é permitido.")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise SandboxViolation(f"Uso de '{node.id}' não é permitido.")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
                raise SandboxViolation(f"Chamada a '{func.id}' não é permitida.")


def _safe_builtins() -> dict:
    import builtins

    return {name: getattr(builtins, name) for name in _ALLOWED_BUILTINS}


def run_pandas_snippet(code: str, tables: dict[str, pd.DataFrame], result_var: str = "resultado"):
    """Executa código pandas restrito. O código deve atribuir a `resultado`."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"Código com erro de sintaxe: {exc}") from exc
    _validate_ast(tree)

    sandbox_globals = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "tables": {name: df.copy() for name, df in tables.items()},
    }
    sandbox_locals: dict = {}
    exec(compile(tree, filename="<agent_code>", mode="exec"), sandbox_globals, sandbox_locals)  # noqa: S102

    if result_var not in sandbox_locals:
        raise SandboxViolation(
            f"O código precisa atribuir o resultado final à variável `{result_var}`."
        )
    return sandbox_locals[result_var]


def run_chart_snippet(code: str, tables: dict[str, pd.DataFrame], fig_var: str = "fig"):
    """Executa código plotly restrito. O código deve atribuir a `fig` (figura Plotly)."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"Código com erro de sintaxe: {exc}") from exc
    _validate_ast(tree)

    sandbox_globals = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "px": px,
        "go": go,
        "tables": {name: df.copy() for name, df in tables.items()},
    }
    sandbox_locals: dict = {}
    exec(compile(tree, filename="<agent_chart_code>", mode="exec"), sandbox_globals, sandbox_locals)  # noqa: S102

    if fig_var not in sandbox_locals:
        raise SandboxViolation(
            f"O código precisa atribuir a figura Plotly à variável `{fig_var}`."
        )
    fig = sandbox_locals[fig_var]
    if not isinstance(fig, go.Figure):
        raise SandboxViolation("A variável `fig` precisa ser uma figura Plotly (px.* ou go.Figure).")
    return fig
