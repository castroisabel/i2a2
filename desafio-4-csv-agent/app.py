"""Desafio 4 (I2A2) -- Interface Inteligente para Consulta de Arquivos CSV.

Duas interfaces, como pedido no enunciado:
  A. Carga dos dados  -- upload de um .zip com CSV(s) + dicionário de dados opcional
  B. Consulta         -- chat em linguagem natural sobre os dados carregados

Rodar localmente:  uv run streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from csvagent.agent import build_agent_executor
from csvagent.autocharts import suggest_default_charts, suggest_example_questions
from csvagent.catalog import DataCatalog
from csvagent.ingestion import load_uploads_into_catalog, sanitize_oversized_integers

load_dotenv()

st.set_page_config(page_title="Agente CSV -- Desafio 4 I2A2", page_icon="📊", layout="wide")


def _example_questions_for_catalog(catalog: DataCatalog) -> list[str]:
    """Perguntas de exemplo derivadas do schema real carregado (heurística, sem LLM) --
    prefixa com o nome da tabela só quando há mais de uma no catálogo."""
    multi = len(catalog.tables) > 1
    questions: list[str] = []
    for name, df in catalog.tables.items():
        questions.extend(suggest_example_questions(df, table_name=name if multi else None))
    return questions[:6]


def _init_session_state() -> None:
    defaults = {
        "catalog": None,
        "catalog_warnings": [],
        "executor": None,
        "agent_session": None,
        "chat_messages": [],  # para exibir na tela: [{"role", "content", "tables", "charts", "trace"}]
        "lc_history": [],  # para a memória do agente: [HumanMessage/AIMessage]
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _get_api_key() -> str:
    """Chave efetiva usada pelo agente nesta sessão.

    IMPORTANTE: a chave do servidor (variável de ambiente / Secret do Streamlit Cloud)
    nunca deve ser usada como `value=` de um widget -- isso a coloca no DOM da página e
    qualquer visitante do app publicado consegue revelá-la clicando no ícone de "mostrar
    senha". O campo de texto abaixo começa sempre vazio; se o usuário não digitar nada,
    caímos silenciosamente para a chave do servidor (usada só internamente, nunca reexibida).
    """
    server_key = os.environ.get("GROQ_API_KEY", "")
    with st.sidebar:
        st.header("Configuração")
        if server_key:
            st.success("Chave da API já configurada para este app.")
            label = "Usar outra Groq API Key nesta sessão -- opcional"
        else:
            label = "Groq API Key"
        user_key = st.text_input(
            label,
            type="password",
            help="Gere gratuitamente em https://console.groq.com/keys",
            key="api_key_input",
        )
        st.caption(
            "Se você digitar uma chave aqui, ela fica só nesta sessão do navegador -- não é "
            "salva em disco nem enviada a lugar nenhum além da API da Groq."
        )
        if st.session_state.catalog is not None:
            st.divider()
            st.subheader("Dados carregados")
            for name, df in st.session_state.catalog.tables.items():
                st.caption(f"`{name}` -- {len(df)} linhas x {len(df.columns)} colunas")
    return user_key or server_key


def _render_tab_upload() -> None:
    st.subheader("Interface A -- Carga dos dados")
    st.write(
        "Envie um ou mais arquivos **.csv**, ou um **.zip** contendo um ou mais CSVs e, "
        "opcionalmente, um **dicionário de dados** (arquivo descrevendo as colunas -- o app "
        "tenta detectá-lo automaticamente pelo nome ou pelo formato)."
    )
    uploaded_files = st.file_uploader(
        "Arquivo(s) CSV ou ZIP", type=["zip", "csv"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Processar arquivo(s)", type="primary"):
            with st.spinner("Lendo os arquivos..."):
                try:
                    catalog, warnings = load_uploads_into_catalog(uploaded_files)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Não foi possível processar o(s) arquivo(s): {exc}")
                    return

            st.session_state.catalog = catalog
            st.session_state.catalog_warnings = warnings
            st.session_state.executor = None  # força reconstrução do agente com o novo catálogo
            st.session_state.chat_messages = []
            st.session_state.lc_history = []
            st.success(f"{len(catalog.tables)} tabela(s) carregada(s) com sucesso!")

    catalog = st.session_state.catalog
    if catalog is None:
        st.info("Nenhum dado carregado ainda. Envie CSV(s) ou um ZIP acima para liberar a aba de Consulta.")
        return

    for warning in st.session_state.catalog_warnings:
        st.warning(warning)

    st.divider()
    st.write("**Pré-visualização das tabelas carregadas:**")
    for name, df in catalog.tables.items():
        dict_note = " (dicionário de dados aplicado)" if catalog.dictionary.get(name) else ""
        with st.expander(f"`{name}` -- {len(df)} linhas x {len(df.columns)} colunas{dict_note}"):
            st.dataframe(df.head(20), width="stretch")
            if catalog.dictionary.get(name):
                st.write("Dicionário de dados:")
                st.json(catalog.dictionary[name])

            charts = suggest_default_charts(df)
            if charts:
                st.write("**Gráficos automáticos** (sugeridos por heurística, sem usar a LLM):")
                for title, fig in charts:
                    st.plotly_chart(fig, width="stretch", key=f"autochart_{name}_{title}")


def _ensure_agent(api_key: str) -> bool:
    if not api_key:
        st.warning("Informe a Groq API Key na barra lateral para liberar o chat.")
        return False
    if st.session_state.executor is None:
        with st.spinner("Preparando o agente..."):
            executor, agent_session = build_agent_executor(st.session_state.catalog, api_key)
            st.session_state.executor = executor
            st.session_state.agent_session = agent_session
    return True


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for table in message.get("tables", []):
            st.dataframe(sanitize_oversized_integers(table), width="stretch")
        for chart in message.get("charts", []):
            st.plotly_chart(chart, width="stretch")
        trace = message.get("trace")
        if trace:
            with st.expander("Ver raciocínio do agente (tools chamadas)"):
                for step in trace:
                    st.markdown(f"**{step['tool']}**")
                    st.code(str(step["input"]), language="python")
                    st.caption(f"Resultado: {step['output_preview']}")


def _render_tab_chat() -> None:
    st.subheader("Interface B -- Consulta em linguagem natural")

    catalog = st.session_state.catalog
    if catalog is None:
        st.info("Carregue um arquivo ZIP na aba 'Carga dos Dados' primeiro.")
        return

    if not _ensure_agent(st.session_state.effective_api_key):
        return

    with st.expander("💡 Perguntas de exemplo (baseadas nos dados carregados)"):
        for question in _example_questions_for_catalog(catalog):
            st.markdown(f"- {question}")

    for message in st.session_state.chat_messages:
        _render_message(message)

    user_question = st.chat_input("Faça uma pergunta sobre os dados carregados...")
    if not user_question:
        return

    user_message = {"role": "user", "content": user_question}
    st.session_state.chat_messages.append(user_message)
    _render_message(user_message)

    agent_session = st.session_state.agent_session
    agent_session.reset_round()

    with st.chat_message("assistant"):
        with st.spinner("Consultando os dados..."):
            try:
                messages = [*st.session_state.lc_history, HumanMessage(content=user_question)]
                result = st.session_state.executor.invoke({"messages": messages})
                last_message = result["messages"][-1]
                # Alguns modelos retornam `.content` como uma lista de blocos em vez de
                # string (ex: Gemini com "thought signatures") -- `.text` normaliza para str.
                answer = str(last_message.text)
            except Exception as exc:  # noqa: BLE001
                answer = (
                    "Não consegui concluir essa consulta. Detalhe técnico: "
                    f"{exc}\n\nTente reformular a pergunta ou verificar se a coluna/tabela citada existe."
                )

        st.markdown(answer)
        for table in agent_session.generated_tables:
            st.dataframe(sanitize_oversized_integers(table), width="stretch")
        for chart in agent_session.generated_charts:
            st.plotly_chart(chart, width="stretch")
        if agent_session.trace:
            with st.expander("Ver raciocínio do agente (tools chamadas)"):
                for step in agent_session.trace:
                    st.markdown(f"**{step['tool']}**")
                    st.code(str(step["input"]), language="python")
                    st.caption(f"Resultado: {step['output_preview']}")

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "tables": list(agent_session.generated_tables),
            "charts": list(agent_session.generated_charts),
            "trace": list(agent_session.trace),
        }
    )
    st.session_state.lc_history.append(HumanMessage(content=user_question))
    st.session_state.lc_history.append(AIMessage(content=answer))


_ARCHITECTURE_DOT = r"""
digraph Architecture {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#4b5563", fontname="Helvetica", fontsize=11];
    edge [fontname="Helvetica", fontsize=9, color="#6b7280", fontcolor="#374151"];

    subgraph cluster_a {
        label="Interface A -- Carga dos Dados";
        style=dashed; color="#9ca3af"; fontname="Helvetica"; fontsize=12;
        Upload [label="Upload\nZIP / CSV"];
        Ingestion [label="ingestion.py\nencoding, separador,\ndicionario de dados"];
        Catalog [label="catalog.py\nDataCatalog\n(DataFrames + schema)"];
        AutoCharts [label="Graficos automaticos\n(heuristica, sem LLM)", fillcolor="#dcfce7"];
    }

    subgraph cluster_b {
        label="Interface B -- Consulta em Linguagem Natural";
        style=dashed; color="#9ca3af"; fontname="Helvetica"; fontsize=12;
        Question [label="Pergunta do\nusuario"];
        Agent [label="agent.py\nLangChain create_agent\n+ LLM (Groq)", fillcolor="#fef3c7"];
        Tools [label="tools.py\n5 tools"];
        Sandbox [label="sandbox.py\nexecucao restrita\n(whitelist AST)"];
        Answer [label="Resposta\ntexto / tabela / grafico", fillcolor="#dcfce7"];
    }

    Upload -> Ingestion -> Catalog;
    Catalog -> AutoCharts;
    Catalog -> Agent [label="schema no\nsystem prompt"];
    Question -> Agent;
    Agent -> Tools [label="decide qual\ntool chamar"];
    Tools -> Sandbox [label="codigo\npandas/plotly"];
    Sandbox -> Catalog [label="le os dados", style=dashed];
    Sandbox -> Tools [label="resultado real"];
    Tools -> Agent [label="tool result"];
    Agent -> Answer;
}
"""


def _render_tab_architecture() -> None:
    st.subheader("Arquitetura da solução")
    st.write(
        "Fluxo completo: da carga do arquivo até a resposta do agente. Os módulos "
        "em verde não dependem da LLM (heurística pura); o módulo em amarelo é onde "
        "o LangChain + Groq efetivamente decidem o que fazer."
    )
    st.graphviz_chart(_ARCHITECTURE_DOT, width="stretch")
    st.caption(
        "`agent.py` monta o agente (prompt + LLM + tools). `tools.py` expõe 5 ferramentas "
        "(listar tabelas, descrever tabela, valores frequentes, executar pandas, gerar gráfico). "
        "`sandbox.py` valida por AST o código pandas/plotly antes de executá-lo -- bloqueia "
        "`import`, `eval`/`exec`, atributos `__dunder__` e loops `while`."
    )


def main() -> None:
    _init_session_state()
    st.title("Agente Inteligente para Consulta de Arquivos CSV")
    st.caption("Desafio 4 -- I2A2 -- Agentes Inteligentes e LLMs")
    st.session_state.effective_api_key = _get_api_key()

    tab_upload, tab_chat, tab_architecture = st.tabs(
        ["1. Carga dos Dados", "2. Consulta", "3. Arquitetura"]
    )
    with tab_upload:
        _render_tab_upload()
    with tab_chat:
        _render_tab_chat()
    with tab_architecture:
        _render_tab_architecture()


if __name__ == "__main__":
    main()
