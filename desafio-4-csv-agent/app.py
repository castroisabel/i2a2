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
from csvagent.ingestion import load_uploads_into_catalog, sanitize_oversized_integers

load_dotenv()

st.set_page_config(page_title="Agente CSV -- Desafio 4 I2A2", page_icon="📊", layout="wide")

EXAMPLE_QUESTIONS = [
    "Qual fornecedor recebeu o maior valor no período?",
    "Qual produto apresentou o maior volume comprado?",
    "Qual foi o total gasto em cada mês?",
    "Quais foram os cinco maiores fornecedores?",
    "Gere um gráfico da evolução mensal do valor total.",
]


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
    server_key = os.environ.get("GOOGLE_API_KEY", "")
    with st.sidebar:
        st.header("Configuração")
        if server_key:
            st.success("Chave da API já configurada para este app.")
            label = "Usar outra Google API Key (Gemini) nesta sessão -- opcional"
        else:
            label = "Google API Key (Gemini)"
        user_key = st.text_input(
            label,
            type="password",
            help="Gere gratuitamente em https://aistudio.google.com/apikey",
            key="api_key_input",
        )
        st.caption(
            "Se você digitar uma chave aqui, ela fica só nesta sessão do navegador -- não é "
            "salva em disco nem enviada a lugar nenhum além da API do Google."
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


def _ensure_agent(api_key: str) -> bool:
    if not api_key:
        st.warning("Informe a Google API Key na barra lateral para liberar o chat.")
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

    with st.expander("💡 Perguntas de exemplo"):
        for question in EXAMPLE_QUESTIONS:
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
                # Alguns modelos (ex: Gemini com "thought signatures") retornam `.content`
                # como uma lista de blocos em vez de string -- `.text` normaliza para str.
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


def main() -> None:
    _init_session_state()
    st.title("📊 Agente Inteligente para Consulta de Arquivos CSV")
    st.caption("Desafio 4 -- I2A2 -- Agentes Inteligentes e LLMs")
    st.session_state.effective_api_key = _get_api_key()

    tab_upload, tab_chat = st.tabs(["1. Carga dos Dados", "2. Consulta"])
    with tab_upload:
        _render_tab_upload()
    with tab_chat:
        _render_tab_chat()


if __name__ == "__main__":
    main()
