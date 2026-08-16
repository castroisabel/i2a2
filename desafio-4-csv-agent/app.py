"""Desafio 4 (I2A2) -- Interface Inteligente para Consulta de Arquivos CSV.

Duas interfaces, como pedido no enunciado:
  A. Carga dos dados  -- upload de um .zip com CSV(s) + dicionário de dados opcional
  B. Consulta         -- chat em linguagem natural sobre os dados carregados

Rodar localmente:  uv run streamlit run app.py
"""

from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from csvagent.agent import build_agent_executor
from csvagent.autocharts import build_chart_from_dataframe, suggest_default_charts, suggest_example_questions
from csvagent.catalog import DataCatalog
from csvagent.ingestion import load_uploads_into_catalog, sanitize_oversized_integers, load_zip_into_catalog

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
    
    col_upload, col_preset = st.columns([1, 1], gap="large")

    with col_upload:
        with st.container(border=True):
            st.markdown("##### 📤 Upload de Arquivo Próprio")
            st.write("Envie arquivos **.csv**, ou um **.zip** contendo CSVs e um dicionário.")
            uploaded_files = st.file_uploader(
                "Arquivo(s) CSV ou ZIP", type=["zip", "csv"], accept_multiple_files=True, label_visibility="collapsed"
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
                    st.session_state.executor = None
                    st.session_state.chat_messages = []
                    st.session_state.lc_history = []
                    st.success(f"{len(catalog.tables)} tabela(s) carregada(s) com sucesso!")

    with col_preset:
        with st.container(border=True):
            st.markdown("##### 📦 Datasets de Exemplo Integrados")
            st.caption("Carregue dados do desafio em 1 clique:")
            
            sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
            c1, c2, c3 = st.columns(3)
            
            def load_preset(filename: str):
                path = os.path.join(sample_dir, filename)
                if not os.path.exists(path):
                    st.error(f"Arquivo não encontrado: {filename}")
                    return
                with st.spinner(f"Carregando {filename}..."):
                    try:
                        catalog, warnings = load_zip_into_catalog(path)
                    except Exception as exc:
                        st.error(f"Erro: {exc}")
                        return
                    st.session_state.catalog = catalog
                    st.session_state.catalog_warnings = warnings
                    st.session_state.executor = None
                    st.session_state.chat_messages = []
                    st.session_state.lc_history = []
                    st.success(f"{len(catalog.tables)} tabela(s) carregada(s) com sucesso!")

            with c1:
                if st.button("📊 202401_NFs", use_container_width=True, help="Notas Fiscais de Serviços (Jan/2024)"):
                    load_preset("202401_NFs.zip")
            with c2:
                if st.button("🏷️ 202505_NFe", use_container_width=True, help="Notas Fiscais Eletrônicas de Equipamentos (Maio/2025)"):
                    load_preset("202505_NFe.zip")
            with c3:
                if st.button("🛒 Compras", use_container_width=True, help="Base Completa de Compras e Fornecedores"):
                    load_preset("compras_empresa_didatico.zip")

    catalog = st.session_state.catalog
    if catalog is None:
        st.info("Nenhum dado carregado ainda. Envie CSV(s) ou um ZIP acima para liberar a aba de Consulta.")
        return

    for warning in st.session_state.catalog_warnings:
        st.warning(warning)

    st.divider()
    
    m1, m2, m3, m4 = st.columns(4)
    total_linhas = sum(len(df) for df in catalog.tables.values())
    total_colunas = sum(len(df.columns) for df in catalog.tables.values())
    tem_dicionario = len(catalog.dictionary) > 0
    with m1:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{len(catalog.tables)}</div><div class="stat-lbl">Tabelas CSV</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{total_linhas}</div><div class="stat-lbl">Total Registros</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{total_colunas}</div><div class="stat-lbl">Colunas</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{"Sim" if tem_dicionario else "Não"}</div><div class="stat-lbl">Dicionário</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Pré-visualização das tabelas carregadas:**")
    for name, df in catalog.tables.items():
        dict_note = " (dicionário de dados aplicado)" if catalog.dictionary.get(name) else ""
        with st.expander(f"`{name}` -- {len(df)} linhas x {len(df.columns)} colunas{dict_note}"):
            st.dataframe(df.head(20), width="stretch")
            if catalog.dictionary.get(name):
                st.write("Dicionário de dados:")
                st.json(catalog.dictionary[name])

            from csvagent.autocharts import suggest_default_charts
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


def _build_didactic_steps(user_question: str, trace: list[dict], tables: list, catalog: DataCatalog | None, exec_time: float) -> list[dict]:
    """Gera as 5 etapas didáticas (Chain of Thought) no padrão pedagógico do projeto de referência."""
    steps = []
    
    # 1. Recepção e Interpretação
    steps.append({
        "icon": "🌸",
        "title": "1. Recepção e Interpretação da Pergunta",
        "details": f"Pergunta recebida: *'{user_question}'*. Analisando a intenção do usuário e identificando os parâmetros da consulta."
    })
    
    # 2. Inspeção do Dicionário de Dados & Tabelas
    schema_details = []
    if catalog and catalog.tables:
        for t_name, t_df in catalog.tables.items():
            col_info = [f"{col} ({dtype})" for col, dtype in zip(t_df.columns, t_df.dtypes)]
            schema_details.append(f"Tabela **'{t_name}'**: colunas [{', '.join(col_info[:7])}{'...' if len(col_info) > 7 else ''}]")
        schema_text = " | ".join(schema_details)
    else:
        schema_text = "Inspeção automática de tabelas e esquemas carregados na memória."
        
    steps.append({
        "icon": "📖",
        "title": "2. Inspeção do Dicionário de Dados & Tabelas",
        "details": f"O agente consultou o dicionário e a estrutura de dados para entender as colunas disponíveis. Esquema Encontrado: {schema_text}"
    })
    
    # 3. Planejamento e Geração de Código Python/Pandas
    codigos = [str(step["input"]) for step in trace if step["tool"] in ("executar_pandas", "gerar_grafico")]
    if codigos:
        code_block = codigos[0]
        details_3 = f"O Agente traduziu a pergunta em linguagem natural para operações de código Pandas:\n\n```python\n{code_block}\n```"
    else:
        details_3 = "O Agente analisou a consulta e formulou o raciocínio analítico diretamente com base nas informações das tabelas."
    steps.append({
        "icon": "⚙️",
        "title": "3. Planejamento e Geração de Código Python/Pandas",
        "details": details_3
    })
    
    # 4. Execução em Sandbox e Obtenção de Dados
    if tables:
        shape_info = f"dataframe ({len(tables[0])} linhas x {len(tables[0].columns)} colunas)"
    elif codigos:
        shape_info = "resultado escalar / agregação"
    else:
        shape_info = "texto direto"
    steps.append({
        "icon": "✅",
        "title": "4. Execução em Sandbox e Obtenção de Dados",
        "details": f"Código executado com sucesso em {exec_time:.3f}s. Resultado obtido no formato: **{shape_info}**"
    })
    
    # 5. Síntese Didática e Formatação Visual
    steps.append({
        "icon": "🎨",
        "title": "5. Síntese Didática e Formatação Visual",
        "details": "Geração do resumo executivo e renderização dos componentes gráficos e tabulares para o usuário final."
    })
    
    return steps


def _render_message(message: dict) -> None:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">🎯 Resposta</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(message["content"])
            
            tables = message.get("tables", [])
            charts = list(message.get("charts", []))
            trace = message.get("trace", [])
            didactic_steps = message.get("didactic_steps", [])
            exec_time = message.get("exec_time", 0.0)
            
            # Garante que qualquer resposta com tabela gere automaticamente o gráfico Plotly se não tiver
            if not charts and tables:
                auto_fig = build_chart_from_dataframe(tables[0])
                if auto_fig is not None:
                    charts = [auto_fig]
            
            if not didactic_steps and (tables or charts or trace):
                didactic_steps = _build_didactic_steps("", trace, tables, st.session_state.catalog, exec_time)
            
            if tables or charts or trace or didactic_steps:
                res_tab1, res_tab2, res_tab3 = st.tabs([
                    "📊 Resposta Multimodal (Tabela + Gráfico)", 
                    "🧠 Bastidores do Agente", 
                    "💻 Código Executado"
                ])
                with res_tab1:
                    col_tbl, col_cht = st.columns([1, 1], gap="medium")
                    with col_tbl:
                        with st.container(border=True):
                            st.markdown("##### 📋 Tabela de Dados")
                            if tables:
                                for table in tables:
                                    st.dataframe(sanitize_oversized_integers(table), use_container_width=True)
                            else:
                                st.info("Nenhuma tabela gerada.")
                    with col_cht:
                        with st.container(border=True):
                            st.markdown("##### 📊 Gráfico Visual (Plotly)")
                            if charts:
                                for chart in charts:
                                    try:
                                        st.plotly_chart(chart, use_container_width=True)
                                    except Exception:
                                        st.info("Para esta pergunta, a exibição em texto/tabela foi a mais adequada.")
                            else:
                                st.info("Para esta pergunta, a exibição em texto/tabela foi a mais adequada.")
                                
                with res_tab2:
                    st.markdown("##### 🎓 Raciocínio Passo a Passo do Agente (Chain of Thought)")
                    st.caption(f"Tempo de execução: {exec_time:.3f}s | Estrutura: Arquitetura do Agente LangChain (Groq)")
                    
                    if didactic_steps:
                        for step in didactic_steps:
                            with st.container(border=True):
                                st.markdown(f"**{step['icon']} {step['title']}**")
                                st.markdown(step["details"])
                    elif trace:
                        for step in trace:
                            with st.container(border=True):
                                st.markdown(f"**Ferramenta:** `{step['tool']}`")
                                st.markdown("**Entrada:**")
                                st.code(str(step["input"]), language="python")
                                st.markdown("**Resultado:**")
                                st.caption(str(step['output_preview']))
                    else:
                        st.info("Nenhuma ferramenta adicional foi chamada pelo agente (respondeu direto).")
                        
                with res_tab3:
                    codigos = [str(step["input"]) for step in trace if step["tool"] in ("executar_pandas", "gerar_grafico")]
                    with st.container(border=True):
                        st.markdown("##### 💻 Código Pandas/Python Gerado")
                        if codigos:
                            for cod in codigos:
                                st.code(cod, language="python")
                            st.caption("Executado de forma segura em ambiente de sandbox.")
                        else:
                            st.info("Nenhum código Python precisou ser executado para esta resposta.")


def _render_tab_chat() -> None:
    st.subheader("Interface B -- Consulta em linguagem natural")

    catalog = st.session_state.catalog
    if catalog is None:
        st.info("Carregue um arquivo ZIP na aba 'Carga dos Dados' primeiro.")
        return

    if not _ensure_agent(st.session_state.effective_api_key):
        return

    st.caption("💡 Perguntas sugeridas (baseadas nos dados carregados):")
    example_qs = _example_questions_for_catalog(catalog)
    cols = st.columns(min(len(example_qs), 4) or 1)
    
    selected_sug = None
    for i, q in enumerate(example_qs[:4]):
        with cols[i]:
            if st.button(f"🔍 {q[:25]}...", use_container_width=True, help=q):
                selected_sug = q

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Pergunta", placeholder="Faça uma pergunta sobre os dados carregados...", label_visibility="collapsed")
        with col2:
            submit_btn = st.form_submit_button("🚀 Enviar", use_container_width=True)

    user_question = user_input if submit_btn else None
    if selected_sug:
        user_question = selected_sug

    if user_question:
        user_message = {"role": "user", "content": user_question}
        _render_message(user_message)
        
        agent_session = st.session_state.agent_session
        agent_session.reset_round()
        
        t0 = time.time()
        with st.chat_message("assistant"):
            with st.spinner("🧠 O Agente Inteligente está processando sua pergunta..."):
                try:
                    messages_history = [*st.session_state.lc_history, HumanMessage(content=user_question)]
                    result = st.session_state.executor.invoke({"messages": messages_history})
                    last_message = result["messages"][-1]
                    answer = str(last_message.text)
                except Exception as exc:  # noqa: BLE001
                    answer = (
                        "Não consegui concluir essa consulta. Detalhe técnico: "
                        f"{exc}\n\nTente reformular a pergunta ou verificar se a coluna/tabela citada existe."
                    )
        exec_time = time.time() - t0
        
        tables = list(agent_session.generated_tables)
        charts = list(agent_session.generated_charts)
        trace = list(agent_session.trace)
        
        # AUTO-CHARTING INTELIGENTE: Se não houver gráfico gerado mas houver tabela, cria com Plotly
        if not charts and tables:
            auto_fig = build_chart_from_dataframe(tables[0], title=user_question)
            if auto_fig is not None:
                charts.append(auto_fig)
                
        didactic_steps = _build_didactic_steps(user_question, trace, tables, catalog, exec_time)
        
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "tables": tables,
            "charts": charts,
            "trace": trace,
            "didactic_steps": didactic_steps,
            "exec_time": exec_time,
        }
        
        st.session_state.chat_messages.append(user_message)
        st.session_state.chat_messages.append(assistant_message)
        st.session_state.lc_history.append(HumanMessage(content=user_question))
        st.session_state.lc_history.append(AIMessage(content=answer))
        st.rerun()

    st.markdown("---")
    
    # Renderizar histórico de respostas em ordem decrescente (mais recente no topo)
    all_msgs = st.session_state.chat_messages
    pairs = []
    for i in range(0, len(all_msgs), 2):
        if i + 1 < len(all_msgs):
            pairs.append((all_msgs[i], all_msgs[i+1]))
        else:
            pairs.append((all_msgs[i], None))
            
    for u_msg, a_msg in reversed(pairs):
        _render_message(u_msg)
        if a_msg:
            _render_message(a_msg)
_ARCHITECTURE_DOT = r"""
digraph Architecture {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#3b82f6", fontname="Helvetica", fontsize=10, fontcolor="#0f172a", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=9, color="#94a3b8", fontcolor="#38bdf8", penwidth=1.2];

    subgraph cluster_a {
        label="Interface A - Carga dos Dados";
        fontcolor="#cbd5e1";
        fontname="Helvetica";
        fontsize=11;
        style="rounded,dashed";
        color="#64748b";
        
        InputData [label="Upload ZIP / CSV ou\nDatasets de Exemplo", fillcolor="#f1f5f9"];
        Ingestion [label="ingestion.py\nEncoding, separador,\ndicionario e sanitizacao"];
        Catalog [label="catalog.py\nDataCatalog\n(DataFrames + Schema)"];
        AutoChartsA [label="autocharts.py\nGraficos Iniciais e\nPerguntas Sugeridas", fillcolor="#dcfce7", color="#10b981"];
    }

    subgraph cluster_b {
        label="Interface B - Consulta em Linguagem Natural";
        fontcolor="#cbd5e1";
        fontname="Helvetica";
        fontsize=11;
        style="rounded,dashed";
        color="#64748b";
        
        Question [label="Pergunta do Usuario\n(Texto ou Sugestoes)", fillcolor="#f1f5f9"];
        Agent [label="agent.py\nLangChain create_agent\n+ LLM (Groq)", fillcolor="#fef3c7", color="#f59e0b"];
        Tools [label="tools.py\n5 Ferramentas\n(Pandas, Plotly, Schema)"];
        Sandbox [label="sandbox.py\nExecucao Restrita\n(Whitelist AST)"];
        AutoChartB [label="autocharts.py\nAuto-Charting Plotly\n(Linhas e Barras)", fillcolor="#dcfce7", color="#10b981"];
        DidacticEngine [label="Bastidores do Agente\n(Chain of Thought 5 Etapas)", fillcolor="#f3e8ff", color="#a855f7"];
        Answer [label="Resposta Multimodal\n(Resumo, Tabela, Grafico, Codigo)", fillcolor="#e0f2fe", color="#0284c7"];
    }

    InputData -> Ingestion -> Catalog;
    Catalog -> AutoChartsA;
    Catalog -> Agent [label="schema no\nsystem prompt"];
    Question -> Agent;
    Agent -> Tools [label="tool calling"];
    Tools -> Sandbox [label="codigo gerado"];
    Sandbox -> Catalog [label="ler tabelas", style=dashed];
    Sandbox -> Tools [label="resultado real"];
    Tools -> Agent [label="observacoes"];
    Agent -> DidacticEngine [label="rastreio das etapas"];
    Agent -> AutoChartB [label="dados tabulares"];
    AutoChartB -> Answer;
    DidacticEngine -> Answer;
}
"""


def _render_tab_architecture() -> None:
    st.subheader("Arquitetura da Solução")
    st.markdown("""
        <div class="insight-card" style="margin-bottom: 1.2rem;">
            <div class="insight-title">🏗️ Visão Geral da Arquitetura do Sistema</div>
            <div class="insight-body">
                Fluxo completo ponta a ponta: desde a ingestão dos dados até a resposta multimodal com raciocínio didático.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.graphviz_chart(_ARCHITECTURE_DOT, width="stretch")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("##### 📥 Interface A: Ingestão e Catálogo")
            st.markdown("""
            - **Upload e Datasets Integrados**: Suporta envio de arquivos `.ZIP`/`.CSV` e seleção de bases pré-carregadas.
            - **`ingestion.py`**: Detecção automática de encoding, separador delimitador e mapeamento de dicionários de dados. Sanitização de identificadores e datas.
            - **`catalog.py`**: Estrutura em memória `DataCatalog` que armazena os DataFrames e gera sumários de contexto.
            - **`autocharts.py`**: Heurística leve que deriva perguntas sugeridas e gráficos iniciais automáticos sem depender do LLM.
            """)
    
    with col2:
        with st.container(border=True):
            st.markdown("##### 🧠 Interface B: Agente e Raciocínio")
            st.markdown("""
            - **`agent.py`**: Orquestrador LangChain (`create_agent`) conectado à LLM de alta velocidade (**Groq**) com prompt contextualizado.
            - **`tools.py`**: Expõe 5 ferramentas analíticas (`listar_tabelas`, `descrever_tabela`, `valores_frequentes`, `executar_pandas`, `gerar_grafico`).
            - **`sandbox.py`**: Validação rigorosa por AST para bloquear operações inseguras (`import`, `eval`/`exec`, `while`, dunder).
            - **Visualização & Didática**: Auto-charting com Plotly e decomposição do raciocínio em **5 etapas pedagógicas** (Chain of Thought).
            """)


def main() -> None:
    _init_session_state()
    st.session_state.effective_api_key = _get_api_key()
    st.markdown("""
        <style>
        /* Fontes e importação */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Estilização da Sidebar */
        [data-testid="stSidebar"], 
        section[data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%) !important;
            color: #FFFFFF !important;
        }
    
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > p,
        [data-testid="stSidebar"] [data-testid="stHeader"] {
            color: #FFFFFF !important;
        }
    
        [data-testid="stSidebar"] .stCaption, 
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #94A3B8 !important;
        }
    
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2) !important;
        }
    
        /* Expanders na Sidebar */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 10px !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
        [data-testid="stSidebar"] [data-testid="stExpander"] svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
    
        /* Botões na Sidebar */
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stDownloadButton > button {
            background: rgba(255, 255, 255, 0.12) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            transition: all 0.2s ease !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] .stDownloadButton > button:hover {
            background: rgba(255, 255, 255, 0.22) !important;
            border-color: rgba(255, 255, 255, 0.4) !important;
            color: #FFFFFF !important;
        }
        
        /* Hero Header Banner */
        .hero-card {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            color: #FFFFFF;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
            margin-bottom: 1rem;
        }
        .hero-badge {
            background: rgba(255, 255, 255, 0.12);
            color: #93C5FD;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 0rem;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 0;
            line-height: 1.25;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #94A3B8;
            margin-top: 0.5rem;
            font-weight: 400;
        }
    
        /* Cards de Estatísticas e Metadados */
        .stat-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            text-align: center;
        }
        .stat-val {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1E293B;
        }
        .stat-lbl {
            font-size: 0.8rem;
            color: #64748B;
            font-weight: 500;
            text-transform: uppercase;
        }
    
        /* Card Didático de Resposta */
        .insight-card {
            background: #F0F9FF;
            border: 1px solid #BAE6FD;
            border-left: 5px solid #0284C7;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.2rem;
        }
        .insight-title {
            font-weight: 700;
            color: #0369A1;
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .insight-body {
            color: #0F172A;
            font-size: 1.05rem;
            line-height: 1.5;
        }
    
        /* Timeline Stepper dos Bastidores do Agente */
        .step-box {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            transition: all 0.2s ease;
        }
        .step-box:hover {
            border-color: #93C5FD;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.05);
        }
        
        /* Estilização Avançada das Abas Principais (Estilo Hero Banner) */
        .stTabs, [data-testid="stTabs"] {
            background-color: transparent !important;
        }
    
        .stTabs [role="tablist"], 
        [data-testid="stTabs"] [role="tablist"], 
        div[data-baseweb="tab-list"],
        div[role="tablist"] {
            gap: 0.8rem !important;
            padding: 0.4rem 0 0.8rem 0 !important;
            border-bottom: none !important;
        }
        
        .stTabs [role="tab"], 
        [data-testid="stTab"], 
        button[role="tab"], 
        div[role="tab"], 
        button[data-baseweb="tab"] {
            border-radius: 12px !important;
            padding: 0.75rem 1.6rem !important;
            font-weight: 600 !important;
            font-size: 0.98rem !important;
            color: #475569 !important;
            background: #F1F5F9 !important;
            border: 1px solid #E2E8F0 !important;
            transition: all 0.25s ease-in-out !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
            margin-right: 0.4rem !important;
        }
    
        .stTabs [role="tab"] p, 
        [data-testid="stTab"] p, 
        button[role="tab"] p, 
        button[role="tab"] span, 
        div[role="tab"] p {
            color: #475569 !important;
            font-weight: 600 !important;
            font-size: 0.98rem !important;
            margin: 0 !important;
        }
    
        .stTabs [role="tab"]:hover, 
        [data-testid="stTab"]:hover, 
        button[role="tab"]:hover {
            background: #E2E8F0 !important;
            border-color: #CBD5E1 !important;
            transform: translateY(-2px);
        }
    
        .stTabs [role="tab"]:hover p, 
        [data-testid="stTab"]:hover p, 
        button[role="tab"]:hover p {
            color: #0F172A !important;
        }
    
        /* Aba Selecionada com Destaque em Gradiente Hero */
        .stTabs [role="tab"][aria-selected="true"], 
        [data-testid="stTab"][aria-selected="true"], 
        button[role="tab"][aria-selected="true"], 
        div[role="tab"][aria-selected="true"], 
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%) !important;
            border: 1px solid #3B82F6 !important;
            border-bottom: 1px solid #3B82F6 !important;
            box-shadow: 0 8px 20px -4px rgba(30, 58, 138, 0.4) !important;
            outline: none !important;
        }
        
        .stTabs [role="tab"][aria-selected="true"] p, 
        [data-testid="stTab"][aria-selected="true"] p, 
        button[role="tab"][aria-selected="true"] p, 
        button[role="tab"][aria-selected="true"] span, 
        div[role="tab"][aria-selected="true"] p {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
    
        /* Remover Pseudo-elementos e Bordas Inferiores que possam gerar linhas vermelhas */
        .stTabs [role="tab"]::after,
        [data-testid="stTab"]::after,
        button[role="tab"]::after,
        .stTabs [role="tab"]::before,
        [data-testid="stTab"]::before,
        button[role="tab"]::before {
            display: none !important;
            content: "" !important;
            height: 0px !important;
            background: transparent !important;
            border: none !important;
        }
    
        /* Ocultar 100% APENAS a linha de destaque / highlight bar do Streamlit */
        [data-baseweb="tab-highlight"], 
        [data-testid="stTabHighlight"], 
        .stTabs [data-baseweb="tab-highlight"], 
        .stTabs [data-testid="stTabHighlight"], 
        div[data-baseweb="tab-highlight"], 
        div[data-testid="stTabHighlight"], 
        div[class*="TabHighlight"], 
        div[class*="tab-highlight"], 
        div[class*="tabHighlight"],
        [data-baseweb="tab-border"],
        [data-testid="stTabBorder"] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0px !important;
            min-height: 0px !important;
            max-height: 0px !important;
            width: 0px !important;
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            border-bottom: none !important;
            box-shadow: none !important;
        }
    
        /* Garantir que o painel de conteúdo das abas fique 100% visível e acessível */
        div[role="tabpanel"], 
        [data-testid="stTabContent"], 
        .stTabs [role="tabpanel"],
        .stTabs div[data-testid="stTabContent"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            height: auto !important;
            padding-top: 1rem !important;
        }
        
        /* Botões Didáticos de Sugestão */
        .stButton>button {
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
    
        /* Ocultar instruções automáticas do st.form ("Press Enter to submit form") */
        div[data-testid="InputInstructions"] {
            display: none !important;
            visibility: hidden !important;
        }
    
        /* Ocultar elementos desnecessários do Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero-card">
            <span class="hero-badge">I2A2 • DESAFIO 04</span>
            <h1 class="hero-title">Interface Inteligente para Consulta de Dados</h1>
            <p class="hero-subtitle">Consulte planilhas e arquivos CSV (.ZIP) utilizando linguagem natural.</p>
        </div>
    """, unsafe_allow_html=True)


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
