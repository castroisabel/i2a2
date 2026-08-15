"""Monta o agente LangChain: prompt + LLM (Gemini) + tools + executor.

Framework exigido pelo desafio: LangChain (`langchain.agents.create_agent`,
a API atual da 1.x, que orquestra o loop de tool-calling via LangGraph por
baixo dos panos). O modelo recebe, no system prompt, um resumo do catálogo de
dados (schema + amostra + dicionário de dados, quando houver) e decide sozinho
quais tools chamar e em que ordem para responder a pergunta em linguagem
natural do usuário.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from csvagent.catalog import DataCatalog
from csvagent.tools import AgentSession, build_tools

SYSTEM_PROMPT_TEMPLATE = """\
Você é um analista de dados que responde perguntas em linguagem natural sobre
arquivos CSV carregados pelo usuário. NUNCA responda de memória ou "chute" um
número -- toda resposta quantitativa deve vir da execução real de uma
ferramenta (tool) sobre os dados. Se não tiver certeza do nome de uma tabela
ou coluna, use `listar_tabelas` e `descrever_tabela` antes de tentar calcular
qualquer coisa.

Catálogo de dados disponível nesta sessão:

{catalogo}

Diretrizes:
- Prefira `executar_pandas` para agregações, somas, contagens, rankings, filtros, etc.
- Use `gerar_grafico` quando o usuário pedir um gráfico ou quando uma visualização
  ajudar a responder (ex: evolução no tempo, comparação entre categorias).
- Sempre que possível, responda com números concretos vindos do resultado da tool
  (ex: "R$ 128.430,00", "Fornecedor XPTO"), não apenas descreva o que a tool fez.
- Se a pergunta for ambígua ou os dados não permitirem respondê-la com confiança,
  diga isso claramente ao usuário em vez de inventar uma resposta.
- Responda sempre em português do Brasil, de forma direta e objetiva.
"""


def build_agent_executor(catalog: DataCatalog, google_api_key: str):
    """Retorna (grafo do agente, AgentSession). O grafo é invocado com
    `{"messages": [...]}` e devolve `{"messages": [...]}` -- a última mensagem
    (sem tool_calls) é a resposta final."""
    session = AgentSession(catalog=catalog)
    tools = build_tools(session)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0,
    )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(catalogo=catalog.context_summary())
    graph = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    return graph, session
