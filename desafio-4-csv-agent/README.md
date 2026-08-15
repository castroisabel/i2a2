# Agente Inteligente para Consulta de Arquivos CSV

Desafio 4 -- I2A2 (Instituto de Inteligência Artificial Aplicada).

Aplicação Streamlit com duas interfaces:

- **A. Carga dos dados** -- upload de um ou mais arquivos `.csv` soltos, ou de um `.zip`
  contendo um ou mais CSVs e, opcionalmente, um arquivo de dicionário de dados (detectado
  automaticamente).
- **B. Consulta** -- chat em linguagem natural sobre os dados carregados, respondido
  por um agente LangChain que escreve e executa pandas/plotly de verdade sobre os
  dados (nunca "chuta" um número).

## Framework exigido

**LangChain** (`langchain.agents.create_agent`, API 1.x baseada em LangGraph) +
**Google Gemini** (`gemini-2.5-flash` via `langchain-google-genai`).

## Rodando localmente

```bash
uv sync
```

Crie um arquivo `.env` na raiz do projeto (nunca é commitado) com sua chave gratuita
do Google AI Studio (https://aistudio.google.com/apikey):

```
GOOGLE_API_KEY=sua_chave_aqui
```

Depois:

```bash
uv run streamlit run app.py
```

Se preferir não usar `.env`, é possível colar a chave diretamente na barra lateral
do app (fica só na sessão do navegador).

## Arquitetura

```
app.py                     Interface Streamlit (abas A e B)
src/csvagent/
  ingestion.py              Extrai o ZIP, detecta encoding/separador, identifica
                             o dicionário de dados
  catalog.py                Catálogo em memória: DataFrames + schema + dicionário
  sandbox.py                Execução restrita (whitelist de AST) do código
                             pandas/plotly gerado pelo LLM
  tools.py                  5 tools do agente (listar, descrever, valores
                             frequentes, executar pandas, gerar gráfico)
  agent.py                  Monta o agente LangChain (prompt + LLM + tools)
```

### Como o agente decide

O LLM recebe, no system prompt, um resumo do catálogo (nome de cada tabela,
colunas, tipos, descrição do dicionário de dados e uma amostra de linhas) e a
instrução explícita de nunca responder de memória. A cada pergunta, ele decide
livremente (tool-calling) qual sequência de ferramentas chamar -- tipicamente:

1. `listar_tabelas` / `descrever_tabela` -- se não tiver certeza do schema
2. `valores_frequentes` -- para explorar os valores de uma coluna categórica
3. `executar_pandas` -- para agregações, somas, contagens, rankings, filtros
4. `gerar_grafico` -- quando a pergunta pede ou se beneficia de um gráfico

O código gerado nos passos 3 e 4 passa por um validador de AST (`sandbox.py`)
antes de ser executado: bloqueia `import`, `eval`/`exec`, acesso a atributos
`__dunder__`, definição de função/classe e loops `while` (para evitar loop
infinito trivial). Isso não é um sandbox de isolamento de processo/SO --
é uma defesa em profundidade adequada ao escopo educacional do desafio.

Na Interface B, cada resposta vem acompanhada de um expander "Ver raciocínio
do agente", mostrando exatamente quais tools foram chamadas, com que código,
e qual resultado real cada uma retornou -- é a evidência de que a resposta
não foi produzida por uma LLM "conversando sozinha", mas pela aplicação
consultando os dados de fato.

## Limitações conhecidas (MVP)

- Um único dicionário de dados é aplicado a todas as tabelas do ZIP (cenário
  mais comum quando cabeçalho/itens de uma mesma nota compartilham glossário).
- Sem timeout de execução (o sandbox impede `while`, mas um `for` sobre uma
  tabela muito grande ainda pode ser lento).
- Sem suporte a múltiplos ZIPs na mesma sessão (recarregar substitui o catálogo).
