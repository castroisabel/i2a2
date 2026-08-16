# Relatório Didático: Interface Inteligente para Consulta de Dados (CSV Agent)
**Curso / Programa:** I2A2 — Instituto de Inteligência Artificial Aplicada  
**Desafio 04:** Construção de uma Interface Inteligente para Consulta de Arquivos CSV com Agente Autônomo  
**Tecnologias Centrais:** Python, Streamlit, LangChain, Groq (LLM), Pandas, Plotly, AST Sandbox, Graphviz  

---

## 1. Introdução e Objetivo Pedagógico

O objetivo deste projeto é capacitar os alunos no desenvolvimento de **sistemas de Inteligência Artificial Generativa aplicados à análise de dados reais**. 

Diferente de chatbots tradicionais que geram respostas textuais puramente probabilísticas (sujeitas a "alucinações" e erros de cálculo matemático), este sistema implementa o padrão **AI Agent (Agente Autônomo com Tool Calling)**. 

### Por que esta abordagem é fundamental para o aluno de IA?
1. **Determinismo Numérico**: A LLM não faz contas de cabeça. Ela formula código Python/Pandas que é executado diretamente sobre a base de dados real em um ambiente controlado.
2. **Transparência Pedagógica (Explainable AI)**: Cada decisão tomada pelo agente é decomposta e apresentada ao usuário em 5 etapas estruturadas (*Bastidores do Agente*).
3. **Multimodalidade**: A resposta combina resumo executivo, tabela de dados detalhada, gráfico interativo Plotly e o script de código gerado.

---

## 2. Arquitetura da Solução e Fluxo Ponta a Ponta

O sistema é dividido em duas interfaces principais e uma aba arquitetural explicativa:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             Interface A • Carga dos Dados                                │
│                                                                                          │
│  [ Upload ZIP/CSV ou ] ──► [ ingestion.py: Detecção ] ──► [ catalog.py: DataCatalog ]   │
│  [ Datasets Nativos  ]     [ Encoding, Sep & Dicion.]     [ (DataFrames + Schemas)  ]    │
│                                                                  │                       │
│                                                                  ▼                       │
│                                                     [ autocharts.py: Perguntas & Gráf. ] │
└──────────────────────────────────────────────────────────────────┬───────────────────────┘
                                                                   │ (Schema no prompt)
┌──────────────────────────────────────────────────────────────────▼───────────────────────┐
│                       Interface B • Consulta em Linguagem Natural                        │
│                                                                                          │
│  [ Pergunta do Usuário ] ──► [ agent.py: LangChain + Groq ]                              │
│                                      │                                                   │
│                                      ▼ (Tool Calling / Chamada de Ferramentas)           │
│                                [ tools.py: 5 Tools ] ◄───► [ sandbox.py: AST Whitelist ] │
│                                      │                                                   │
│                 ┌────────────────────┴────────────────────┐                              │
│                 ▼                                         ▼                              │
│   [ autocharts.py: Auto-Charting ]        [ Bastidores do Agente: 5 Etapas ]            │
│   (Plotly: Linhas & Barras)               (Chain of Thought Didático)                    │
│                 │                                         │                              │
│                 └────────────────────┬────────────────────┘                              │
│                                      ▼                                                   │
│                 [ Resposta Multimodal: Resumo + Tabela + Gráfico + Código ]              │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pilha de Tecnologias, Bibliotecas e Funcionalidades

Abaixo estão detalhadas todas as bibliotecas adotadas no projeto e o papel de cada uma no aprendizado dos alunos:

| Biblioteca | Versão / Tipo | Função no Projeto | Por que foi escolhida? |
| :--- | :--- | :--- | :--- |
| **Streamlit** | `>= 1.35.0` | Interface Web Interativa (UI) | Permite criar SPAs (Single Page Applications) profissionais em Python puro, com suporte nativo a reatividade, abas, formulários e injeção de CSS personalizado. |
| **LangChain** | `>= 0.3.0` | Framework de Orquestração de Agentes | Padroniza a criação de agentes cognitivos (`create_agent`), gerenciamento de histórico de conversação (`HumanMessage`, `AIMessage`) e integração de ferramentas (`@tool`). |
| **Groq (`langchain-groq`)** | `>= 0.2.0` | Provedor de Inferência LLM de Ultra-Baixa Latência | Utiliza processadores LPU (Language Processing Units) oferecendo respostas quase instantâneas (centenas de tokens por segundo) em tier gratuito ideal para fins educacionais. |
| **Pandas** | `>= 2.2.0` | Manipulação e Análise de Dados Tabulares | Motor analítico fundamental para filtragem, agrupamento (`groupby`), ordenação, agregações matemáticas e tratamento de datas/períodos. |
| **Plotly Express & Graph Objects** | `>= 5.20.0` | Visualização Gráfica Interativa | Geração de gráficos interativos (zoom, hover, exportação PNG) com renderização baseada em JavaScript via WebAssembly no Streamlit. |
| **AST (`ast` - Python Built-in)** | Módulo Nativo | Validação de Segurança (Sandbox Estático) | Inspeciona a Árvore Sintática Abstrata do código gerado pela LLM antes de sua execução, bloqueando comandos maliciosos ou loops infinitos. |
| **Chardet** | `>= 5.2.0` | Detecção Heurística de Codificação | Identifica automaticamente encodings de arquivos brasileiros (como `UTF-8`, `ISO-8859-1`, `Windows-1252`) prevenindo falhas de leitura de caracteres com acentuação. |
| **Graphviz** | Dot Engine | Diagramação Declarativa da Arquitetura | Gera o diagrama de blocos visual da arquitetura a partir de texto na aba informativa. |
| **Dotenv (`python-dotenv`)** | `>= 1.0.0` | Gestão de Variáveis de Ambiente | Isola segredos e chaves de API (`GROQ_API_KEY`) fora do repositório de código fonte, seguindo o padrão 12-Factor App. |
| **UV** | Modern Package Manager | Gerenciador de Dependências e Ambiente | Alternativa ultrarrápida ao pip/poetry escrita em Rust, garantindo reprodutibilidade exata com `uv.lock`. |

---

## 4. Estrutura Modular e Funcionamento dos Arquivos do Projeto

O código do projeto foi organizado segundo o princípio de **responsabilidade única (Single Responsibility Principle)**:

```
desafio-4-csv-agent/
├── app.py                          # Ponto de entrada: UI Streamlit, Layout em Cards e Gestão de Estado
├── pyproject.toml / uv.lock        # Metadados e dependências fixadas
├── sample_data/                    # Datasets de exemplo (.ZIP) integrados para teste rápido
│   ├── 202401_NFs.zip
│   ├── 202505_NFe.zip
│   └── compras_empresa_didatico.zip
└── src/
    └── csvagent/
        ├── __init__.py             # Módulo Python
        ├── agent.py                # Prompt do Sistema, Configuração Groq e Agente LangChain
        ├── catalog.py              # Classe DataCatalog (tabelas, esquemas e metadados)
        ├── ingestion.py            # Ingestão de ZIP/CSV, detecção de separador, encoding e dicionário
        ├── tools.py                # 5 Ferramentas disponibilizadas para a LLM
        ├── sandbox.py              # Validador de Segurança AST e Sandbox de Execução
        └── autocharts.py           # Auto-charting Plotly e Heurística de Perguntas Sugeridas
```

### 4.1. `src/csvagent/ingestion.py` (Ingestão Resiliente de Dados)
- **Detecção de Codificação (`_detect_encoding`)**: Amostra os primeiros 32KB do arquivo e usa `chardet` para inferir a codificação real.
- **Detecção de Separador (`_detect_separator`)**: Avalia a consistência de linhas com delimitadores comuns (`,`, `;`, `\t`, `|`).
- **Mapeamento de Dicionário de Dados (`_parse_dictionary`)**: Identifica planilhas que funcionam como dicionários/glossários e cria um mapa de `coluna -> descrição` para orientar o modelo.
- **Sanitização de Tipos (`sanitize_oversized_integers`)**: Converte inteiros gigantescos (ex: chaves de 44 dígitos de NF-e) e tipos `pd.Period` para texto, evitando estouro de memória no Apache Arrow e falhas de serialização no Plotly.

### 4.2. `src/csvagent/catalog.py` (Catálogo em Memória)
- Mantém o dicionário `tables: dict[str, pd.DataFrame]` com todas as tabelas carregadas.
- Gera o **resumo contextual (`context_summary`)** injetado no System Prompt do Agente contendo:
  - Nomes das tabelas e quantidade de linhas.
  - Lista de colunas e seus tipos de dados (`int64`, `float64`, `object`, `datetime64`).
  - Descrições extraídas do dicionário de dados (quando existentes).
  - Amostra representativa das 3 primeiras linhas de cada tabela.

### 4.3. `src/csvagent/agent.py` (Orquestração do Agente Cognitivo)
- Utiliza a API `create_agent` do LangChain acoplada ao modelo `openai/gpt-oss-20b` via Groq.
- O **System Prompt** define o comportamento rigoroso do analista de dados:
  - **Proibição estrita de adivinhação**: Nenhum número pode ser inventado.
  - **Uso obrigatório de ferramentas**: O modelo deve consultar o catálogo e executar comandos reais para responder perguntas quantitativas.
  - **Foco Executivo**: A resposta em texto deve ser objetiva e direta, pois os dados completos e gráficos são gerados automaticamente pelas ferramentas na UI.

### 4.4. `src/csvagent/tools.py` (As 5 Ferramentas do Agente)
O agente possui 5 ferramentas especializadas decoradas com `@tool`:
1. `listar_tabelas()`: Lista os arquivos e dimensões disponíveis.
2. `descrever_tabela(nome_tabela)`: Mostra o schema completo, tipos, estatísticas descritivas de colunas numéricas (`describe()`) e amostra de dados.
3. `valores_frequentes(nome_tabela, coluna, top_n)`: Retorna a contagem das categorias mais frequentes de uma coluna para guiar filtros.
4. `executar_pandas(codigo)`: Executa snippets de agregação, agrupamento, somas e filtros sobre as tabelas e registra os DataFrames gerados na sessão.
5. `gerar_grafico(codigo)`: Executa snippets com `plotly.express` (`px`) ou `plotly.graph_objects` (`go`) gerando figuras personalizadas.

### 4.5. `src/csvagent/sandbox.py` (Segurança com AST)
Para garantir que a execução de código arbitrário gerado por IA seja segura, o código passa por um validador de **Árvore Sintática Abstrata (AST)**:
- **Nós Permitidos (Whitelist)**: Apenas expressões matemáticas, chamadas de métodos, atribuições simples, indexações, dicionários e listas são aceitos.
- **Operações Bloqueadas**:
  - `import` e `__import__` (bloqueia acesso a módulos do sistema operacional como `os`, `sys`, `subprocess`).
  - Funções de execução dinâmica (`eval`, `exec`, `compile`, `open`).
  - Atributos mágicos/dunder (`__class__`, `__subclasses__`, `__globals__`).
  - Estruturas de repetição `while` (evita travamento por loop infinito).

### 4.6. `src/csvagent/autocharts.py` (Visualização Gráfica Inteligente)
- **Auto-Charting Dinâmico (`build_chart_from_dataframe`)**: 
  - Detecta se o resultado da consulta possui colunas temporais (meses, anos, datas) e cria automaticamente um **gráfico de linha com marcadores roxos (`#7C3AED`)**.
  - Detecta dados categóricos e gera um **gráfico de barras azuis (`#3B82F6`)**.
- **Sugestões de Perguntas Heurísticas (`suggest_example_questions`)**: Examina as colunas do dataset recém-carregado e cria chips de perguntas contextualizadas para o usuário clicar e testar imediatamente.

---

## 5. Jornada Didática do Agente

Para tornar o raciocínio da IA totalmente auditável e compreensível para os alunos, a aplicação organiza o ciclo de resolução da consulta em **5 etapas sequenciais**:

1. **🌸 1. Recepção e Interpretação da Pergunta**:
   - Captura a consulta em linguagem natural enviada pelo usuário e extrai as entidades, intenção analítica e parâmetros temporais/categóricos.
2. **📖 2. Inspeção do Dicionário de Dados & Tabelas**:
   - O agente analisa as tabelas em memória, verifica nomes exatos de colunas e consulta o dicionário de dados para entender os tipos de variáveis.
3. **⚙️ 3. Planejamento e Geração de Código Python/Pandas**:
   - Tradução da intenção analítica para uma consulta estruturada em Pandas (ex: agrupamento por período, soma de quantidades e renomeação de colunas).
4. **✅ 4. Execução em Sandbox e Obtenção de Dados**:
   - Validação da sintaxe pelo validador AST e execução restrita em sandbox. Mede com precisão o tempo de resposta em milissegundos e retorna a dimensão da matriz de dados resultante.
5. **🎨 5. Síntese Didática e Formatação Visual**:
   - Consolidação executiva da resposta, exibição da tabela de dados higienizada e acionamento do gerador gráfico Plotly.

---

## 6. Como Executar e Demonstrar o Projeto

### Passo 1: Instalação e Preparação do Ambiente
```bash
# 1. Clone o repositório
git clone https://github.com/castroisabel/i2a2.git
cd i2a2

# 2. Sincronize as dependências com uv
uv sync
```

### Passo 2: Configuração da Chave Groq
Crie um arquivo `.env` na raiz do projeto:
```env
GROQ_API_KEY=gsk_sua_chave_aqui
```
*(Caso não deseje criar o arquivo `.env`, a chave pode ser colada diretamente na barra lateral da aplicação web).*

### Passo 3: Execução da Aplicação
```bash
uv run streamlit run app.py
```

### Roteiro Recomendado de Demonstração:
1. **Aba 1 (Carga dos Dados)**: Clique em um dos botões da seção *Datasets de Exemplo Integrados* (ex: `🛒 Compras`). Demonstre como o sistema detecta as tabelas, tipos de dados e já gera perguntas de exemplo baseadas nas colunas reais.
2. **Aba 2 (Consulta)**: Clique em uma pergunta sugerida como *"Qual foi o total de quantidade em cada mês?"* ou digite sua própria pergunta.
3. **Exploração Multimodal**:
   - Mostre o **Resumo da Resposta** no card superior.
   - Na aba **📊 Resposta Multimodal**, mostre a **Tabela de Dados** à esquerda e o **Gráfico Plotly interativo** à direita.
   - Na aba **🧠 Bastidores do Agente**, navegue pelas 5 etapas didáticas explicando como o agente raciocinou.
   - Na aba **💻 Código Executado**, mostre o código Python gerado e explique o funcionamento da sandbox de segurança.
4. **Aba 3 (Arquitetura)**: Apresente o diagrama Graphviz e os blocos conceituais de cada módulo de software.

---

## 7. Conclusão e Lições Aprendidas para Projetos de IA

A implementação deste projeto demonstra conceitos fundamentais da engenharia moderna de Inteligência Artificial:
- **Agentes baseados em código (Code-as-Action)** são imensamente superiores a abordagens puramente textuais para análise de dados tabulares.
- **Segurança em Primeiro Lugar**: Ambientes que executam código gerado por LLM exigem defesas em camadas (inspeção de AST, restrição de escopo de variáveis e isolamento de I/O).
- **Design e Didática Importam**: Interfaces ricas (com cards, abas e rastreabilidade visual) elevam a confiança do usuário final e facilitam o diagnóstico e aprendizado de alunos e desenvolvedores.
