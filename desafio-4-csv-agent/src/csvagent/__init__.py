"""Agente inteligente para consulta de arquivos CSV em linguagem natural.

Desenvolvido para o Desafio 4 (I2A2). Pacote organizado em módulos:

- ingestion:  descompactação do ZIP e carga dos CSVs (encoding/separador automáticos)
- catalog:    catálogo em memória dos DataFrames + dicionário de dados
- sandbox:    execução restrita (AST) de código pandas/plotly gerado pelo LLM
- tools:      ferramentas (tools) que o agente LangChain pode chamar
- agent:      montagem do agente (prompt + LLM + tools + executor)
"""
