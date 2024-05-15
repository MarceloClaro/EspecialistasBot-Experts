import json
import streamlit as st
import os
from typing import Tuple, Optional
from groq import Groq

# Configura a página para um layout amplo
st.set_page_config(layout="wide")

# Caminho para o arquivo JSON que contém as informações dos agentes
ARQUIVO_AGENTES = "agents.json"

# Mapeamento dos modelos de Groq para seus respectivos valores máximos de tokens
MODELO_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama2-70b-4096': 4096,
    'gemma-7b-it': 8192,
}

# Função para carregar as opções dos agentes a partir do arquivo JSON
def carregar_opcoes_agentes() -> list:
    opcoes_agentes = ['Criar (ou escolher) um especialista...']
    if os.path.exists(ARQUIVO_AGENTES):
        with open(ARQUIVO_AGENTES, 'r') as arquivo:
            try:
                agentes = json.load(arquivo)
                opcoes_agentes.extend([agente["agente"] for agente in agentes if "agente" in agente])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de agentes. Por favor, verifique o formato.")
    return opcoes_agentes

# Função para obter o número máximo de tokens permitido para um modelo específico
def obter_max_tokens(nome_modelo: str) -> int:
    return MODELO_MAX_TOKENS.get(nome_modelo, 4096)

# Função para atualizar a página
def atualizar_pagina():
    st.rerun()

# Função para salvar um novo especialista no arquivo JSON
def salvar_especialista(titulo_especialista: str, descricao_especialista: str):
    with open(ARQUIVO_AGENTES, 'r+') as arquivo:
        agentes = json.load(arquivo) if os.path.getsize(ARQUIVO_AGENTES) > 0 else []
        agentes.append({"agente": titulo_especialista, "descricao": descricao_especialista})
        arquivo.seek(0)
        json.dump(agentes, arquivo, indent=4)
        arquivo.truncate()

# Função para buscar a resposta do assistente com base na entrada do usuário
def buscar_resposta_assistente(entrada_usuario: str, nome_modelo: str, temperatura: float, selecao_agente: str) -> Tuple[str, str]:
    resposta_fase_dois = ""
    titulo_especialista = ""

    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Variável de ambiente GROQ_API_KEY não definida.")

        cliente = Groq(api_key=groq_api_key)

        def obter_completude(prompt: str) -> str:
            completude = cliente.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=nome_modelo,
                temperature=temperatura,
                max_tokens=obter_max_tokens(nome_modelo),
                top_p=1,
                stop=None,
                stream=False
            )
            return completude.choices[0].message.content

        if selecao_agente == "Criar (ou escolher) um especialista...":
            prompt_fase_um = f"Atue como engenheiro de prompt especialista. Analise a seguinte entrada para determinar o título e as características do melhor especialista para responder à pergunta. Comece a resposta com o título do especialista seguido de um ponto ['.'], depois forneça uma descrição concisa desse especialista: {entrada_usuario}"
            resposta_fase_um = obter_completude(prompt_fase_um)
            indice_primeiro_ponto = resposta_fase_um.find(".")
            titulo_especialista = resposta_fase_um[:indice_primeiro_ponto].strip()
            descricao_especialista = resposta_fase_um[indice_primeiro_ponto + 1:].strip()
            salvar_especialista(titulo_especialista, descricao_especialista)
        else:
            with open(ARQUIVO_AGENTES, 'r') as arquivo:
                agentes = json.load(arquivo)
                agente_encontrado = next((agente for agente in agentes if agente["agente"] == selecao_agente), None)
                if agente_encontrado:
                    titulo_especialista = agente_encontrado["agente"]
                    descricao_especialista = agente_encontrado["descricao"]
                else:
                    raise ValueError("Especialista selecionado não encontrado no arquivo.")

        prompt_fase_dois = f"Atue como {titulo_especialista}, um especialista no assunto, e forneça uma resposta completa e bem formatada para a seguinte pergunta: {entrada_usuario}"
        resposta_fase_dois = obter_completude(prompt_fase_dois)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return titulo_especialista, resposta_fase_dois

# Função para refinar a resposta do especialista
def refinar_resposta(titulo_especialista: str, resposta_fase_dois: str, entrada_usuario: str, nome_modelo: str, temperatura: float) -> str:
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Variável de ambiente GROQ_API_KEY não definida.")

        cliente = Groq(api_key=groq_api_key)

        def obter_completude(prompt: str) -> str:
            completude = cliente.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=nome_modelo,
                temperature=temperatura,
                max_tokens=obter_max_tokens(nome_modelo),
                top_p=1,
                stop=None,
                stream=False
            )
            return completude.choices[0].message.content

        prompt_refinamento = f"Atue como {titulo_especialista}, um especialista no assunto. Aqui está a resposta original à pergunta '{entrada_usuario}': {resposta_fase_dois}\n\nPor favor, revise e refine completamente esta resposta, fazendo melhorias e abordando quaisquer deficiências. Retorne uma versão atualizada da resposta que incorpore seus refinamentos."
        
        resposta_refinada = obter_completude(prompt_refinamento)
        return resposta_refinada

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

# Carrega as opções de agentes
opcoes_agentes = carregar_opcoes_agentes()

# Define o título da página
st.title("Groqbot Experts")
st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

# Divide a página em duas colunas
col1, col2 = st.columns(2)

with col1:
    # Entrada de texto para o usuário
    entrada_usuario = st.text_area("Por favor, insira sua solicitação:", "", key="entrada_usuario")
    # Seleciona um agente da lista de opções
    selecao_agente = st.selectbox("Escolha um Especialista", options=opcoes_agentes, index=0, key="selecao_agente")
    # Seleciona um modelo da lista de modelos
    nome_modelo = st.selectbox("Escolha um Modelo", list(MODELO_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    # Slider para selecionar o nível de criatividade
    temperatura = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    # Exibe o número máximo de tokens para o modelo selecionado
    max_tokens = obter_max_tokens(nome_modelo)
    st.write(f"Número Máximo de Tokens para o modelo selecionado: {max_tokens}")

    # Botões para buscar, refinar e atualizar a página
    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    refresh_clicked = st.button("Atualizar")

with col2:
    # Inicializa as variáveis de estado da sessão se elas ainda não existirem
    if 'resposta_assistente' not in st.session_state:
        st.session_state.resposta_assistente = ""
    if 'descricao_especialista_ideal' not in st.session_state:
        st.session_state.descricao_especialista_ideal = ""
    if 'resposta_refinada' not in st.session_state:
        st.session_state.resposta_refinada = ""
    if 'resposta_original' not in st.session_state:
        st.session_state.resposta_original = ""

    # Container para exibir as respostas
    container_saida = st.container()

    # Se o botão de busca foi clicado, busca a resposta do assistente
    if fetch_clicked:
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = buscar_resposta_assistente(entrada_usuario, nome_modelo, temperatura, selecao_agente)
        st.session_state.resposta_original = st.session_state.resposta_assistente  # Armazena a resposta original
        st.session_state.resposta_refinada = ""  # Limpa a resposta refinada ao buscar uma nova resposta

    # Se o botão de refinamento foi clicado, refina a resposta do assistente
    if refine_clicked:
        if st.session_state.resposta_assistente:
            st.session_state.resposta_refinada = refinar_resposta(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, entrada_usuario, nome_modelo, temperatura)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    # Exibe as respostas
    with container_saida:
        st.write(f"**Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**Resposta do Especialista:**\n{st.session_state.resposta_original}")  # Exibe a resposta original
        if st.session_state.resposta_refinada:
            st.write(f"\n**Resposta Refinada:**\n{st.session_state.resposta_refinada}")  # Exibe a resposta refinada

# Se o botão de atualização foi clicado, limpa todas as variáveis de estado da sessão
if refresh_clicked:
    st.session_state.clear()  # Limpa todas as variáveis de estado da sessão
    st.experimental_rerun()  # Reroda o app para resetar a visualização
