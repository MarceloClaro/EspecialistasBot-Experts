import json
import streamlit as st
import os
from typing import Tuple, Optional
from groq import Groq

# Define o layout da página como "wide" (amplo)
st.set_page_config(layout="wide")

# Define o caminho do arquivo para salvar os agentes
FILEPATH = "agents.json"

# Dicionário que mapeia os nomes dos modelos aos seus limites máximos de tokens
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama2-70b-4096': 4096,
    'gemma-7b-it': 8192,
}

def load_agent_options() -> list:
    # Função que carrega as opções de agentes do arquivo ou retorna uma lista com "Create (or choose) an expert..."
    agent_options = ['Create (or choose) an expert...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agent"] for agent in agents if "agent" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de agentes. Por favor, verifique o formato.")
    return agent_options

def get_max_tokens(model_name: str) -> int:
    # Função que retorna o limite máximo de tokens para um modelo específico
    return MODEL_MAX_TOKENS.get(model_name, 4096)

def refresh_page():
    # Função que recarrega a página
    st.rerun()

def save_expert(expert_title: str, expert_description: str):
    # Função que salva um novo expert no arquivo
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agent": expert_title, "description": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

def fetch_assistant_response(user_input: str, model_name: str, temperature: float, agent_selection: str) -> Tuple[str, str]:
    # Função que obtém a resposta do assistente com base na entrada do usuário, modelo, temperatura e seleção de agente
    phase_two_response = ""
    expert_title = ""

    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Variável de ambiente GROQ_API_KEY não definida.")

        client = Groq(api_key=groq_api_key)

        def get_completion(prompt: str) -> str:
            # Função auxiliar que obtém a resposta do modelo com base em um prompt
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=get_max_tokens(model_name),
                top_p=1,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content

        if agent_selection == "Create (or choose) an expert...":
            # Se a opção selecionada for criar um novo expert
            phase_one_prompt = f"Aja como um engenheiro de prompts especialistas. Analise a seguinte entrada para determinar o título e as características do melhor expert para responder à pergunta. Comece a resposta com o título do expert seguido de um ponto ['.'], em seguida, forneça uma descrição concisa desse expert: {user_input}"
            phase_one_response = get_completion(phase_one_prompt)
            first_period_index = phase_one_response.find(".")
            expert_title = phase_one_response[:first_period_index].strip()
            expert_description = phase_one_response[first_period_index + 1:].strip()
            save_expert(expert_title, expert_description)
        else:
            # Se um expert existente foi selecionado
            with open(FILEPATH, 'r') as file:
                agents = json.load(file)
                agent_found = next((agent for agent in agents if agent["agent"] == agent_selection), None)
                if agent_found:
                    expert_title = agent_found["agent"]
                    expert_description = agent_found["description"]
                else:
                    raise ValueError("Expert selecionado não encontrado no arquivo.")

        phase_two_prompt = f"Aja como {expert_title}, um expert no tópico, e forneça uma resposta completa e bem formatada para a seguinte pergunta: {user_input}"
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

def refine_response(expert_title: str, phase_two_response: str, user_input: str, model_name: str, temperature: float) -> str:
    # Função que refina a resposta do assistente
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Variável de ambiente GROQ_API_KEY não definida.")

        client = Groq(api_key=groq_api_key)

        def get_completion(prompt: str) -> str:
            # Função auxiliar que obtém a resposta do modelo com base em um prompt
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=get_max_tokens(model_name),
                top_p=1,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content

        refine_prompt = f"Aja como {expert_title}, um expert no tópico. Aqui está a resposta original para a pergunta '{user_input}': {phase_two_response}\n\nPor favor, revise e refine completamente esta resposta, fazendo melhorias e abordando quaisquer deficiências. Retorne uma versão atualizada da resposta que incorpore seus refinamentos."
        
        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

# Carrega as opções de agentes do arquivo
agent_options = load_agent_options()

# Define o título da página
st.title("Groqbot Experts")
st.write("Digite sua solicitação para ser abordada pelo expert ideal.")

# Divide a página em duas colunas
col1, col2 = st.columns(2)

with col1:
    # Área de texto para o usuário inserir sua solicitação
    user_input = st.text_area("Por favor, insira sua solicitação:", "", key="user_input")
    
    # Caixa de seleção para escolher um expert
    agent_selection = st.selectbox("Escolha um Expert", options=agent_options, index=0, key="agent_selection")
    
    # Caixa de seleção para escolher um modelo
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="model_name")
    
    # Controle deslizante para definir o nível de criatividade
    temperature = st.slider("Nível de Criativ
