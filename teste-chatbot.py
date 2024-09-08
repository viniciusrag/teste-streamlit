import streamlit as st
from openai import OpenAI
import tiktoken
import io
import os
import PyPDF2
import json

from datetime import datetime

# Função para contar tokens
def count_tokens(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Cada mensagem tem um custo fixo de 4 tokens
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":  # Se houver um nome, adiciona mais 1 token
                num_tokens += 1
    num_tokens += 2  # Uma mensagem de sistema é adicionada implicitamente
    return num_tokens

# Função para limpar o chat
def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar você hoje?"}]
    st.session_state.token_count = {model: 0 for model in ["gpt-3.5-turbo", "gpt-4o-mini"]}
    st.session_state.file_content = None

# Função para ler o conteúdo do arquivo
def read_file_content(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return read_pdf(uploaded_file)
    elif uploaded_file.type.startswith("text/"):
        return uploaded_file.getvalue().decode("utf-8")
    else:
        return "Formato de arquivo não suportado."

# Função para ler PDF
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# New function to save conversation
def save_conversation(messages, model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    data = {
        "timestamp": timestamp,
        "model": model,
        "messages": messages
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    return filename

# New function to load conversation
def load_conversation(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data["messages"], data["model"]

# Modified clear_chat function
def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar você hoje?"}]
    st.session_state.token_count = {model: 0 for model in ["gpt-3.5-turbo", "gpt-4o-mini"]}
    st.session_state.file_content = None
    st.session_state.conversation_filename = None

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar você hoje?"}]
if "token_count" not in st.session_state:
    st.session_state.token_count = {
        "gpt-4": 0,
        "gpt-4o": 0,
        "gpt-3.5-turbo": 0,
        "gpt-4o-mini": 0
    }
if "total_token_count" not in st.session_state:
    st.session_state.total_token_count = {
        "gpt-4": 0,
        "gpt-4o": 0,
        "gpt-3.5-turbo": 0,
        "gpt-4o-mini": 0
    }
if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "conversation_filename" not in st.session_state:
    st.session_state.conversation_filename = None

# Sidebar configuration
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    
    model_option = st.selectbox(
        "Escolha o modelo da OpenAI:",
        ("gpt-3.5-turbo", "gpt-4o-mini")
    )
    
    uploaded_file = st.file_uploader("Faça upload de um arquivo para análise", type=["txt", "pdf"])
    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file)
        st.session_state.file_content = file_content
        st.success("Arquivo carregado com sucesso!")
    
    # Save conversation button
    if st.button("Salvar Conversa"):
        if st.session_state.messages:
            filename = save_conversation(st.session_state.messages, model_option)
            st.success(f"Conversa salva como {filename}")
    
    # Load conversation selectbox
    saved_conversations = [f for f in os.listdir() if f.startswith("conversation_") and f.endswith(".json")]
    if saved_conversations:
        selected_conversation = st.selectbox("Carregar Conversa", [""] + saved_conversations)
        if selected_conversation:
            loaded_messages, loaded_model = load_conversation(selected_conversation)
            st.session_state.messages = loaded_messages
            st.session_state.conversation_filename = selected_conversation
            st.success(f"Conversa carregada de {selected_conversation}")
    
    # Clear chat button
    if st.button("Limpar Chat"):
        clear_chat()
    
    # Token counters display
    st.markdown("### Contadores de Tokens")
    
    st.markdown("#### Sessão Atual")
    for model, count in st.session_state.token_count.items():
        st.write(f"{model}: {count}")
    
    st.markdown("#### Total Acumulado")
    for model, count in st.session_state.total_token_count.items():
        st.write(f"{model}: {count}")

st.title("💬 Chatbot - Método RGT")

# Display conversation filename if loaded
if st.session_state.conversation_filename:
    st.info(f"Conversa carregada: {st.session_state.conversation_filename}")

# Display message history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input and processing
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Por favor, adicione sua chave de API da OpenAI para continuar.")
        st.stop()

    try:
        client = OpenAI(api_key=openai_api_key)
        
        if st.session_state.file_content:
            prompt = f"Arquivo carregado:\n\n{st.session_state.file_content}\n\nPergunta do usuário: {prompt}"
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        response = client.chat.completions.create(
            model=model_option,
            messages=st.session_state.messages
        )
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

        # Update token counters
        tokens_used = count_tokens(st.session_state.messages, model_option)
        st.session_state.token_count[model_option] = tokens_used
        st.session_state.total_token_count[model_option] += tokens_used

        # Update sidebar token counters
        st.sidebar.markdown("### Contadores de Tokens")
        
        st.sidebar.markdown("#### Sessão Atual")
        for model, count in st.session_state.token_count.items():
            st.sidebar.write(f"{model}: {count}")
        
        st.sidebar.markdown("#### Total Acumulado")
        for model, count in st.session_state.total_token_count.items():
            st.sidebar.write(f"{model}: {count}")

    except Exception as e:
        st.error(f"Ocorreu um erro: {str(e)}")