import streamlit as st
import os
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ==== CONFIGURAR CHAVE ====

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("A chave da OpenAI (OPENAI_API_KEY) não foi definida.")
st.stop()

# ==== BANCO VECTORIAL ====

vector_db = FAISS.load_local(
"legal_index_mz",
OpenAIEmbeddings(openai_api_key=openai_api_key),
allow_dangerous_deserialization=True
)

# ==== PROMPT DE CONTEXTO ====

prompt_template = """
Você é um assistente jurídico treinado exclusivamente com base nas leis de Moçambique.

Use SOMENTE o contexto abaixo para responder à pergunta.
Se a resposta não estiver no contexto, diga: "Não sei."

Contexto:
{context}

Pergunta:
{question}
"""

prompt = PromptTemplate(
input_variables=["context", "question"],
template=prompt_template,
)

retriever = vector_db.as_retriever()
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

qa_chain = RetrievalQA.from_chain_type(
llm=llm,
retriever=retriever,
return_source_documents=True,
chain_type_kwargs={"prompt": prompt}
)

# ==== STREAMLIT CONFIG ====

st.set_page_config(page_title="Moz Legal Agent", layout="centered")
st.title("Moz Legal Agent")
st.markdown("Um assistente inteligente para apoiar advogados e legisladores com base nas leis de Moçambique.")

with st.expander("Sobre este agente"):
    st.markdown("""
- Este agente usa um modelo de linguagem (IA) com base apenas nas leis de Moçambique.
- As respostas são baseadas em documentos legais indexados como a Constituição, o Código Penal e a Lei do Trabalho.
- Importante: Ele não substitui um advogado, apenas auxilia na interpretação textual da legislação.
""")

# ==== CONTROLE DE ESTADO ====

if "query" not in st.session_state:
    st.session_state.query = ""
if "result" not in st.session_state:
    st.session_state.result = None

# ==== FUNÇÃO DE CONSULTA ====

def responder():
    if st.session_state.query.strip() != "":
        with st.spinner("Analisando a base legal..."):
            st.session_state.result = qa_chain.invoke({"query": st.session_state.query})
st.session_state.query = ""

# ==== INTERFACE PRINCIPAL ====

st.subheader("Faça sua pergunta:")

st.text_input(
"Exemplo: 'Quais são os direitos do trabalhador segundo a Lei do Trabalho?'",
key="query",
on_change=responder
)

# ==== MOSTRAR RESULTADO ====

if st.session_state.result:
    resposta = st.session_state.result

st.markdown("### Resposta:")  
st.success(resposta["result"])  

# Criar conteúdo para download  
resposta_txt = f"Pergunta: {resposta.get('query', '')}\n\nResposta:\n{resposta['result']}\n\nFontes:\n"  
for doc in resposta["source_documents"]:  
    resposta_txt += f"\nArquivo: {doc.metadata.get('filename', 'desconhecido')}\n"  
    resposta_txt += doc.page_content[:500] + "\n---\n"  

st.download_button(  
    label="Baixar resposta como .txt",  
    data=resposta_txt,  
    file_name=f"resposta_legal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",  
    mime="text/plain"  
)  

st.markdown("### Fontes utilizadas:")  
for doc in resposta["source_documents"]:  
    st.markdown(f"**Arquivo:** `{doc.metadata.get('filename', 'desconhecido')}`")  
    st.text(doc.page_content[:500])  
    st.markdown("---")  


