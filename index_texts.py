import os
import streamlit as st
from datetime import datetime

# LangChain (versões recentes)
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# =========================
# CONFIGURAÇÕES BÁSICAS
# =========================
st.set_page_config(page_title="Moz Legal Agent", layout="centered")
st.title("Moz Legal Agent")
st.markdown("Um assistente para apoiar advogados e legisladores com base nas leis de Moçambique.")

with st.expander("Sobre este agente"):
    st.markdown("""
    - O agente usa um modelo de linguagem com base somente nos documentos carregados (leis de Moçambique).
    - As respostas são baseadas em trechos recuperados do índice.
    - Importante: não substitui aconselhamento jurídico profissional.
    """)

# =========================
# OPENAI API KEY
# =========================
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("A chave da OpenAI (OPENAI_API_KEY) não foi definida.")
    st.stop()

# =========================
# CARREGAR INDEX FAISS
# =========================

embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

vector_db = FAISS.load_local(
    "legal_index_mz",
    embeddings,
    allow_dangerous_deserialization=True
)

# =========================
# RETRIEVER ROBUSTO (FAISS MMR + BM25)
# =========================
# FAISS com MMR para diversidade de trechos
faiss_retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.7}
)

# BM25 lexical: idealmente construído a partir de TODOS os chunks do corpus.
# Como fallback simples (sem reabrir corpus bruto aqui), usamos uma semente de docs recuperados.
seed_queries = [
    "Artigo", "Lei", "Constituição", "Código Penal", "Código Civil", "Lei do Trabalho",
    "CAPÍTULO", "TÍTULO", "SECÇÃO"
]
seed_docs = []
seen_ids = set()
for q in seed_queries:
    hits = vector_db.similarity_search(q, k=200)
    for d in hits:
        # Evitar duplicar objetos idênticos; usamos hash do conteúdo+filename
        key = (d.page_content[:200], d.metadata.get("filename", ""))
        if key not in seen_ids:
            seed_docs.append(d)
            seen_ids.add(key)

# Se nada foi semeado (índice muito pequeno), BM25 sobre uma consulta genérica
if not seed_docs:
    seed_docs = vector_db.similarity_search("Lei", k=200)

bm25 = BM25Retriever.from_documents(seed_docs)
bm25.k = 8

# Ensemble: combina semântico (FAISS) e lexical (BM25)
retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25],
    weights=[0.6, 0.4]
)

# =========================
# LLM E PROMPT
# =========================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

prompt_template = """
Você é um assistente jurídico baseado exclusivamente nos documentos carregados (leis de Moçambique).
Use o contexto recuperado para responder. Quando possível, indique o número do artigo e o nome da lei,
se constarem nos trechos. Se o contexto não trouxer base suficiente, responda:
"Não encontrei fundamento no material carregado."

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# =========================
# ESTADO DA SESSÃO
# =========================
if "query" not in st.session_state:
    st.session_state.query = ""
if "history" not in st.session_state:
    # cada item: {"pergunta": str, "resposta": str, "fontes": list}
    st.session_state.history = []

# =========================
# AÇÃO DE CONSULTA
# =========================
def responder():
    q = st.session_state.query.strip()
    if not q:
        return
    with st.spinner("Analisando a base legal..."):
        result = qa_chain.invoke({"query": q})
    st.session_state.history.append({
        "pergunta": q,
        "resposta": result["result"],
        "fontes": result["source_documents"]
    })
    st.session_state.query = ""

# =========================
# HISTÓRICO (último no topo)
# =========================
if st.session_state.history:
    for idx, item in enumerate(reversed(st.session_state.history), 1):
        st.markdown("**Pergunta:**")
        st.info(item["pergunta"])

        st.markdown("**Resposta:**")
        st.success(item["resposta"])

        # Botão de download por item
        resposta_txt = f"Pergunta: {item['pergunta']}\n\nResposta:\n{item['resposta']}\n\nFontes:\n"
        for doc in item["fontes"]:
            resposta_txt += f"\nArquivo: {doc.metadata.get('filename', 'desconhecido')}\n"
            resposta_txt += doc.page_content[:500] + "\n---\n"

        st.download_button(
            label="Baixar resposta como .txt",
            data=resposta_txt,
            file_name=f"resposta_legal_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key=f"download_{idx}"
        )

        st.markdown("**Fontes utilizadas:**")
        for doc in item["fontes"]:
            st.markdown(f"- Arquivo: `{doc.metadata.get('filename', 'desconhecido')}`")
            st.text(doc.page_content[:500])
        st.divider()

# =========================
# DEBUG (opcional): ver o que está a ser recuperado
# =========================
with st.expander("Debug: documentos recuperados"):
    q_preview = st.text_input("Consulta de teste (debug)", value="Artigo 7")
    if st.button("Testar recuperação"):
        st.write("Top-5 FAISS (similarity_search_with_score):")
        try:
            hits = vector_db.similarity_search_with_score(q_preview, k=5)
            for d, score in hits:
                st.write({"filename": d.metadata.get("filename"), "score": float(score)})
                st.text(d.page_content[:400])
                st.markdown("---")
        except Exception as e:
            st.write(f"Falha no similarity_search_with_score: {e}")

        st.write("Top-5 BM25:")
        try:
            bm_hits = bm25.get_relevant_documents(q_preview)
            for d in bm_hits[:5]:
                st.write({"filename": d.metadata.get("filename")})
                st.text(d.page_content[:400])
                st.markdown("---")
        except Exception as e:
            st.write(f"Falha no BM25: {e}")

# =========================
# INPUT NO FINAL (move “para baixo” após cada envio)
# =========================
st.subheader("Faça sua pergunta:")
st.text_input(
    "Exemplo: 'Quais são os direitos do trabalhador segundo a Lei do Trabalho?'",
    key="query",
    on_change=responder
)
