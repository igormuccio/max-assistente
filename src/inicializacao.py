import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def carregar_prompt():
    with open(os.path.join(BASE_DIR, 'prompts', 'system.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def carregar_base_conhecimento():
    caminho_politicas = os.path.join(BASE_DIR, 'data', 'politicas.txt')
    caminho_indice = os.path.join(BASE_DIR, 'data', 'faiss_index')
    caminho_metadata = os.path.join(BASE_DIR, 'data', 'faiss_metadata.txt')

    data_modificacao_atual = str(os.path.getmtime(caminho_politicas))
    embeddings = OpenAIEmbeddings()

    indice_existe = os.path.exists(caminho_indice)
    metadata_existe = os.path.exists(caminho_metadata)

    if indice_existe and metadata_existe:
        with open(caminho_metadata, 'r') as f:
            data_modificacao_salva = f.read().strip()

        if data_modificacao_salva == data_modificacao_atual:
            print('Índice já atualizado, carregando do disco...')
            vectorstore = FAISS.load_local(caminho_indice, embeddings, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={'score_threshold': 0.68, 'k': 4}
            )

    print('Recalculando embeddings...')
    loader = TextLoader(caminho_politicas, encoding='utf-8')
    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documentos)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(caminho_indice)
    with open(caminho_metadata, 'w') as f:
        f.write(data_modificacao_atual)

    return vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={'score_threshold': 0.68, 'k': 4}
    )

def carregar_indice_saudacoes():
    exemplos_saudacao = ['olá', 'oi', 'oii', 'bom dia', 'boa tarde', 'boa noite', 'tudo bem', 'e aí', 'opa', 'salve']
    documentos_saudacao = [Document(page_content=texto) for texto in exemplos_saudacao]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documentos_saudacao, embeddings)