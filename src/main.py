import os
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

logging.basicConfig(
    filename=os.path.join(BASE_DIR, 'logs', 'app.log'),
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.captureWarnings(True)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

load_dotenv()

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

def eh_saudacao(vectorstore_saudacoes, pergunta):
    resultados = vectorstore_saudacoes.similarity_search_with_relevance_scores(pergunta, k=1)
    _, score = resultados[0]
    return score >= 0.85

def buscar_contexto(retriever, pergunta):
    docs = retriever.invoke(pergunta)
    return '\n'.join([doc.page_content for doc in docs])

def verificar_grounding(llm, contexto, resposta):
    prompt_verificacao = f"""Você é um verificador de fatos. Analise se a resposta abaixo usa APENAS informações presentes no contexto fornecido, sem inferências ou combinações não explícitas.

Contexto:
{contexto}

Resposta a verificar:
{resposta}

A resposta contém alguma afirmação, recomendação ou instrução que NÃO está literalmente escrita no contexto acima? Responda apenas SIM ou NÃO."""

    verificacao = llm.invoke(prompt_verificacao)
    return 'SIM' in verificacao.content.upper()

def main():
    print('Carregando Max...')
    system_prompt = carregar_prompt()
    retriever = carregar_base_conhecimento()
    vectorstore_saudacoes = carregar_indice_saudacoes()

    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.3,
        streaming=True
    )

    messages = [SystemMessage(content=system_prompt)]
    tentativas_sem_contexto = 0

    print('Max: Olá! Sou o Max, assistente da XYZ Entregas. Como posso ajudar?')

    while True:
        pergunta = input('Você: ')
        if pergunta.lower() == 'sair':
            print('Max: Até mais!')
            break

        if eh_saudacao(vectorstore_saudacoes, pergunta):
            print('Max: Olá! Como posso te ajudar hoje?')
            print()
            continue

        contexto = buscar_contexto(retriever, pergunta)

        if not contexto.strip():
            tentativas_sem_contexto += 1

            if tentativas_sem_contexto >= 2:
                print('Max: Não consegui entender sua solicitação. Vou te transferir para um atendente.')
                print('[Sistema]: Transferindo...')
                break

            print('Max: Não entendi muito bem sua pergunta. Você pode explicar de outra forma, com mais detalhes sobre seu pedido?')
            print()
            continue

        tentativas_sem_contexto = 0

        mensagem_com_contexto = f'{pergunta}\n\nInformações relevantes:\n{contexto}'
        messages.append(HumanMessage(content=mensagem_com_contexto))

        print('Max: ', end='', flush=True)
        reply = ''
        for chunk in llm.stream(messages):
            texto = chunk.content
            if texto:
                reply += texto

        if 'TRANSFER_HUMANO' in reply.upper():
            print('Aguarde, vou transferir para um atendente.')
            print('[Sistema]: Transferindo...')
            break

        grounding_falhou = verificar_grounding(llm, contexto, reply)

        if grounding_falhou:
            print('Max: Não tenho essa informação específica no momento, vou te transferir para um atendente humano que pode te ajudar melhor.')
            print('[Sistema]: Transferindo...')
            break

        print(reply)
        print()
        messages.append(AIMessage(content=reply))

if __name__ == '__main__':
    main()