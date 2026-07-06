import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def carregar_prompt():
    with open(os.path.join(BASE_DIR, 'prompts', 'system.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def carregar_base_conhecimento():
    loader = TextLoader(os.path.join(BASE_DIR, 'data', 'politicas.txt'), encoding='utf-8')
    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documentos)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

def buscar_contexto(retriever, pergunta):
    docs = retriever.invoke(pergunta)
    return '\n'.join([doc.page_content for doc in docs])

def main():
    print('Carregando Max...')
    system_prompt = carregar_prompt()
    retriever = carregar_base_conhecimento()

    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.3,
        streaming=True
    )

    messages = [SystemMessage(content=system_prompt)]

    print('Max: Olá! Sou o Max, assistente da XYZ Entregas. Como posso ajudar?')

    while True:
        pergunta = input('Você: ')
        if pergunta.lower() == 'sair':
            print('Max: Até mais!')
            break

        contexto = buscar_contexto(retriever, pergunta)
        mensagem_com_contexto = f'{pergunta}\n\nInformações relevantes:\n{contexto}'

        messages.append(HumanMessage(content=mensagem_com_contexto))

        print('Max: ', end='', flush=True)
        reply = ''
        for chunk in llm.stream(messages):
            texto = chunk.content
            if texto:
                reply += texto

        if 'TRANSFERIR_HUMANO' in reply:
            print('Aguarde, vou transferir para um atendente.')
            print('[Sistema]: Transferindo...')
            break

        print(reply)
        print()
        messages.append(AIMessage(content=reply))

if __name__ == '__main__':
    main()