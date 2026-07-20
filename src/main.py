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

from inicializacao import carregar_prompt, carregar_base_conhecimento, carregar_indice_saudacoes
from busca_semantica import buscar_contexto, eh_saudacao
from verificacao_llm import verificar_grounding
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

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