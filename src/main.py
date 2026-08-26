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

from inicializacao import carregar_prompt, carregar_base_conhecimento, carregar_indice_saudacoes, carregar_indice_saida
from busca_semantica import buscar_contexto, eh_saudacao, eh_intencao_saida
from verificacao_llm import verificar_grounding, verificar_informacao_suficiente
from db.models import Cliente, Conversa, Mensagem
from db.session import obter_session
from tools import todas_as_tools
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

load_dotenv()


def identificar_cliente(session):
    email = input('[Simulando login do site] Email do cliente: ').strip()

    cliente = session.query(Cliente).filter_by(email=email).first()

    if cliente:
        print(f'[Sistema]: Bem-vindo de volta, {cliente.nome}!')
    else:
        nome = input('[Sistema]: Cliente novo. Qual o seu nome? ').strip()
        cliente = Cliente(nome=nome, email=email)
        session.add(cliente)
        session.commit()
        print(f'[Sistema]: Cadastro criado, {cliente.nome}!')

    return cliente


def main():
    print('Carregando Max...')
    system_prompt = carregar_prompt()
    retriever = carregar_base_conhecimento()
    vectorstore_saudacoes = carregar_indice_saudacoes()
    vectorstore_saida = carregar_indice_saida()

    llm_chat = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.3,
        streaming=True
    )

    llm_chat_com_tools = llm_chat.bind_tools(todas_as_tools)

    llm_verificador = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0
    )

    with obter_session() as session:
        cliente = identificar_cliente(session)
        conversa = Conversa(cliente_id=cliente.id)
        session.add(conversa)
        session.commit()

        messages = [SystemMessage(content=system_prompt)]
        tentativas_sem_contexto = 0

        print('Max: Olá! Sou o Max, assistente da XYZ Entregas. Como posso ajudar?')

        while True:
            pergunta = input('Você: ')
            if eh_intencao_saida(vectorstore_saida, pergunta):
                print('Max: Até mais!')
                break

            if eh_saudacao(vectorstore_saudacoes, pergunta):
                print('Max: Olá! Como posso te ajudar hoje?')
                print()
                continue

            if verificar_informacao_suficiente(llm_verificador, pergunta):
                print('Max: Para te ajudar melhor, preciso de mais alguns detalhes. Você pode informar sua região e, se possível, há quantos dias fez o pedido?')
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

            mensagem_cliente = Mensagem(conversa_id=conversa.id, remetente='cliente', conteudo=pergunta)
            session.add(mensagem_cliente)
            session.commit()

            mensagem_com_contexto = f'{pergunta}\n\nInformações relevantes:\n{contexto}\n\nO cliente_id do cliente atual é {cliente.id}.'
            messages.append(HumanMessage(content=mensagem_com_contexto))

            resposta_decisao = llm_chat_com_tools.invoke(messages)

            if resposta_decisao.tool_calls:
                messages.append(resposta_decisao)

                for chamada in resposta_decisao.tool_calls:
                    tool_chamada = next(t for t in todas_as_tools if t.name == chamada['name'])
                    resultado = tool_chamada.invoke(chamada['args'])
                    messages.append(ToolMessage(content=resultado, tool_call_id=chamada['id']))

            print('Max: ', end='', flush=True)
            reply = ''
            for chunk in llm_chat.stream(messages):
                texto = chunk.content
                if texto:
                    reply += texto

            if 'TRANSFER_HUMANO' in reply.upper():
                print('Aguarde, vou transferir para um atendente.')
                print('[Sistema]: Transferindo...')
                break

            grounding_falhou = verificar_grounding(llm_verificador, pergunta, contexto, reply)

            if grounding_falhou:
                print('Max: Não tenho essa informação específica no momento, vou te transferir para um atendente humano que pode te ajudar melhor.')
                print('[Sistema]: Transferindo...')
                break

            print(reply)
            print()
            messages.append(AIMessage(content=reply))

            mensagem_max = Mensagem(conversa_id=conversa.id, remetente='max', conteudo=reply)
            session.add(mensagem_max)
            session.commit()


if __name__ == '__main__':
    main()