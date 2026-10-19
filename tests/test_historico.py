import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from db.models import Cliente, Conversa, Mensagem
from db.session import obter_session
from tools.historico import buscar_historico_anterior, ORCAMENTO_CARACTERES


def criar_cliente_teste(session, email):
    cliente = Cliente(nome='Cliente Teste', email=email)
    session.add(cliente)
    session.commit()
    return cliente


def criar_conversa(session, cliente_id):
    conversa = Conversa(cliente_id=cliente_id)
    session.add(conversa)
    session.commit()
    return conversa


def inserir_mensagem(session, conversa_id, remetente, conteudo, enviada_em):
    mensagem = Mensagem(
        conversa_id=conversa_id,
        remetente=remetente,
        conteudo=conteudo,
        enviada_em=enviada_em,
    )
    session.add(mensagem)
    session.commit()
    return mensagem


def teste_sem_historico():
    print('--- Teste 1: cliente sem nenhum histórico ---')
    with obter_session() as session:
        cliente = criar_cliente_teste(session, f'teste_sem_historico_{datetime.now().timestamp()}@email.com')
        cliente_id = cliente.id

    resultado = buscar_historico_anterior.invoke({'cliente_id': cliente_id})
    esperado = 'Nenhum histórico de conversa anterior encontrado nos últimos 30 dias.'

    print(f'Esperado: {esperado}')
    print(f'Obtido:   {resultado}')
    print('PASSOU' if resultado == esperado else 'FALHOU')
    print()


def teste_janela_30_dias():
    print('--- Teste 2: janela de 30 dias ---')
    with obter_session() as session:
        cliente = criar_cliente_teste(session, f'teste_janela_{datetime.now().timestamp()}@email.com')
        conversa = criar_conversa(session, cliente.id)

        inserir_mensagem(session, conversa.id, 'cliente', 'mensagem dentro da janela', datetime.now() - timedelta(days=29))
        inserir_mensagem(session, conversa.id, 'cliente', 'mensagem fora da janela', datetime.now() - timedelta(days=31))

        cliente_id = cliente.id

    resultado = buscar_historico_anterior.invoke({'cliente_id': cliente_id})

    dentro_presente = 'dentro da janela' in resultado
    fora_ausente = 'fora da janela' not in resultado

    print(f'Mensagem dentro da janela presente: {dentro_presente}')
    print(f'Mensagem fora da janela ausente: {fora_ausente}')
    print('PASSOU' if dentro_presente and fora_ausente else 'FALHOU')
    print()


def teste_orcamento_caracteres():
    print('--- Teste 3: orçamento de caracteres ---')
    with obter_session() as session:
        cliente = criar_cliente_teste(session, f'teste_orcamento_{datetime.now().timestamp()}@email.com')
        conversa = criar_conversa(session, cliente.id)

        for i in range(100):
            inserir_mensagem(session, conversa.id, 'cliente', f'mensagem numero {i} ' + 'x' * 100, datetime.now() - timedelta(minutes=100 - i))

        cliente_id = cliente.id

    resultado = buscar_historico_anterior.invoke({'cliente_id': cliente_id})
    total_linhas = len([linha for linha in resultado.strip().split('\n') if linha.strip()])

    dentro_do_orcamento = len(resultado) <= ORCAMENTO_CARACTERES
    cortou_algo = 0 < total_linhas < 100
    contem_mais_recente = 'mensagem numero 99 ' in resultado
    nao_contem_mais_antiga = 'mensagem numero 0 ' not in resultado

    print(f'Caracteres retornados: {len(resultado)} (orçamento: {ORCAMENTO_CARACTERES})')
    print(f'Linhas retornadas: {total_linhas} (de 100 inseridas)')
    print(f'Dentro do orçamento: {dentro_do_orcamento}')
    print(f'Cortou parte das mensagens: {cortou_algo}')
    print(f'Contém a mais recente (numero 99): {contem_mais_recente}')
    print(f'NÃO contém a mais antiga (numero 0): {nao_contem_mais_antiga}')
    print('PASSOU' if (dentro_do_orcamento and cortou_algo and contem_mais_recente and nao_contem_mais_antiga) else 'FALHOU')
    print()


def teste_multiplas_conversas():
    print('--- Teste 4: histórico cruza múltiplas conversas do mesmo cliente ---')
    with obter_session() as session:
        cliente = criar_cliente_teste(session, f'teste_multiplas_conversas_{datetime.now().timestamp()}@email.com')

        conversa_antiga = criar_conversa(session, cliente.id)
        inserir_mensagem(session, conversa_antiga.id, 'cliente', 'mensagem da conversa antiga', datetime.now() - timedelta(days=5))

        conversa_nova = criar_conversa(session, cliente.id)
        inserir_mensagem(session, conversa_nova.id, 'cliente', 'mensagem da conversa nova', datetime.now())

        cliente_id = cliente.id

    resultado = buscar_historico_anterior.invoke({'cliente_id': cliente_id})

    contem_antiga = 'conversa antiga' in resultado
    contem_nova = 'conversa nova' in resultado

    print(f'Contém mensagem da conversa antiga: {contem_antiga}')
    print(f'Contém mensagem da conversa nova: {contem_nova}')
    print('PASSOU' if contem_antiga and contem_nova else 'FALHOU')
    print()


def teste_ordem_cronologica():
    print('--- Teste 5: mensagens retornadas em ordem cronológica ---')
    with obter_session() as session:
        cliente = criar_cliente_teste(session, f'teste_ordem_{datetime.now().timestamp()}@email.com')
        conversa = criar_conversa(session, cliente.id)

        inserir_mensagem(session, conversa.id, 'cliente', 'primeira mensagem', datetime.now() - timedelta(hours=2))
        inserir_mensagem(session, conversa.id, 'max', 'segunda mensagem', datetime.now() - timedelta(hours=1))
        inserir_mensagem(session, conversa.id, 'cliente', 'terceira mensagem', datetime.now())

        cliente_id = cliente.id

    resultado = buscar_historico_anterior.invoke({'cliente_id': cliente_id})

    posicao_primeira = resultado.find('primeira mensagem')
    posicao_segunda = resultado.find('segunda mensagem')
    posicao_terceira = resultado.find('terceira mensagem')

    ordem_correta = posicao_primeira < posicao_segunda < posicao_terceira

    print(f'Posições: primeira={posicao_primeira}, segunda={posicao_segunda}, terceira={posicao_terceira}')
    print('PASSOU' if ordem_correta else 'FALHOU')
    print()


if __name__ == '__main__':
    teste_sem_historico()
    teste_janela_30_dias()
    teste_orcamento_caracteres()
    teste_multiplas_conversas()
    teste_ordem_cronologica()