from datetime import datetime, timedelta
from langchain_core.tools import tool

from db.models import Mensagem, Conversa
from db.session import obter_session

LIMITE_MENSAGENS = 150
ORCAMENTO_CARACTERES = 6000


def consultar_historico(cliente_id: int) -> str:
    janela_data = datetime.now() - timedelta(days=30)

    with obter_session() as session:
        mensagens_recentes = (
            session.query(Mensagem)
            .join(Conversa)
            .filter(Conversa.cliente_id == cliente_id)
            .filter(Mensagem.enviada_em >= janela_data)
            .order_by(Mensagem.enviada_em.desc())
            .limit(LIMITE_MENSAGENS)
            .all()
        )

        if not mensagens_recentes:
            return "Nenhum histórico de conversa anterior encontrado nos últimos 30 dias."

        linhas = []
        total_caracteres = 0
        for msg in mensagens_recentes:
            linha = f"[{msg.enviada_em.strftime('%d/%m %H:%M')}] {msg.remetente}: {msg.conteudo}\n"
            if linhas and total_caracteres + len(linha) > ORCAMENTO_CARACTERES:
                break
            linhas.append(linha)
            total_caracteres += len(linha)

        linhas.reverse()
        return "".join(linhas)


@tool
def buscar_historico_anterior(cliente_id: int) -> str:
    """Busca o histórico de conversas anteriores deste cliente, quando ele parece estar retomando um assunto já discutido antes (ex: 'e como ficou meu problema?', 'vocês resolveram aquilo?')."""
    return consultar_historico(cliente_id)