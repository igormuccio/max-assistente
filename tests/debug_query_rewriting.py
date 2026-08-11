import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from verificacao_llm import reescrever_query, verificar_informacao_suficiente
from inicializacao import carregar_base_conhecimento

load_dotenv()

llm_verificador = ChatOpenAI(model='gpt-4o-mini', temperature=0)
retriever = carregar_base_conhecimento()

perguntas = [
    "cadê minha encomenda?",
    "sumiu meu troço, o que faço?",
    "não recebi minha compra ainda",
    "mano, cadê minhas paradas? já era pra ter chegado",
    "vc pode me ajudar? tipo, faz tempo q n chega naaada",
    "paguei por uma coisa que nunca vi na minha vida",
    "tô esperando uma coisa que parece que não existe mais",
    "meu pedido atrasou, aí eu liguei e não resolveram, agora quero saber se rola reembolso ou troca ou o que seja, já tentei de tudo",
    "e aí, cadê?",
    "nada ainda?",
]

def melhor_score(pergunta):
    resultados = retriever.vectorstore.similarity_search_with_relevance_scores(pergunta, k=4)
    return max(score for _, score in resultados) if resultados else None

for pergunta in perguntas:
    precisa_mais_info = verificar_informacao_suficiente(llm_verificador, pergunta)

    if precisa_mais_info:
        print("ORIGINAL:", pergunta)
        print("INTERCEPTADA em needs_more_information — nunca chegaria no retriever")
        print('---')
        continue

    reescrita = reescrever_query(llm_verificador, pergunta)
    score_original = melhor_score(pergunta)
    score_reescrita = melhor_score(reescrita)

    print("ORIGINAL: ", pergunta, f"(score: {score_original:.4f})")
    print("REESCRITA:", reescrita, f"(score: {score_reescrita:.4f})")
    print(f"DIFERENÇA: {score_reescrita - score_original:+.4f}")
    print('---')