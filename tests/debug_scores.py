# Debug: inspeciona scores de similaridade dos chunks retornados pelo retriever
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from inicializacao import carregar_base_conhecimento

retriever = carregar_base_conhecimento()

docs_com_score = retriever.vectorstore.similarity_search_with_relevance_scores("meu pedido foi extraviado", k=4)

for i, (doc, score) in enumerate(docs_com_score, start=1):
    print(f"\n--- Chunk {i} | score={score:.4f} ---")
    print(doc.page_content)