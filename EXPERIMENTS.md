# Experimentos técnicos — Retrieval e Alucinação

Este documento registra uma investigação prática sobre três parâmetros centrais do pipeline de RAG do Max: `chunk_size`, `k` (top-k retrieval) e o comportamento de alucinação do modelo mesmo com contexto correto disponível. O objetivo foi entender, com testes reais, os trade-offs de cada decisão — não apenas usar valores padrão.

## 1. Por que RAG neste projeto

O Max responde perguntas sobre políticas de uma empresa fictícia de entregas. Um LLM genérico não tem conhecimento sobre essas políticas — sem RAG, ele responderia com base em suposições (alucinação) ou se recusaria a responder. RAG resolve isso buscando, a cada pergunta, apenas o trecho relevante da base de conhecimento e injetando esse trecho no prompt, em vez de:

- fazer fine-tuning (caro, lento, precisa retreinar a cada mudança de política);
- ou colar a base inteira no prompt (caro em tokens, e o modelo perde precisão com excesso de informação irrelevante).

## 2. `chunk_size`: calibrando pelo conteúdo da base de conhecimento

**Teste:** medi o tamanho real de cada bloco de política no arquivo de conhecimento (~150–200 caracteres por regra) e comparei com o `chunk_size=500` usado inicialmente.

**Resultado:** com 500, os chunks gerados (400–436 caracteres) misturavam 2–3 tópicos distintos em um único chunk — por exemplo, dados institucionais, prazos de entrega e política de reembolso no mesmo bloco. Reduzindo para `chunk_size=200`, os chunks passaram a corresponder a uma única regra de negócio por vez.

**Conclusão:** `chunk_size` deveria ser definido a partir do tamanho natural das unidades de sentido do conteúdo, não copiado de um exemplo genérico. Chunks grandes demais geram contexto ruidoso (informação irrelevante misturada); chunks pequenos demais podem fragmentar uma regra no meio — um bloco de política com mais de 200 caracteres, por exemplo, é dividido em dois chunks distintos, separando uma condição da sua consequência.

## 3. `k` (top-k retrieval): por que ele mascarava o problema

`.as_retriever()` sem parâmetros usa um valor padrão do LangChain, `k=4` — não declarado explicitamente em nenhum lugar do código-fonte original.

**Observação:** como a base de conhecimento deste projeto tem apenas ~6–7 blocos de política, `k=4` recuperava quase a base inteira em qualquer pergunta. Isso mascarava a fragmentação causada por `chunk_size`: mesmo quando um chunk relevante vinha cortado, a informação faltante costumava aparecer em outro chunk vizinho, também recuperado.

**Teste com `k=2`:** reduzindo o valor, ficou mais fácil observar quando um chunk relevante ficava de fora da resposta.

**Conclusão:** o efeito de `chunk_size` e `k` é interdependente, e o tamanho da base de conhecimento determina se um problema fica visível ou escondido. Um `chunk_size` fragmentado combinado com um `k` proporcionalmente baixo em uma base grande (milhares de documentos) deixaria muito mais informação relevante de fora do que em uma base pequena como esta.

## 4. Alucinação por combinação de fatos legítimos

**Pergunta de teste:**
> "Meu pedido está atrasado só um pouco, ainda não chegou mas também não sumiu, o que eu faço?"

Esse cenário — atraso simples, sem extravio — não está coberto explicitamente na base de conhecimento, que só define regras para "extravio" e para "pedido que consta como entregue mas não recebido".

**Resultado:** o modelo combinou dois fatos reais (prazos de entrega por região + regra de pedido não recebido) para gerar uma recomendação plausível ("aguarde mais um pouco"), que não está escrita em nenhum lugar da base. Isso persistiu através de três formulações diferentes de instrução no *system prompt* — proibição direta, checagem explícita de "posso responder isso?", e restrição literal contra combinar informações de contextos diferentes — e também com `temperature=0`.

**Causa raiz:** o modelo não estava inventando um fato aleatório; estava fazendo uma inferência lógica sobre fatos reais, algo que ele não classifica como "invenção". Reduzir a temperatura não resolve, porque temperatura controla aleatoriedade na escolha de palavras, não a capacidade do modelo de conectar fatos e concluir algo a partir deles.

## 5. Conclusões gerais

- RAG reduz alucinação, mas não a elimina — mesmo com contexto correto recuperado, o modelo pode combinar fatos legítimos de formas não autorizadas pelo negócio.
- Instruções em linguagem natural no *system prompt* têm um teto de eficácia: proibições, checagens explícitas e restrições literais foram testadas e nenhuma bloqueou o comportamento por completo.
- `chunk_size` e `k` não devem ser avaliados isoladamente — o tamanho da base de conhecimento determina se os efeitos de cada um ficam visíveis ou escondidos.

## 6. Próximos passos identificados (não implementados ainda)

- **Grounding verification / self-checking:** uma segunda chamada ao modelo (ou validação em código) para verificar se a resposta usa alguma informação que não está literalmente no contexto, antes de exibi-la ao usuário.
- **Few-shot prompting:** incluir no *system prompt* um exemplo concreto de pergunta ambígua com a resposta correta esperada, em vez de apenas descrever a regra de forma abstrata.
- **Cobertura de conteúdo:** adicionar uma regra explícita para o cenário de "atraso simples" na base de conhecimento, eliminando a lacuna que hoje força o modelo a inferir.
- **`score_threshold` no retriever:** usar um limiar de similaridade em vez de um `k` fixo, para que o número de chunks recuperados varie conforme a relevância real, e não um teto arbitrário.
