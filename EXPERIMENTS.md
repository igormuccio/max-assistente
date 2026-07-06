# Experimentos técnicos — Retrieval e Alucinação

Este documento registra uma investigação prática sobre os parâmetros centrais do pipeline de RAG do Max: `chunk_size`, `k` (top-k retrieval), `score_threshold` e o comportamento de alucinação do modelo mesmo com contexto correto disponível. O objetivo foi entender, com testes reais, os trade-offs de cada decisão — não apenas usar valores padrão.

## Índice

- [1. Por que RAG neste projeto](#1-por-que-rag-neste-projeto)
- [2. `chunk_size`: calibrando pelo conteúdo da base de conhecimento](#2-chunk_size-calibrando-pelo-conteúdo-da-base-de-conhecimento)
- [3. `k` (top-k retrieval): por que ele mascarava o problema](#3-k-top-k-retrieval-por-que-ele-mascarava-o-problema)
- [4. Alucinação por combinação de fatos legítimos](#4-alucinação-por-combinação-de-fatos-legítimos)
- [5. `score_threshold`: filtrando por relevância em vez de um `k` fixo](#5-score_threshold-filtrando-por-relevância-em-vez-de-um-k-fixo)
- [6. Bug de marcador de controle confundido com linguagem natural](#6-bug-de-marcador-de-controle-confundido-com-linguagem-natural)
- [7. Conclusões gerais](#7-conclusões-gerais)
- [8. Próximos passos identificados](#8-próximos-passos-identificados-não-implementados-ainda)

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

## 5. `score_threshold`: filtrando por relevância em vez de um `k` fixo

Como identificado na seção anterior, um `k` fixo sempre retorna o mesmo número de chunks, mesmo quando nem todos são relevantes. A alternativa testada foi o `search_type='similarity_score_threshold'` do LangChain, que descarta qualquer chunk abaixo de um limiar mínimo de relevância, usando `k` apenas como teto máximo.

**Observação técnica importante:** o FAISS, por padrão, mede distância L2 (onde menor = mais parecido), enquanto o `score_threshold` do LangChain espera um score de relevância normalizado entre 0 e 1 (onde maior = mais relevante). O LangChain faz essa conversão internamente — o valor de threshold configurado deve ser pensado nessa segunda escala.

**Metodologia de calibração:** em vez de escolher um valor por estimativa, testei o score de relevância retornado para quatro tipos de pergunta:

| Tipo de pergunta | Exemplo | Score mais alto observado |
|---|---|---|
| Específica e relevante | "prazo de entrega para o sul" | 0.85 |
| Difusa mas relevante | "meu pedido foi extraviado" | 0.72–0.77 |
| Fora do domínio | "copa do mundo fifa" | 0.63–0.66 |
| Fora do domínio | "como fazer miojo" | ~0.60 |

**Resultado:** existe uma margem de separação real (cerca de 6 a 12 pontos percentuais) entre o pior caso relevante (~0.72) e o pior caso fora do domínio (~0.66). Com base nisso, o threshold foi calibrado em `0.68` — posicionado dentro dessa margem, testado e replicado em múltiplas execuções com as mesmas perguntas.

**Configuração final:**
```python
retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.68, 'k': 4}
)
```

**Limitação identificada:** o threshold é um valor fixo calibrado empiricamente com um conjunto pequeno de perguntas de teste, não uma constante matemática. Ele reflete um trade-off consciente entre dois erros possíveis — deixar passar uma pergunta fora do domínio ou cortar contexto relevante em perguntas mais amplas —, não uma "resposta certa" universal. Em produção, um conjunto de teste maior (um eval set mais robusto) seria necessário para validar esse valor com mais confiança.

## 6. Bug de marcador de controle confundido com linguagem natural

Ao testar o `score_threshold` com uma pergunta fora do domínio (sem nenhum chunk retornado), o modelo deveria responder com o marcador de controle `TRANSFERIR_HUMANO`, definido no *system prompt*, para acionar a transferência para um atendente humano.

**Resultado observado:** o modelo gerou `TRANSFIRIR_HUMANO` (com erro de grafia — "transfIrir" em vez de "transfErir"). Como a checagem no código (`if 'TRANSFERIR_HUMANO' in reply`) busca a string exata, a condição não foi satisfeita, e o fluxo de transferência não foi acionado.

**Causa raiz:** o próprio *system prompt* usa, em outras regras, o verbo "transfira" (imperativo correto de "transferir", com "i"). O modelo aparentemente generalizou esse padrão de conjugação por cima do marcador de controle, que deveria ser reproduzido literalmente, e não interpretado como parte do texto em português.

**Correção aplicada:**
- Substituição do marcador por um token que não se pareça com uma palavra natural do idioma: `###TRANSFER_HUMANO###`.
- Checagem no código tornada mais tolerante a variações, verificando apenas o núcleo do token em maiúsculas: `if 'TRANSFER_HUMANO' in reply.upper()`.

**Conclusão:** confiar na reprodução exata de uma palavra-chave de controle por um LLM é frágil, especialmente quando essa palavra se assemelha a vocabulário comum do idioma usado no restante do prompt. Marcadores de controle devem ser visualmente distintos de linguagem natural, e a validação no código deve ser tolerante a pequenas variações de grafia.

## 7. Conclusões gerais

- RAG reduz alucinação, mas não a elimina — mesmo com contexto correto recuperado, o modelo pode combinar fatos legítimos de formas não autorizadas pelo negócio.
- Instruções em linguagem natural no *system prompt* têm um teto de eficácia: proibições, checagens explícitas e restrições literais foram testadas e nenhuma bloqueou o comportamento por completo.
- `chunk_size` e `k` não devem ser avaliados isoladamente — o tamanho da base de conhecimento determina se os efeitos de cada um ficam visíveis ou escondidos.
- Um `score_threshold` calibrado com dados reais é mais robusto que um `k` fixo, mas ainda depende de uma escolha de engenharia dentro de uma margem, não de um valor absoluto.
- Marcadores de controle (tokens especiais usados para acionar lógica no código) precisam ser distintos de linguagem natural, e a validação correspondente no código deve tolerar variações — nenhuma reprodução de texto por um LLM deve ser considerada 100% garantida.

## 8. Próximos passos identificados (não implementados ainda)

- **Grounding verification / self-checking:** uma segunda chamada ao modelo (ou validação em código) para verificar se a resposta usa alguma informação que não está literalmente no contexto, antes de exibi-la ao usuário.
- **Few-shot prompting:** incluir no *system prompt* um exemplo concreto de pergunta ambígua com a resposta correta esperada, em vez de apenas descrever a regra de forma abstrata.
- **Cobertura de conteúdo:** adicionar uma regra explícita para o cenário de "atraso simples" na base de conhecimento, eliminando a lacuna que hoje força o modelo a inferir.
- **Eval set mais robusto:** ampliar o conjunto de perguntas de teste usado para calibrar `score_threshold`, cobrindo mais variações de pergunta específica, difusa e fora do domínio.
- **Persistência do índice FAISS:** salvar o índice em disco em vez de recriá-lo a cada execução, reduzindo custo de reprocessamento e eliminando qualquer variável de recomputação de embeddings entre execuções.
