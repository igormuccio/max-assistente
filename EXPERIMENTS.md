# Experimentos técnicos — Retrieval e Alucinação

Este documento registra uma investigação prática sobre os parâmetros centrais do pipeline de RAG do Max: `chunk_size`, `k` (top-k retrieval), `score_threshold` e o comportamento de alucinação do modelo mesmo com contexto correto disponível. O objetivo foi entender, com testes reais, os trade-offs de cada decisão — não apenas usar valores padrão.

## Índice

- [1. Por que RAG neste projeto](#1-por-que-rag-neste-projeto)
- [2. `chunk_size`: calibrando pelo conteúdo da base de conhecimento](#2-chunk_size-calibrando-pelo-conteúdo-da-base-de-conhecimento)
- [3. `k` (top-k retrieval): por que ele mascarava o problema](#3-k-top-k-retrieval-por-que-ele-mascarava-o-problema)
- [4. Alucinação por combinação de fatos legítimos](#4-alucinação-por-combinação-de-fatos-legítimos)
- [5. `score_threshold`: filtrando por relevância em vez de um `k` fixo](#5-score_threshold-filtrando-por-relevância-em-vez-de-um-k-fixo)
- [6. Limitação do `score_threshold`: transferência prematura sem contexto](#6-limitação-do-score_threshold-transferência-prematura-sem-contexto)
- [7. Bug de marcador de controle confundido com linguagem natural](#7-bug-de-marcador-de-controle-confundido-com-linguagem-natural)
- [8. Grounding verification: bloqueando inferências não fundamentadas](#8-grounding-verification-bloqueando-inferências-não-fundamentadas)
- [9. Persistência do índice FAISS: eliminando reprocessamento desnecessário](#9-persistência-do-índice-faiss-eliminando-reprocessamento-desnecessário)
- [10. Separando logs técnicos da interface do usuário](#10-separando-logs-técnicos-da-interface-do-usuário)
- [11. Detecção de saudação: evitando penalizar small talk](#11-detecção-de-saudação-evitando-penalizar-small-talk)
- [12. Conclusões gerais](#12-conclusões-gerais)
- [13. Próximos passos identificados](#13-próximos-passos-identificados-não-implementados-ainda)

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

**Nota:** o `k=2` foi usado aqui apenas como teste de diagnóstico, não como configuração final — o objetivo era tornar visível o efeito da fragmentação que `k=4` estava mascarando. Esse experimento evidenciou as limitações de depender só de um número fixo de documentos, motivando a adoção de um `score_threshold` (limiar de similaridade), descrito na seção seguinte.

## 4. Alucinação por combinação de fatos legítimos

**Pergunta de teste:**
> "Meu pedido está atrasado só um pouco, ainda não chegou mas também não sumiu, o que eu faço?"

Esse cenário — atraso simples, sem extravio — não está coberto explicitamente na base de conhecimento, que só define regras para "extravio" e para "pedido que consta como entregue mas não recebido".

**Resultado:** o modelo combinou dois fatos reais (prazos de entrega por região + regra de pedido não recebido) para gerar uma recomendação plausível ("aguarde mais um pouco"), que não está escrita em nenhum lugar da base. Isso persistiu através de três formulações diferentes de instrução no *system prompt* — proibição direta, checagem explícita de "posso responder isso?", e restrição literal contra combinar informações de contextos diferentes — e também com `temperature=0`.

**Causa raiz:** o modelo não estava inventando um fato aleatório; estava fazendo uma inferência lógica sobre fatos reais, algo que ele não classifica como "invenção". Reduzir a temperatura não resolve, porque temperatura controla aleatoriedade na escolha de palavras, não a capacidade do modelo de conectar fatos e concluir algo a partir deles.

**Observação posterior (validação cruzada):** após os experimentos do Max, um comportamento semelhante foi observado de forma independente em uma conversa com outro modelo (Claude Sonnet), fora do escopo direto deste projeto. Ao explicar por que a segunda chamada de verificação era "mais barata", o modelo combinou dois fatos reais e documentados (o custo do risco existe; o custo de API é comparativamente menor) para concluir que esse custo seria "desprezível" — uma quantificação que não era sustentada pelas premissas disponíveis. A observação reforça que a inferência por combinação de fatos verdadeiros não parece ser exclusiva do modelo utilizado no Max, tendo sido observada também em outro modelo de capacidade superior. Embora esse episódio isolado não permita generalizações sobre o comportamento de LLMs como um todo, ele é consistente com a hipótese investigada neste projeto: modelos podem produzir respostas plausíveis extrapolando o que está explicitamente fundamentado no contexto.

## 5. `score_threshold`: filtrando por relevância em vez de um `k` fixo

Como observado na investigação sobre `k` (Seção 3), um `k` fixo sempre retorna o mesmo número de chunks, mesmo quando nem todos são relevantes. A alternativa testada foi o `search_type='similarity_score_threshold'` do LangChain, que descarta qualquer chunk abaixo de um limiar mínimo de relevância, usando `k` apenas como teto máximo.

Com o `score_threshold` ativo, o papel do `k` muda: deixa de ser o principal filtro de relevância e passa a atuar apenas como teto máximo de chunks retornados, já que o threshold descarta antecipadamente qualquer chunk abaixo do limiar de relevância. Por esse motivo, o valor final adotado foi `k=4` — testado e confirmado no cenário de "meu pedido foi extraviado", onde 4 chunks distintos, todos genuinamente relevantes, passaram no filtro de relevância ao mesmo tempo.

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

## 6. Limitação do `score_threshold`: transferência prematura sem contexto

O `score_threshold` resolve o problema de trazer chunks irrelevantes, mas introduz um efeito colateral: quando nenhum chunk atinge o limiar, o contexto retornado fica vazio, e o *system prompt* instrui o modelo a usar o marcador de transferência (`###TRANSFER_HUMANO###`) nesse caso. Isso significa que qualquer pergunta ambígua, mal formulada ou genuinamente fora do domínio resultava em transferência **imediata** para um atendente humano, sem nenhuma chance de o cliente reformular a pergunta.

**Por que isso é um problema de produto, não só técnico:** transferir para atendimento humano tem custo real — tempo de fila, carga de trabalho do atendente, e perda de contexto (o atendente não tem acesso ao histórico da conversa com o Max). Tratar "não encontrei contexto relevante" como sinônimo de "preciso de um humano" descarta casos em que o problema era simplesmente uma pergunta mal formulada, resolvível com um pedido de esclarecimento.

**Estratégia adotada:** um contador de tentativas sem contexto (`tentativas_sem_contexto`), controlado inteiramente pelo código — não pelo modelo, para evitar depender da confiabilidade do LLM em "lembrar" quantas vezes uma regra já foi aplicada.

- Na primeira vez que uma pergunta não retorna contexto relevante, o Max responde com uma mensagem fixa pedindo para o cliente reformular ou detalhar a pergunta, sem chamar o LLM.
- Se a tentativa seguinte também não retornar contexto, a transferência para atendente humano é acionada diretamente pelo código.
- Se, em qualquer momento, uma pergunta retornar contexto válido, o contador é reiniciado — o "crédito" de tentativas é renovado.

```python
if not contexto.strip():
    tentativas_sem_contexto += 1

    if tentativas_sem_contexto >= 2:
        print('Max: Não consegui entender sua solicitação. Vou te transferir para um atendente.')
        print('[Sistema]: Transferindo...')
        break

    print('Max: Não entendi muito bem sua pergunta. Você pode explicar de outra forma, com mais detalhes sobre seu pedido?')
    continue

tentativas_sem_contexto = 0
```

**Por que o controle ficou no código, e não no prompt:** essa decisão segue a mesma lição da Seção 7 — contar tentativas ou aplicar uma regra de forma consistente é um tipo de lógica que um LLM pode falhar em seguir de forma confiável ao longo de uma conversa longa. Colocando o contador como uma variável Python comum, o comportamento fica determinístico e não depende da interpretação do modelo.

**Conclusão:** um mecanismo de recuperação (`score_threshold`) que descarta contexto irrelevante precisa de uma camada de decisão adicional para não converter automaticamente "sem contexto" em "transferir para humano". Separar essas duas coisas — dar ao cliente uma chance de reformular antes de escalar — reduz transferências desnecessárias sem comprometer o fallback para casos genuinamente fora do escopo do assistente.

## 7. Bug de marcador de controle confundido com linguagem natural

Ao testar o `score_threshold` com uma pergunta fora do domínio (sem nenhum chunk retornado), o modelo deveria responder com o marcador de controle `TRANSFERIR_HUMANO`, definido no *system prompt*, para acionar a transferência para um atendente humano.

**Resultado observado:** o modelo gerou `TRANSFIRIR_HUMANO` (com erro de grafia — "transfIrir" em vez de "transfErir"). Como a checagem no código (`if 'TRANSFERIR_HUMANO' in reply`) busca a string exata, a condição não foi satisfeita, e o fluxo de transferência não foi acionado.

**Causa raiz:** o próprio *system prompt* usa, em outras regras, o verbo "transfira" (imperativo correto de "transferir", com "i"). O modelo aparentemente generalizou esse padrão de conjugação por cima do marcador de controle, que deveria ser reproduzido literalmente, e não interpretado como parte do texto em português.

**Correção aplicada:**
- Substituição do marcador por um token que não se pareça com uma palavra natural do idioma: `###TRANSFER_HUMANO###`.
- Checagem no código tornada mais tolerante a variações, verificando apenas o núcleo do token em maiúsculas: `if 'TRANSFER_HUMANO' in reply.upper()`.

**Conclusão:** confiar na reprodução exata de uma palavra-chave de controle por um LLM é frágil, especialmente quando essa palavra se assemelha a vocabulário comum do idioma usado no restante do prompt. Marcadores de controle devem ser visualmente distintos de linguagem natural, e a validação no código deve ser tolerante a pequenas variações de grafia.

## 8. Grounding verification: bloqueando inferências não fundamentadas

A Seção 4 documentou um problema que ficou em aberto por toda a investigação: quando o contexto recuperado é relacionado à pergunta, mas não cobre exatamente o cenário descrito, o modelo tende a preencher a lacuna combinando fatos reais em uma inferência não autorizada (ex.: "aguarde mais um pouco"). Nem `score_threshold`, nem o contador de tentativas resolvem esse caso — os dois só agem quando o contexto está **vazio**, e aqui o contexto existe, só está incompleto.

**Abordagens consideradas:** duas estratégias foram avaliadas antes da implementação.

| | Segunda chamada ao modelo (LLM-as-judge) | Validação estruturada em código |
|---|---|---|
| Custo de API | Alto (dobra chamadas) | Baixo |
| Latência | Maior | Menor |
| Complexidade de implementação | Menor | Maior (exige formato de citação rígido e verificável) |
| Robustez | Maior | Menor (depende do modelo seguir o formato exigido) |

A decisão foi pela primeira abordagem, priorizando segurança sobre custo operacional: para um chatbot de atendimento, o custo de uma resposta incorreta (reputação, retrabalho) supera o custo de uma chamada extra de API.

**Implementação:** depois que a resposta do Max (`reply`) é gerada, uma segunda chamada ao modelo — com um prompt isolado, sem as regras de atendimento do Max — verifica se a resposta contém alguma afirmação não presente literalmente no contexto.

```python
def verificar_grounding(llm, contexto, resposta):
    prompt_verificacao = f"""Você é um verificador de fatos. Analise se a resposta abaixo usa APENAS informações presentes no contexto fornecido, sem inferências ou combinações não explícitas.

Contexto:
{contexto}

Resposta a verificar:
{resposta}

A resposta contém alguma afirmação, recomendação ou instrução que NÃO está literalmente escrita no contexto acima? Responda apenas SIM ou NÃO."""

    verificacao = llm.invoke(prompt_verificacao)
    return 'SIM' in verificacao.content.upper()
```

**Por que o prompt de verificação é isolado do *system prompt* do Max:** reaproveitar o mesmo prompt (com suas 12 regras de atendimento) geraria instruções concorrentes — "seja o Max, atendente empático" e "seja um verificador crítico" ao mesmo tempo. Um prompt dedicado, sem outras responsabilidades, evita esse conflito.

**Por que a checagem de grounding vem depois da checagem de `TRANSFER_HUMANO`:** se o modelo já respondeu com o marcador de transferência, `reply` não contém uma afirmação factual a ser verificada — rodar o grounding nesse caso seria uma chamada de API desperdiçada.

**Convenção usada nos testes:** `verificar_grounding` retorna `True` quando a resposta é considerada **não fundamentada** (contém algo fora do contexto, e deve ser bloqueada), e `False` quando a resposta está corretamente fundamentada e pode ser exibida. Nos termos usados a seguir, um **falso negativo** é quando o verificador retorna `False` (deixa passar) para uma resposta que, na verdade, continha uma inferência não fundamentada.

**Resultados de teste:**

| Pergunta | Contexto recuperado | Grounding | Avaliação |
|---|---|---|---|
| "atraso simples" (Seção 4) | Parcial, sem regra explícita | Bloqueou | ✅ Correto |
| "prazo pro sul" | Específico e completo | Passou | ✅ Correto |
| "meu pedido foi extraviado" (4 chunks) | Múltiplas fontes legítimas | Passou | ✅ Correto (combinação válida) |
| "6 dias sem receber reembolso" | Comparação numérica explícita | Transferiu antes do grounding (o próprio Max reconheceu a lacuna) | ✅ Correto |
| "reembolso demorando um pouco mais" | Prazo real + convite a sugestão genérica | Passou, mas a resposta continha "acione o suporte" — não fundamentado | ❌ Falso negativo |
| "reenvio sem atualização de status" | Mesmo padrão do caso anterior | Bloqueou | ✅ Correto |
| "sem código de rastreamento" | Mesmo padrão | Bloqueou | ✅ Correto |

**Limitação identificada:** de 7 perguntas testadas, 6 tiveram o comportamento esperado e 1 vazou uma inferência (uma sugestão de ação genérica, não um dado inventado). Tentativas de reproduzir esse mesmo padrão em outras perguntas estruturalmente parecidas não repetiram a falha — sugerindo um caso isolado, não uma falha sistemática. A causa provável é que o verificador usa o mesmo modelo (e portanto os mesmos vieses) que gera a resposta original: uma sugestão como "acione o suporte" pode não ser reconhecida como violação por parecer bom senso de atendimento, em vez de uma invenção factual explícita. Isso é consistente com uma limitação conhecida da técnica de LLM-as-judge — usar o mesmo modelo (ou modelo da mesma família) para gerar e verificar tende a ter pontos cegos correlacionados.

**Teste de confirmação da hipótese:** para avaliar se a causa era mesmo o modelo do verificador (e não um problema aleatório), a mesma pergunta que gerou o falso negativo foi testada novamente, trocando apenas o modelo usado em `verificar_grounding` de `gpt-4o-mini` para `gpt-4o` (mantendo o `gpt-4o-mini` como gerador das respostas do Max). Com o modelo mais forte como verificador, o mesmo caso que antes passava (`False`) foi corretamente bloqueado (`True`). O resultado reforçou a hipótese de que o falso negativo estava relacionado à capacidade do modelo utilizado como verificador, e não a um problema na lógica de validação ou no prompt — um único teste não é suficiente para confirmar isso de forma definitiva, mas é evidência consistente a favor da explicação.

**Decisão sobre uso em produção:** apesar do ganho de confiabilidade observado com o `gpt-4o`, a configuração final do projeto permaneceu utilizando `gpt-4o-mini` em ambos os papéis. O custo por token do `gpt-4o` é cerca de 17x maior (US$ 2,50/US$ 10,00 vs. US$ 0,15/US$ 0,60 por milhão de tokens, entrada/saída), e o objetivo educacional do projeto, somado à natureza pontual do falso negativo encontrado (1 em 7 testes), não justificou o custo extra. Em um ambiente de produção real, essa escolha poderia ser diferente, considerando o impacto financeiro de respostas incorretas e a criticidade do domínio — nesse caso, o custo do risco (reputação, retrabalho) pode superar o custo da chamada mais cara. A troca de modelo permanece documentada como uma opção validada, disponível para revisão caso a taxa de falso negativo se mostre maior em uso real.

**Conclusão:** grounding verification reduz de forma significativa a taxa de alucinação por combinação de fatos — um problema que nenhuma outra camada (prompt engineering, `temperature`, `score_threshold`) havia conseguido bloquear. Ainda assim, não elimina o problema por completo: uma segunda camada de verificação com o mesmo modelo que gerou a resposta carrega parte dos mesmos vieses, então o resultado deve ser tratado como redução de risco, não garantia absoluta. Um verificador com modelo mais forte reduz esse viés, em evidência observada num teste controlado, mas ao custo de uma chamada de API significativamente mais cara — uma decisão de trade-off entre segurança e custo operacional, não uma correção "gratuita".

**Possível evolução futura:** a escolha de modelo não precisa ser a mesma para os dois papéis. Uma arquitetura mais madura poderia manter um modelo econômico (`gpt-4o-mini`) para gerar respostas — a etapa de maior volume de chamadas — e reservar um modelo mais robusto apenas para a etapa crítica de verificação, que ocorre uma vez por resposta. Isso concentraria o custo mais alto exatamente onde a segurança importa mais, em vez de pagar o mesmo prêmio em toda a interação.

## 9. Persistência do índice FAISS: eliminando reprocessamento desnecessário

Nas versões anteriores do projeto, `carregar_base_conhecimento()` recalculava o índice FAISS do zero a cada execução — carregando o `politicas.txt`, quebrando em chunks e gerando embeddings via API da OpenAI para cada um deles, mesmo quando nada havia mudado desde a última vez. Isso levava entre 10 e 15 segundos por execução, um custo que cresceria proporcionalmente ao tamanho da base de conhecimento em um cenário de produção real.

**Estratégia adotada:** salvar o índice em disco após o primeiro cálculo, e nas execuções seguintes, carregar esse índice já pronto — recalculando do zero apenas quando o `politicas.txt` for alterado. Para detectar essa alteração, a data de modificação do arquivo (`os.path.getmtime`) é salva junto com o índice, em um arquivo de metadados separado; a cada execução, essa data é comparada com a data atual do arquivo antes de decidir qual caminho seguir.

```python
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
            vectorstore = FAISS.load_local(caminho_indice, embeddings, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={'score_threshold': 0.68, 'k': 4}
            )

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
```

**Por que a checagem de data precisa vir antes de qualquer processamento:** o objetivo da persistência é evitar trabalho desnecessário. Se a comparação de datas acontecesse depois de já ter carregado o arquivo e gerado os embeddings, o tempo e o custo que se queria evitar já teriam sido gastos antes da decisão ser tomada.

**Por que a data de modificação precisa ser salva em um arquivo próprio:** `vectorstore.save_local()` gera dois arquivos (`index.faiss`, com os vetores, e `index.pkl`, com o texto original de cada chunk) — nenhum dos dois guarda informação sobre quando o arquivo de origem foi editado, porque esse não é o propósito deles. Um terceiro arquivo, criado especificamente para esse controle, foi necessário.

**Sobre `allow_dangerous_deserialization=True`:** esse parâmetro é uma confirmação explícita, exigida pelo LangChain, de que a origem do arquivo carregado é confiável — o formato de serialização usado (`pickle`) pode, em tese, executar código arbitrário se o arquivo carregado vier de uma fonte não verificada. Como o índice é gerado pelo próprio projeto, na própria máquina, esse risco não se aplica aqui; a confirmação existe para casos onde um índice fosse compartilhado ou baixado de terceiros.

**Resultado medido:** para confirmar o ganho real (e não apenas a percepção de "ficou mais rápido"), o tempo de cada etapa foi medido com `time.time()` em uma execução com o índice já persistido:

| Etapa | Tempo |
|---|---|
| Imports das bibliotecas (langchain, faiss, etc.) | 2.73s |
| Criar `OpenAIEmbeddings()` | 0.70s |
| `FAISS.load_local()` | 0.17s |
| Total até a saudação do Max aparecer | 3.64s |

**Descoberta inesperada:** o tempo ainda percebido como "não tão rápido quanto esperado" (uns 5-6 segundos, na sensação inicial) não vinha mais do FAISS ou dos embeddings — a persistência eliminou esse gargalo com sucesso (`load_local()` levou apenas 0.17s). O tempo restante é dominado pelos **imports das bibliotecas** (2.73s, mais de 75% do tempo total), uma etapa anterior a qualquer lógica de persistência, comum a qualquer projeto que use LangChain e não relacionada ao tamanho da base de conhecimento.

**Conclusão:** a persistência resolveu o problema real que motivou a mudança — o reprocessamento repetido de embeddings, que escalaria mal com uma base de conhecimento maior. O tempo de import das bibliotecas é um custo fixo e comum ao framework, não um sintoma do problema original, e não vale a pena otimizar mais a fundo para um projeto deste porte. Medir antes de continuar otimizando evitou gastar esforço perseguindo um gargalo que já não existia mais.

## 10. Separando logs técnicos da interface do usuário

O projeto acumulou dois tipos de aviso técnico ao longo do desenvolvimento: um `DeprecationWarning` do `langchain-community` (emitido pelo módulo `warnings` do Python no momento do import) e um `WARNING` interno do LangChain quando `score_threshold` não encontra nenhum chunk relevante (emitido pelo módulo `logging`). Nas primeiras versões, cada um foi resolvido de forma pontual, com filtros que **descartavam** a mensagem por completo (`warnings.filterwarnings('ignore', ...)` e `logging.getLogger(...).setLevel(logging.ERROR)`) — suficiente para manter o terminal limpo, mas às custas de perder qualquer rastro desses eventos.

**Problema com a abordagem de descarte:** silenciar um aviso o torna invisível também para quem desenvolve o projeto. Se um comportamento inesperado começasse a gerar avisos com mais frequência em uso real, não haveria como perceber, porque a mensagem nunca chega a existir em lugar nenhum.

**Estratégia adotada:** em vez de descartar, os avisos passaram a ser **redirecionados** para um arquivo de log (`logs/app.log`), mantendo o terminal visível ao usuário limpo, mas preservando o histórico para consulta e depuração.

```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

logging.basicConfig(
    filename=os.path.join(BASE_DIR, 'logs', 'app.log'),
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.captureWarnings(True)
```

**Por que `logging.captureWarnings(True)` era necessário:** os módulos `warnings` e `logging` são sistemas independentes no Python, que não se comunicam por padrão. Essa função cria uma ponte, redirecionando o que passaria pelo `warnings` (como o `DeprecationWarning`) para dentro do sistema de `logging`, permitindo que os dois tipos de aviso — de origens diferentes — sejam capturados pela mesma configuração de arquivo.

**Por que essa configuração precisa vir antes dos imports do LangChain:** o `DeprecationWarning` é disparado no exato momento em que `from langchain_community.vectorstores import FAISS` é executado. Se a configuração de log viesse depois dessa linha, o aviso já teria sido descartado (ou impresso no terminal) antes de existir qualquer lugar para redirecioná-lo.

**Por que a pasta `logs/` é criada em código, e não apenas documentada como pré-requisito:** ela foi adicionada ao `.gitignore` (arquivos de log são artefatos de execução, não código-fonte, e crescem a cada uso). Isso significa que ela nunca existirá automaticamente ao clonar o repositório. Documentar "crie a pasta antes de rodar" transferiria ao usuário uma responsabilidade que o próprio programa pode cumprir de forma confiável com uma linha (`os.makedirs(..., exist_ok=True)`), sem custo perceptível em qualquer execução.

**Resultado:** ambos os tipos de aviso passaram a ser registrados em `logs/app.log`, com data, hora e nível de severidade, sem aparecer no terminal:

```
2026-07-11 21:51:36,916 - WARNING - .../main.py:16: DeprecationWarning: `langchain-community` is being sunset...
2026-07-11 21:52:40,706 - WARNING - No relevant docs were retrieved using the relevance score threshold 0.68
```

Vale notar que o segundo aviso passou a ser capturado sem precisar de nenhum filtro específico por módulo (como o `logging.getLogger('langchain_core.vectorstores')` usado anteriormente) — a configuração de `level=logging.WARNING` no `basicConfig` já captura qualquer aviso desse nível ou mais grave, de qualquer origem que use o sistema `logging`, tornando a solução mais genérica e resiliente a avisos futuros ainda não identificados.

**Conclusão:** existe uma diferença prática entre "silenciar" e "redirecionar" um aviso técnico. Descartar é apropriado quando a mensagem é comprovadamente irrelevante; redirecionar para um log é mais apropriado quando a mensagem pode ter valor de diagnóstico futuro, mesmo não sendo destinada ao usuário final. Separar canais de saída — interface do usuário via `print()`, diagnóstico técnico via `logging` em arquivo — é uma prática comum em aplicações reais, especialmente à medida que um projeto de terminal evolui para algo servido como aplicação (API, interface web).

## 11. Detecção de saudação: evitando penalizar small talk

Testando o fluxo manualmente, foi observado que uma saudação simples ("ola") gerava contexto vazio na busca (nenhum chunk do `politicas.txt` é relevante para um cumprimento) e disparava o contador `tentativas_sem_contexto` da mesma forma que uma pergunta genuinamente fora do domínio. Em uma sequência de duas saudações seguidas — comportamento humano plausível em qualquer atendimento real — o cliente seria transferido para um atendente sem ter feito nenhuma pergunta de negócio.

**Por que isso é um problema de produto:** o fallback de contador foi desenhado para capturar perguntas fora do escopo do assistente, não para penalizar conversa social sem conteúdo. Tratar as duas situações da mesma forma gera transferências desnecessárias logo no início do atendimento.

**Abordagens descartadas antes da solução final:**

- **Lista de palavras-chave fixas** (`if pergunta in ['oi', 'ola', ...]`) — falha diante de variações informais de escrita ("oii", "olar", "eae"), pelo mesmo motivo que uma correspondência de texto exata já havia se mostrado frágil no bug do marcador `TRANSFERIR_HUMANO` (Seção 7).
- **Filtro por tamanho da mensagem** — descartado ao se considerar o contra-exemplo "meu pedido ta atrasado", que tem tamanho comparável a uma saudação estendida, mas é uma pergunta de negócio legítima. Tamanho não correlaciona de forma confiável com a distinção que importa.
- **Detecção de mensagens fragmentadas ou incompletas** (ex.: cliente envia a mensagem sem querer, no meio de digitar) — considerada, mas não implementada. Diferente de saudação, que segue um padrão finito e reconhecível, um fragmento não tem um conjunto fixo de exemplos comparáveis; julgar se uma frase está "gramaticalmente completa" exige um tipo de julgamento semântico mais próximo do que motivou o uso de LLM no grounding verification (Seção 8), o que reintroduziria custo de chamada por mensagem. Fica documentado como limitação conhecida, não resolvida nesta versão.

**Solução adotada:** uma segunda base vetorial, pequena e independente da base de conhecimento principal, criada a partir de uma lista de exemplos de saudação. A pergunta do cliente é comparada contra essa base antes de qualquer busca no `politicas.txt`; se a similaridade for alta o suficiente, a mensagem é tratada como saudação — respondida com uma mensagem fixa (sem chamar o LLM) e sem contar como tentativa falha.

```python
def carregar_indice_saudacoes():
    exemplos_saudacao = ['olá', 'oi', 'oii', 'bom dia', 'boa tarde', 'boa noite', 'tudo bem', 'e aí', 'opa', 'salve']
    documentos_saudacao = [Document(page_content=texto) for texto in exemplos_saudacao]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documentos_saudacao, embeddings)

def eh_saudacao(vectorstore_saudacoes, pergunta):
    resultados = vectorstore_saudacoes.similarity_search_with_relevance_scores(pergunta, k=1)
    _, score = resultados[0]
    return score >= 0.85
```

**Por que a checagem precisa vir antes de `buscar_contexto`, e não dentro de `verificar_grounding`:** o `verificar_grounding` só é alcançado após a resposta do LLM já ter sido gerada, dentro do fluxo normal. Como uma saudação gera contexto vazio, ela já é interceptada pelo bloco `if not contexto.strip()` (via `continue`) antes de a execução chegar perto do LLM ou do verificador — qualquer regra colocada dentro de `verificar_grounding` para tratar saudação nunca seria executada para esse caso.

**Por que a resposta à saudação é fixa no código, e não gerada pelo LLM:** mesmo princípio já aplicado ao fallback de "não entendi" (Seção 6) — uma saudação não exige raciocínio, então gerar a resposta via `llm.stream()` seria custo desnecessário para uma tarefa totalmente previsível.

**Calibração do `threshold` (0.85):** o valor inicial de 0.75 foi testado e rejeitado com dado real — "meu pedido ta atrasado" obteve score 0.7657, acima do valor testado, o que classificaria incorretamente uma pergunta de negócio como saudação. Elevando para 0.85, a margem de segurança contra falsos positivos se sustentou em múltiplos testes:

| Frase | Score | Classificação esperada | Resultado |
|---|---|---|---|
| "oi" | 1.0000 | Saudação | ✅ |
| "oie" | 0.9094 | Saudação (variação) | ✅ |
| "bom diaa" | 0.9569 | Saudação (variação) | ✅ |
| "hello" | 0.7962 | Saudação | ❌ (abaixo do threshold) |
| "meu pedido ta atrasado" | 0.7657 | Não-saudação | ✅ |
| "qual o prazo" | 0.7413 | Não-saudação | ✅ |
| "oi, meu pedido atrasou" (mensagem mista) | 0.7756 | Não-saudação | ✅ |

**Limitação aceita conscientemente:** com 0.85, saudações em outro idioma ("hello") ou gírias regionais não incluídas na lista de exemplos ficam abaixo do threshold e não são reconhecidas como saudação — o cliente recebe o fallback de "não entendi, pode reformular?" na primeira tentativa. Essa foi uma escolha deliberada: abaixar o threshold para cobrir esses casos reduziria a margem de segurança contra falsos positivos em perguntas de negócio curtas, que é o risco mais custoso dos dois. Quando um caso específico se mostrou relevante o suficiente para justificar tratamento (a gíria "salve", mais comum no contexto brasileiro do que "hello"), a solução adotada foi ampliar a lista de exemplos de referência, não reduzir o threshold — isso resolveu o caso sem comprometer a margem de segurança já validada.

**Teste de ambiguidade semântica:** como a palavra "salve" também pode ser usada como verbo ("salve meu número de rastreamento"), esse cenário foi testado deliberadamente antes de considerar a solução validada. O resultado (score 0.7854, abaixo do threshold) confirmou que o embedding distingue corretamente a interjeição isolada do verbo em contexto de frase — a comparação por similaridade captura a estrutura semântica da frase completa, não apenas a presença da palavra.

**Conclusão:** o mesmo mecanismo de embedding usado para RAG de negócio pode ser reaproveitado, de forma barata, para classificar categorias de mensagem que não são sobre conteúdo de negócio (como small talk) — evitando tanto correspondência de texto frágil (listas fixas) quanto o custo de uma chamada de LLM completa para uma tarefa que não exige julgamento complexo. A calibração do threshold seguiu a mesma metodologia usada em `score_threshold` (Seção 5): testar categorias antagônicas, medir a margem real entre elas, e tratar exceções conhecidas ampliando a base de exemplos em vez de comprometer a margem de segurança já validada.

## 12. Conclusões gerais

- RAG reduz alucinação, mas não a elimina — mesmo com contexto correto recuperado, o modelo pode combinar fatos legítimos de formas não autorizadas pelo negócio.
- Instruções em linguagem natural no *system prompt* têm um teto de eficácia: proibições, checagens explícitas e restrições literais foram testadas e nenhuma bloqueou o comportamento por completo.
- `chunk_size` e `k` não devem ser avaliados isoladamente — o tamanho da base de conhecimento determina se os efeitos de cada um ficam visíveis ou escondidos.
- Um `score_threshold` calibrado com dados reais é mais robusto que um `k` fixo, mas ainda depende de uma escolha de engenharia dentro de uma margem, não de um valor absoluto.
- Marcadores de controle (tokens especiais usados para acionar lógica no código) precisam ser distintos de linguagem natural, e a validação correspondente no código deve tolerar variações — nenhuma reprodução de texto por um LLM deve ser considerada 100% garantida.
- Regras que dependem de contagem ou estado ao longo da conversa (como "quantas vezes isso já aconteceu") são mais confiáveis quando controladas por código determinístico do que quando delegadas inteiramente ao modelo.
- Grounding verification com uma segunda chamada ao mesmo modelo reduz drasticamente, mas não elimina, alucinação por inferência — porque o verificador herda parte dos vieses do modelo que está verificando. Um modelo mais forte no papel de verificador comprovadamente reduz esse viés, mas a decisão de adotá-lo é uma escolha de custo, não uma correção óbvia.
- Otimização de performance deve ser guiada por medição, não por sensação: o gargalo percebido nem sempre é o gargalo real, e resolver o problema errado consome tempo sem resultado.
- Silenciar um aviso técnico e redirecioná-lo para um log são decisões diferentes: a primeira descarta informação, a segunda a preserva para diagnóstico sem expô-la à interface do usuário.
- Embeddings são úteis além da recuperação de conteúdo de negócio: classificar tipo de mensagem (saudação vs. pergunta real) é uma aplicação barata da mesma técnica, desde que a margem entre categorias seja validada com casos antagônicos reais, não presumida.

## 13. Próximos passos identificados (não implementados ainda)

- **Query rewriting (contextual retrieval) e HyDE (Hypothetical Document Embeddings):** `buscar_contexto` recebe apenas a pergunta atual, isolada do histórico da conversa — diferente do LLM de geração, que recebe `messages` completo. Em uma sequência como "meu pedido atrasou" seguida de "e já faz 5 dias", a segunda busca vetorial usaria só a frase vaga, sem termos que o embedding relacione bem ao `politicas.txt`, mesmo a pergunta fazendo sentido no contexto da conversa. Duas técnicas resolvem esse tipo de problema por caminhos diferentes: *query rewriting* reformula a pergunta do usuário (com base no histórico) antes de gerar o embedding de busca; *HyDE* gera uma resposta hipotética para a pergunta e usa o embedding dessa resposta hipotética na busca, em vez do embedding da pergunta em si — a ideia é que uma resposta hipotética tende a ser semanticamente mais próxima de um chunk real (que também é texto de afirmação) do que uma pergunta pura. Ambas exigem uma chamada de LLM adicional antes do retriever. **Decisão consciente de não implementar nenhuma agora:** o objetivo deste projeto é educacional, e o custo extra por mensagem não se justifica na base de conhecimento atual (pequena, ~7 blocos). A decisão entre as duas técnicas — ou nenhuma — depende de um dado que só existe com uma base maior: qual o comportamento real de retrieval multi-turno em um volume de conteúdo mais próximo de produção. Fica planejado reavaliar as duas quando a base de conhecimento for expandida (ver item "base de conhecimento maior via PDF" no roadmap de estudos), testando-as sob a mesma base, em vez de decidir com a base pequena atual.
- **Detecção de mensagens fragmentadas ou incompletas:** cenário identificado na Seção 11, mas não implementado por exigir julgamento semântico (provavelmente via LLM), reintroduzindo custo de chamada por mensagem recebida.
- **Few-shot prompting:** incluir no *system prompt* um exemplo concreto de pergunta ambígua com a resposta correta esperada, em vez de apenas descrever a regra de forma abstrata.
- **Cobertura de conteúdo:** adicionar uma regra explícita para o cenário de "atraso simples" na base de conhecimento, eliminando a lacuna que hoje força o modelo a inferir.
- **Eval set mais robusto:** ampliar o conjunto de perguntas de teste usado para calibrar `score_threshold` e o grounding verification, cobrindo mais variações de pergunta específica, difusa e fora do domínio — para determinar se a taxa de falso negativo do verificador `gpt-4o-mini` justifica o custo extra do `gpt-4o` em uso real.
