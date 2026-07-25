# Max - Assistente Virtual de Entregas

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--Mini-black)
![License](https://img.shields.io/badge/License-MIT-yellow)

Projeto desenvolvido para estudo de aplicações de IA utilizando Large Language Models (LLMs), Prompt Engineering e Retrieval-Augmented Generation (RAG).

Assistente de atendimento ao cliente que responde com base nas políticas reais da empresa, transfere automaticamente para atendentes humanos quando necessário e mantém contexto durante toda a conversa.

> Além do código-fonte, este projeto documenta os experimentos realizados durante o desenvolvimento — incluindo falhas, hipóteses testadas e decisões arquiteturais recalibradas com dado real. Veja [EXPERIMENTS.md](./EXPERIMENTS.md).

## Objetivo

Desenvolver um assistente virtual de atendimento ao cliente baseado em LLMs e Retrieval-Augmented Generation (RAG), com foco não apenas em fazer o sistema funcionar, mas em entender os limites reais de cada decisão de arquitetura — calibração de parâmetros com dado real, mitigação de alucinação, trade-offs de custo e confiabilidade, e organização de código sustentável.

O projeto simula um cenário real de atendimento (a XYZ Entregas, empresa fictícia). Boa parte do valor deste repositório está no processo de investigação por trás de cada escolha técnica.

## Exemplo de conversa

**Fluxo de saudação — resposta fixa, sem custo de chamada ao modelo:**
```
Você: oii

Max: Olá! Como posso te ajudar hoje?
```

**Fluxo de informação insuficiente — antes mesmo da busca de contexto:**
```
Você: meu pedido está atrasado há 1 semana, o que eu faço?

Max: Para te ajudar melhor, preciso de mais alguns detalhes. Você pode
informar sua região e, se possível, há quantos dias fez o pedido?
```

**Fluxo padrão — pergunta de negócio com contexto encontrado:**
```
Max: Olá! Sou o Max, assistente da XYZ Entregas. Como posso ajudar?

Você: Meu pedido está constando como entregue mas não recebi nada.

Max: Entendo a situação. De acordo com nossas políticas, você deve acionar
o suporte em até 48h após a data de entrega. Abriremos uma investigação
com prazo de 3 dias úteis. Deseja prosseguir com o reenvio ou reembolso?

Você: Quero falar com um atendente.

Max: Aguarde, vou transferir para um atendente.
[Sistema]: Transferindo...
```

**Fluxo de contexto não encontrado — reformulação antes de transferir:**
```
Você: qual o horário do próximo jogo do brasileirão

Max: Não entendi muito bem sua pergunta. Você pode explicar de outra
forma, com mais detalhes sobre seu pedido?
```

## Arquitetura

```mermaid
flowchart TD
    Usuario[Usuário]
    mainpy[main.py]

    subgraph Inicializacao [Inicialização]
        inicializacaopy[inicializacao.py] --> CarregaPrompt[Carrega Prompt do Sistema]
        inicializacaopy --> CarregaBase[Carrega Base de Conhecimento por Seção]
        inicializacaopy --> CarregaIndice[Carrega ou Cria Índice Vetorial]
    end

    Inicializacao --> mainpy
    Usuario --> mainpy
    mainpy --> Saudacao{É saudação?}

    Saudacao -->|Sim| RespostaSaudacao[Resposta de saudação]
    RespostaSaudacao --> Usuario

    Saudacao -->|Não| FaltaInfo{Falta informação do pedido?}
    FaltaInfo -->|Sim| SolicitaInfo[Solicita região, prazo ou data]
    SolicitaInfo --> Usuario

    FaltaInfo -->|Não| buscapy[busca_semantica.py]
    buscapy --> ContextoEncontrado{Contexto encontrado?}

    ContextoEncontrado -->|Não| SolicitaDetalhes[Solicita mais detalhes ao usuário]
    SolicitaDetalhes --> PrimeiraTentativa{Primeira tentativa?}
    PrimeiraTentativa -->|Sim| Usuario
    PrimeiraTentativa -->|Não| Transferencia[Transferência para atendente]

    ContextoEncontrado -->|Sim| LLM[LLM]
    LLM --> MarcadorTransferencia{Contém marcador de transferência?}
    MarcadorTransferencia -->|Sim| Transferencia
    MarcadorTransferencia -->|Não| verificacaopy[verificacao_llm.py]

    verificacaopy --> RespostaFundamentada{Resposta fundamentada?}
    RespostaFundamentada -->|Não| Transferencia
    RespostaFundamentada -->|Sim| RespostaExibida[Resposta exibida ao usuário]
    RespostaExibida --> Usuario
```

## Como funciona

Antes de qualquer busca, o Max verifica se a mensagem é uma saudação — usando o mesmo mecanismo de embedding do RAG, mas contra uma base pequena de exemplos, sem gastar chamada ao modelo de linguagem para isso.

Em seguida, uma chamada ao LLM verifica se a pergunta já contém as informações necessárias para determinar qual regra se aplica (região, prazo, data da compra). Perguntas que dependem desses dados e ainda não os informaram recebem um pedido de mais detalhes antes de qualquer busca — evitando gastar uma chamada de geração e uma busca vetorial em uma pergunta que já se sabe estar incompleta.

Para perguntas de negócio, o sistema gera embeddings da consulta e realiza uma busca vetorial utilizando FAISS, com um limiar de relevância mínimo (`score_threshold`). A base de conhecimento é dividida por seção — uma política de negócio por chunk, não por contagem de caracteres — evitando que uma regra seja fragmentada entre sua condição e sua consequência.

Se nenhum chunk for relevante o suficiente, o Max pede para o cliente reformular; se isso se repetir, a conversa é transferida para um atendente, sem gastar uma chamada de geração em uma pergunta sem contexto útil.

Quando contexto relevante é encontrado, ele é inserido no prompt enviado ao GPT-4o Mini junto com a pergunta original do cliente, permitindo respostas fundamentadas na base de conhecimento da empresa — não no conhecimento genérico do modelo — e capazes de aplicar a política correta mesmo quando mais de uma política candidata aparece no contexto. Antes de exibir a resposta, uma segunda chamada ao modelo verifica se ela usa apenas informação presente no contexto e no relato do cliente (*grounding verification*); se a resposta contiver uma inferência não fundamentada, ela é descartada e a conversa é transferida para um atendente humano, em vez de mostrada ao cliente.

O índice vetorial é persistido em disco e só é recalculado quando a base de conhecimento (`politicas.txt`) é alterada, evitando reprocessamento desnecessário a cada execução.

Um eval set automatizado (`tests/`) roda um conjunto de perguntas de teste contra o pipeline completo — saudação, recuperação de contexto, necessidade de informação adicional e grounding — usado para validar que mudanças no prompt ou no chunking não introduzem regressões em comportamentos já calibrados.

## Funcionalidades

- Atendimento automatizado sobre problemas de entrega
- Detecção de saudação via embedding, sem custo de chamada ao modelo
- Verificação de informação suficiente antes da busca, evitando retrieval desnecessário em perguntas incompletas
- Respostas baseadas nas políticas da empresa via RAG, com limiar de relevância calibrado
- Chunking por seção da base de conhecimento, evitando fragmentar uma política de negócio no meio
- Verificação de grounding: uma segunda checagem que descarta respostas não fundamentadas no contexto ou no relato do cliente
- Fallback de reformulação antes de transferir, para perguntas fora do escopo
- Transferência automática para atendente humano quando necessário
- Persistência do índice vetorial em disco, recalculado apenas quando a base muda
- Eval set automatizado para validar regressões após mudanças de prompt, chunking ou verificação
- Streaming de respostas em tempo real
- Histórico de conversa durante a sessão
- Log técnico separado da interface do usuário

## Tecnologias

- Python 3.10+
- LangChain (core, text-splitters, community, openai)
- OpenAI API (GPT-4o Mini)
- OpenAI Embeddings
- FAISS (banco de vetores)
- python-dotenv

## Estrutura do projeto

```
max-assistente/
├── data/
│   ├── politicas.txt         # Base de conhecimento da empresa
│   ├── faiss_index/          # Índice vetorial persistido (gerado, ignorado no Git)
│   └── faiss_metadata.txt    # Controle de atualização do índice (gerado, ignorado no Git)
├── logs/
│   └── app.log                # Log técnico separado da interface do usuário (gerado, ignorado no Git)
├── prompts/
│   └── system.txt             # Personalidade e regras do assistente
├── src/
│   ├── main.py                 # Orquestração e loop de conversa
│   ├── inicializacao.py        # Carregamento de prompt, base de conhecimento por seção e índice de saudação
│   ├── busca_semantica.py      # Busca de contexto e detecção de saudação por embedding
│   └── verificacao_llm.py      # Verificação de informação suficiente e de grounding
├── tests/
│   ├── eval_set.json           # Casos de teste do eval set automatizado
│   └── run_evals.py            # Script de execução do eval set contra o pipeline completo
├── .env.example                # Exemplo de variáveis de ambiente
├── .gitignore
├── requirements.txt
├── EXPERIMENTS.md              # Documentação de testes e decisões técnicas
└── README.md
```

## Como rodar

1. Clone o repositório
```bash
git clone https://github.com/igormuccio/max-assistente.git
cd max-assistente
```

2. Crie e ative o ambiente virtual
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac
```

3. Instale as dependências
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente
```bash
cp .env.example .env
# Adicione sua OPENAI_API_KEY no arquivo .env
```

5. Execute o programa
```bash
python src/main.py
```

6. (Opcional) Rode o eval set
```bash
python tests/run_evals.py
```

## O que este projeto explora

- Prompt engineering para controle de comportamento, incluindo marcadores de controle, mitigação de regras concorrentes e coordenação entre o prompt de geração e os prompts de verificação
- Pipeline RAG completo: chunking calibrado por seção, embeddings, busca vetorial com FAISS, limiar de relevância
- Ambiguidade semântica entre políticas de negócio próximas, e técnicas de chunk enrichment para separá-las
- Detecção de alucinação e verificação de grounding como camada de segurança
- Trade-offs de custo vs. confiabilidade na escolha de modelo e arquitetura
- Persistência de índice vetorial e separação de logs técnicos
- Eval set automatizado para prevenir regressões entre camadas de retrieval, prompt e verificação
- Organização de projeto em módulos por responsabilidade

Cada uma dessas decisões foi testada empiricamente, não apenas assumida — incluindo casos em que a primeira solução falhou e precisou ser recalibrada. Para o histórico completo de testes, hipóteses e limitações conhecidas, veja [EXPERIMENTS.md](./EXPERIMENTS.md).

## Melhorias futuras

- Ampliar o eval set com mais variações de pergunta e um check dedicado a validar qual política de negócio foi aplicada na resposta final
- Few-shot prompting sistemático nos prompts de geração e verificação, informado pelo eval set expandido
- Query rewriting ou HyDE para conversas multi-turno, a avaliar com uma base de conhecimento maior
- Interface Web com Streamlit
- API REST utilizando FastAPI
- Banco vetorial dedicado (Chroma ou Pinecone)
- Docker para facilitar o deploy

## Observações

As políticas presentes em `data/politicas.txt` são fictícias e utilizadas apenas para fins de demonstração.
