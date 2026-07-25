def verificar_informacao_suficiente(llm, pergunta):
    prompt = f"""Analise a pergunta do cliente abaixo.

Pergunta: {pergunta}

Considere SIM apenas se a pergunta descreve uma situação (como atraso, tempo decorrido, ou problema) cuja resposta correta MUDA dependendo de um dado do pedido que não foi informado (região, prazo, data da compra) — ou seja, sem esse dado, é impossível saber qual regra se aplica.

Considere NÃO se:
- A pergunta já menciona a região ou dado necessário (ex: "prazo pro sul" já tem a região).
- A pergunta é sobre uma política que não depende de região ou prazo (ex: extravio, reembolso já aprovado, horário de atendimento, canais de atendimento).
- A pergunta é sobre algo que as políticas da empresa não cobrem — nesse caso a limitação não é falta de região/prazo/data, é falta de política, o que não deve ser tratado por este verificador.
- A pergunta é genérica sobre uma política, sem descrever uma situação específica do pedido do cliente.

Exemplos:
"meu pedido está atrasado há 3 dias, quero reembolso" → SIM (não sabemos a região, nem o prazo dela)
"qual o prazo pro sul" → NÃO (região já informada)
"meu pedido foi extraviado" → NÃO (extravio não depende de região)
"qual o horário de atendimento" → NÃO (não depende de nenhum dado do pedido)
"posso trocar meu pedido por outro produto de valor maior?" → NÃO (a resposta não depende de região, prazo ou data — depende de existir ou não uma política de troca, o que é um tipo de limitação diferente)

Primeiro, explique em uma frase: a resposta correta dependeria especificamente de região, prazo ou data da compra — ou de outra coisa?

Depois, na última linha, responda apenas com SIM ou NÃO.

Formato obrigatório:
Raciocínio: <sua explicação>
Veredito: SIM ou NÃO"""

    verificacao = llm.invoke(prompt)

    linhas = verificacao.content.strip().split('\n')
    veredito = linhas[-1] if linhas else verificacao.content
    return 'SIM' in veredito.upper()

def verificar_grounding(llm, pergunta, contexto, resposta):
    prompt_verificacao = f"""Você é um verificador de fatos. Analise se a resposta abaixo é sustentada pelo contexto fornecido.

Pergunta do cliente:
{pergunta}

Contexto:
{contexto}

Resposta a verificar:
{resposta}

A resposta pode aplicar uma política do contexto com base na condição de aplicabilidade dela, desde que essa condição corresponda a algo que o cliente afirmou na pergunta. Isso NÃO é considerado inferência indevida.

O que NÃO é permitido: a resposta inventar dados sobre o pedido (região, prazo, datas, status) que não foram mencionados nem pelo cliente nem pelo contexto, ou combinar políticas de forma que nenhuma delas sustente isoladamente a afirmação.

Primeiro, explique em uma frase seu raciocínio: identifique qual trecho do contexto sustenta (ou não sustenta) cada afirmação da resposta.

Depois, na última linha, responda apenas com SIM (se houver algo não sustentado) ou NÃO (se tudo estiver sustentado).

Formato obrigatório:
Raciocínio: <sua explicação>
Veredito: SIM ou NÃO"""

    verificacao = llm.invoke(prompt_verificacao)

    linhas = verificacao.content.strip().split('\n')
    veredito = linhas[-1] if linhas else verificacao.content
    return 'SIM' in veredito.upper()