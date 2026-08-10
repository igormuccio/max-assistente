from inicializacao import criar_llm
from verificacao_llm import verificar_informacao_suficiente

llm = criar_llm()

pergunta = "qual o prazo de entrega para o sul?"

prompt = f"""Analise a pergunta do cliente abaixo.

Pergunta: {pergunta}

Considere SIM apenas se a pergunta descreve uma situação cuja resposta correta MUDA dependendo de um dado do pedido que não foi informado — e esse dado é necessário especificamente para o assunto perguntado, não em geral.

A empresa atende pedidos nacionais e, em caráter piloto, pedidos internacionais para dois países: Portugal e Estados Unidos. O dado que falta depende do assunto da pergunta:

- PRAZO DE ENTREGA: se o pedido é internacional, o prazo muda entre Portugal (10 a 15 dias úteis) e Estados Unidos (8 a 12 dias úteis) — nesse caso, saber apenas "internacional" NÃO é suficiente, é necessário saber qual dos dois países.
- EXTRAVIO: a regra muda apenas entre nacional e internacional — o procedimento é o mesmo para Portugal e Estados Unidos. Nesse caso, saber "internacional" (sem precisar do país específico) já é suficiente.
- Se o cliente menciona um país que não é Portugal nem Estados Unidos, isso não é falta de informação — é um destino que a empresa não cobre, o que é um tipo de limitação diferente e não deve ser tratado por este verificador.
- Outros assuntos (reembolso já aprovado, taxas alfandegárias, horário de atendimento, canais de atendimento) não dependem de região, país ou prazo — o dado nacional/internacional é irrelevante para eles.

Considere NÃO se:
- A pergunta já contém o dado necessário para o assunto específico perguntado (ex: região doméstica já basta para prazo nacional; "internacional" já basta para extravio; o país específico já foi informado, quando o assunto exige).
- A pergunta é sobre um assunto que não depende de região, país de destino ou prazo.
- A pergunta é sobre algo que as políticas da empresa não cobrem (incluindo destinos internacionais fora de Portugal e Estados Unidos) — isso é falta de política, não falta de dado, e não deve ser tratado por este verificador.
- A pergunta é genérica sobre uma política, sem descrever uma situação específica do pedido do cliente.

Exemplos:
"meu pedido está atrasado há 3 dias, quero reembolso" → SIM (não sabemos a região, nem o prazo dela)
"qual o prazo pro sul" → NÃO (região doméstica já indica pedido nacional, suficiente para prazo)
"qual o prazo do meu pedido internacional" → SIM (para prazo, "internacional" sozinho não basta — precisa saber Portugal ou Estados Unidos)
"qual o prazo do meu pedido para os Estados Unidos" → NÃO (país já definido, suficiente para prazo)
"meu pedido foi extraviado" → SIM (não sabemos se é nacional ou internacional, e a regra muda entre os dois)
"meu pedido internacional foi extraviado" → NÃO (para extravio, "internacional" já basta — a regra é igual para Portugal e Estados Unidos)
"meu pedido para a França foi extraviado" → NÃO (não é falta de dado — a empresa não cobre esse destino, é uma limitação de escopo, não deste verificador)
"qual o horário de atendimento" → NÃO (não depende de nenhum dado do pedido)
"posso trocar meu pedido por outro produto de valor maior?" → NÃO (depende de existir ou não uma política de troca, não de região, país ou prazo)

Primeiro, explique em uma frase: qual é o assunto da pergunta, e esse assunto especificamente exige um dado que não foi informado?

Depois, na última linha, responda apenas com SIM ou NÃO.

Formato obrigatório:
Raciocínio: <sua explicação>
Veredito: SIM ou NÃO"""

resposta = llm.invoke(prompt)

print("===== RESPOSTA COMPLETA DO MODELO =====")
print(resposta.content)
print("=========================================")

resultado = verificar_informacao_suficiente(llm, pergunta)
print(f"\nResultado da função (parseado): {resultado}")