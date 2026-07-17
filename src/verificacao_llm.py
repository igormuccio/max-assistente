def verificar_grounding(llm, contexto, resposta):
    prompt_verificacao = f"""Você é um verificador de fatos. Analise se a resposta abaixo usa APENAS informações presentes no contexto fornecido, sem inferências ou combinações não explícitas.

Contexto:
{contexto}

Resposta a verificar:
{resposta}

A resposta contém alguma afirmação, recomendação ou instrução que NÃO está literalmente escrita no contexto acima? Responda apenas SIM ou NÃO."""

    verificacao = llm.invoke(prompt_verificacao)
    return 'SIM' in verificacao.content.upper()