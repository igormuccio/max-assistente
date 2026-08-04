import os
import pdfplumber
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extrair_palavras(caminho_pdf):
    with pdfplumber.open(caminho_pdf) as pdf:
        palavras = []
        for pagina in pdf.pages:
            palavras.extend(pagina.extract_words(extra_attrs=['size', 'fontname']))
    return palavras


def identificar_perfis(palavras):
    contagem = Counter((round(p['size'], 1), p['fontname']) for p in palavras)
    perfil_corpo = contagem.most_common(1)[0][0]
    tamanho_corpo = perfil_corpo[0]

    perfis_maiores = sorted(
        [perfil for perfil in contagem if perfil != perfil_corpo and perfil[0] > tamanho_corpo],
        key=lambda perfil: perfil[0],
        reverse=True
    )

    perfil_run_in = perfis_maiores[-1] if perfis_maiores else None
    headings = perfis_maiores[:-1] if perfis_maiores else []

    outros_perfis = [
        perfil for perfil in contagem
        if perfil != perfil_corpo and perfil != perfil_run_in and perfil not in headings
    ]

    return {
        'corpo': perfil_corpo,
        'headings': headings,
        'run_in': perfil_run_in,
        'outros': outros_perfis
    }


def montar_chunks(palavras, perfis):
    nivel_chunk = perfis['headings'][-1]
    niveis_contexto = perfis['headings'][:-1]
    nivel_h2 = niveis_contexto[-1] if niveis_contexto else None

    chunks = []
    chunk_atual = None
    titulo_por_nivel = {nivel: '' for nivel in niveis_contexto}
    corpo_por_nivel = {nivel: '' for nivel in niveis_contexto}
    perfil_anterior = None
    aguardando_chunk = True
    teve_h3_no_h2_atual = True
    run_in_orfao = None

    def contexto_atual_formatado(ate_nivel=None):
        niveis = niveis_contexto if ate_nivel is None else niveis_contexto[:-1]
        resultado = []
        for nivel in niveis:
            texto = titulo_por_nivel[nivel]
            if corpo_por_nivel[nivel]:
                texto += ' ' + corpo_por_nivel[nivel]
            if texto:
                resultado.append(texto.strip())
        return resultado

    def fechar_h2_sem_h3():
        if not teve_h3_no_h2_atual and titulo_por_nivel.get(nivel_h2):
            chunks.append({
                'contexto': contexto_atual_formatado(ate_nivel=True),
                'titulo': titulo_por_nivel[nivel_h2],
                'texto': [corpo_por_nivel[nivel_h2]] if corpo_por_nivel[nivel_h2] else [],
                'run_in_ativo': run_in_orfao
            })

    for palavra in palavras:
        chave = (round(palavra['size'], 1), palavra['fontname'])

        if chave == nivel_chunk:
            teve_h3_no_h2_atual = True
            if chave == perfil_anterior:
                chunk_atual['titulo'] += ' ' + palavra['text']
            else:
                if chunk_atual:
                    chunks.append(chunk_atual)
                chunk_atual = {
                    'contexto': contexto_atual_formatado(),
                    'titulo': palavra['text'],
                    'texto': [],
                    'run_in_ativo': None
                }
                aguardando_chunk = False

        elif chave in niveis_contexto:
            if chave == nivel_h2 and chave != perfil_anterior:
                fechar_h2_sem_h3()
                teve_h3_no_h2_atual = False
                run_in_orfao = None

            if chave == perfil_anterior:
                titulo_por_nivel[chave] += ' ' + palavra['text']
            else:
                titulo_por_nivel[chave] = palavra['text']
                corpo_por_nivel[chave] = ''
            aguardando_chunk = True

        elif chave == perfis['run_in']:
            if teve_h3_no_h2_atual and chunk_atual:
                chunk_atual['run_in_ativo'] = palavra['text']
            elif not teve_h3_no_h2_atual:
                run_in_orfao = palavra['text']

        else:
            if aguardando_chunk:
                ultimo_nivel = max(titulo_por_nivel, key=lambda n: niveis_contexto.index(n))
                corpo_por_nivel[ultimo_nivel] += ' ' + palavra['text']
            elif chunk_atual:
                chunk_atual['texto'].append(palavra['text'])

        perfil_anterior = chave

    if chunk_atual:
        chunks.append(chunk_atual)
    fechar_h2_sem_h3()

    return chunks


def buscar_perfil_por_texto(palavras, textos_procurados):
    encontrados = {texto: [] for texto in textos_procurados}
    for palavra in palavras:
        for texto in textos_procurados:
            if texto in palavra['text']:
                perfil = (round(palavra['size'], 1), palavra['fontname'])
                encontrados[texto].append((palavra['text'], perfil))
    return encontrados


if __name__ == '__main__':
    MOSTRAR_CHUNKS = True
    MOSTRAR_BUSCA_DIRIGIDA = False

    caminho_pdf = os.path.join(BASE_DIR, 'data', 'politicas_xyz.pdf')

    palavras = extrair_palavras(caminho_pdf)
    perfis = identificar_perfis(palavras)

    print(f"\n===== PERFIS IDENTIFICADOS =====")
    print(f"Corpo: {perfis['corpo']}")
    print(f"Headings (maior -> menor): {perfis['headings']}")
    print(f"Run-in: {perfis['run_in']}")
    print(f"Outros (candidato a tabela): {perfis['outros']}")

    chunks = montar_chunks(palavras, perfis)

    if MOSTRAR_CHUNKS:
        print(f"\n===== {len(chunks)} CHUNKS CRIADOS =====")
        for i, chunk in enumerate(chunks, start=1):
            texto_completo = ' '.join(chunk['texto'])
            print(f"\n--- Chunk {i} ---")
            print(f"Contexto: {chunk['contexto']}")
            print(f"Título: {chunk['titulo']}")
            print(f"Run-in ativo: {chunk['run_in_ativo']}")
            print(f"Texto ({len(texto_completo)} caracteres): {texto_completo[:200]}...")

    if MOSTRAR_BUSCA_DIRIGIDA:
        alvos = buscar_perfil_por_texto(palavras, ['PRAZOS', 'INTERNACIONAIS', 'EXTRAVIO'])
        for texto, ocorrencias in alvos.items():
            print(f"\n'{texto}' encontrado em:")
            for texto_real, perfil in ocorrencias:
                print(f"  {texto_real} -> {perfil}")