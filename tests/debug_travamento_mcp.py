import asyncio
import sys
import time
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PASTA_TESTES = Path(__file__).resolve().parent
PASTA_LOGS = PASTA_TESTES.parent / "logs"
TIMEOUT_SEGUNDOS = 5


def exception_original(erro):
    while hasattr(erro, "exceptions"):
        erro = erro.exceptions[0]
    return erro


async def tentar(rotulo, caminho_servidor, nome_log):
    print(f"--- {rotulo} ---")
    parametros = StdioServerParameters(command=sys.executable, args=[str(caminho_servidor)])
    caminho_log = PASTA_LOGS / nome_log
    inicio = time.monotonic()
    etapa = "criar processo"

    with open(caminho_log, "w", encoding="utf-8") as log:
        try:
            async with stdio_client(parametros, errlog=log) as (leitura, escrita):
                async with ClientSession(leitura, escrita) as sessao:
                    etapa = "initialize"
                    with anyio.fail_after(TIMEOUT_SEGUNDOS):
                        await sessao.initialize()
                    print("initialize passou")

                    etapa = "list_tools"
                    with anyio.fail_after(TIMEOUT_SEGUNDOS):
                        ferramentas = await sessao.list_tools()
                    print([f.name for f in ferramentas.tools])

                    etapa = "call_tool"
                    with anyio.fail_after(TIMEOUT_SEGUNDOS):
                        resultado = await sessao.call_tool("buscar_historico_anterior", {"cliente_id": 1})
                    print(resultado.content[0].text)
        except Exception as erro:
            original = exception_original(erro)
            print(f"Falhou na etapa '{etapa}': {type(original).__name__}: {original}")

    conteudo = caminho_log.read_text(encoding="utf-8").strip()
    print(f"stderr do servidor: {conteudo if conteudo else '(vazio)'}")
    print(f"Tempo total: {time.monotonic() - inicio:.1f}s")
    print()


async def main():
    PASTA_LOGS.mkdir(exist_ok=True)
    await tentar(
        "Cenário 1: servidor que sobe e nunca responde",
        PASTA_TESTES / "test_servidor_falso_mudo.py",
        "debug_travamento_cenario1.log",
    )
    await tentar(
        "Cenário 2: servidor responde, mas a tool trava",
        PASTA_TESTES / "test_servidor_falso_lento.py",
        "debug_travamento_cenario2.log",
    )


asyncio.run(main())