import asyncio
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

RAIZ = Path(__file__).resolve().parents[1]
PASTA_LOGS = RAIZ / "logs"


def exception_original(erro):
    while hasattr(erro, "exceptions"):
        erro = erro.exceptions[0]
    return erro


async def tentar(rotulo, parametros, nome_log):
    print(f"--- {rotulo} ---")
    caminho_log = PASTA_LOGS / nome_log

    with open(caminho_log, "w", encoding="utf-8") as log:
        try:
            async with stdio_client(parametros, errlog=log) as (leitura, escrita):
                async with ClientSession(leitura, escrita) as sessao:
                    await sessao.initialize()
                    print("initialize passou")
                    ferramentas = await sessao.list_tools()
                    print([f.name for f in ferramentas.tools])
        except Exception as erro:
            original = exception_original(erro)
            print(f"Falhou: {type(original).__name__}: {original}")

    conteudo = caminho_log.read_text(encoding="utf-8").strip()
    print(f"stderr do servidor: {conteudo if conteudo else '(vazio)'}")
    print()


async def main():
    PASTA_LOGS.mkdir(exist_ok=True)
    await tentar(
        "Cenário 1: script do servidor inexistente",
        StdioServerParameters(command=sys.executable, args=[str(RAIZ / "src" / "servidor_que_nao_existe.py")]),
        "debug_mcp_stderr_cenario1.log",
    )
    await tentar(
        "Cenário 2: executável inexistente",
        StdioServerParameters(command="programa_que_nao_existe", args=[]),
        "debug_mcp_stderr_cenario2.log",
    )


asyncio.run(main())