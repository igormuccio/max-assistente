import asyncio
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CAMINHO_SERVIDOR = Path(__file__).resolve().parents[1] / "src" / "mcp_servidor.py"
CLIENTE_ID = 30


async def main():
    parametros = StdioServerParameters(command=sys.executable, args=[str(CAMINHO_SERVIDOR)])

    async with stdio_client(parametros) as (leitura, escrita):
        async with ClientSession(leitura, escrita) as sessao:
            await sessao.initialize()

            ferramentas = await sessao.list_tools()
            print([f.name for f in ferramentas.tools])

            resultado = await sessao.call_tool("buscar_historico_anterior", {"cliente_id": CLIENTE_ID})
            print(resultado.content[0].text)


asyncio.run(main())