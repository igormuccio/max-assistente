from mcp.server.mcpserver import MCPServer

from tools.historico import consultar_historico

mcp = MCPServer("max-tools")


@mcp.tool()
def buscar_historico_anterior(cliente_id: int) -> str:
    """Busca o histórico de conversas anteriores deste cliente, quando ele parece estar retomando um assunto já discutido antes."""
    return consultar_historico(cliente_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")