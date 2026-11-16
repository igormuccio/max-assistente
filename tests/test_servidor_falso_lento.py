import time

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("max-tools-lento")


@mcp.tool()
def buscar_historico_anterior(cliente_id: int) -> str:
    """Tool que trava de propósito."""
    time.sleep(3600)
    return "nunca chega"


if __name__ == "__main__":
    mcp.run(transport="stdio")