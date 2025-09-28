# calculator_server.py
from mcp.server.fastmcp import FastMCP
import sys


mcp = FastMCP("Calculator Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    result = a + b
    print(f"[MCP SERVER DEBUG] add called with a={a}, b={b} -> {result}", file=sys.stderr, flush=True)
    return result

if __name__ == "__main__":
    print("[MCP SERVER DEBUG] Starting MCP server: Calculator Server", file=sys.stderr, flush=True)
    mcp.run()
