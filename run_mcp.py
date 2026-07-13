"""Entry point for the Slate Memory MCP server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
