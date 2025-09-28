import asyncio
import os
import re
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Purely local model via ctransformers (GGUF)
try:
    from ctransformers import AutoModelForCausalLM as CTransformersModel
except Exception:  # pragma: no cover
    CTransformersModel = None  # type: ignore

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Load configuration from JSON file if present
        config: Dict[str, Any] = {}
        try:
            with open(os.path.join("config", "local_llm.json"), "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}

        # Resolve config with env fallbacks
        def get_flag(name: str, default: bool = False) -> bool:
            if name in config:
                val = str(config[name]).lower()
            else:
                val = os.getenv(name, str(default)).lower()
            return val in {"1", "true", "yes"}

        def get_value(name: str, default: Optional[str] = None) -> Optional[str]:
            if name in config and config[name] not in (None, ""):
                return str(config[name])
            return os.getenv(name, default)

        # Local-only configuration
        self.use_ctransformers = get_flag("USE_LOCAL_CTRANSFORMERS", True)
        local_gguf_path = get_value("LOCAL_GGUF_PATH")

        if self.use_ctransformers and local_gguf_path:
            if CTransformersModel is None:
                raise RuntimeError("ctransformers not installed. Run: pip install ctransformers")
            if not local_gguf_path:
                raise RuntimeError("Set LOCAL_GGUF_PATH to the full path of your .gguf file.")
            model_type = get_value("LOCAL_MODEL_TYPE", "mistral") or "mistral"
            gpu_layers = int(get_value("LOCAL_GPU_LAYERS", "0") or "0")
            self.ctransformers_model = CTransformersModel.from_pretrained(
                local_gguf_path,
                model_type=model_type,
                gpu_layers=gpu_layers,
            )
            self.use_ctransformers = True
        else:
            # No local config provided; fail early with guidance
            raise RuntimeError(
                "No local model configured. Set LOCAL_GGUF_PATH in config/local_llm.json to your .gguf file."
            )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\n[MCP DEBUG] Connected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using a local model (ctransformers)."""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()

        # Deterministic fallback: route common addition phrasings to MCP tool
        def parse_addition_query(text: str):
            patterns = [
                r"\bplease\s+add\s+(\d+)\s+(?:and|&|\+)\s+(\d+)\b",
                r"\badd\s+(\d+)\s+(?:and|&|\+)\s+(\d+)\b",
                r"\bsum\s+of\s+(\d+)\s+(?:and|&|\+)\s+(\d+)\b",
                r"\bwhat(?:'s|\s+is)?\s+(\d+)\s*\+\s*(\d+)\b",
                r"\b(\d+)\s*\+\s*(\d+)\b",
            ]
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    return int(m.group(1)), int(m.group(2))
            return None

        parsed = parse_addition_query(query)
        if parsed is not None:
            a_val, b_val = parsed
            print(f"[MCP DEBUG] Direct tool path: add({a_val}, {b_val})")
            result = await self.session.call_tool("add", {"a": a_val, "b": b_val})
            return str(getattr(result, "content", result))

        # Pure local GGUF via ctransformers: simple one-shot generation
        if getattr(self, "use_ctransformers", False):
            print("[MCP DEBUG] Local mode: plain generation (no tool protocol)")
            prompt = f"You are a helpful assistant.\nUser: {query}\nAssistant:"

            def _call() -> str:
                try:
                    out_local = self.ctransformers_model(
                        prompt,
                        max_new_tokens=256,
                        temperature=0.2,
                    )
                except TypeError:
                    out_local = self.ctransformers_model(prompt)
                return out_local if isinstance(out_local, str) else str(out_local)

            try:
                text = await asyncio.wait_for(asyncio.to_thread(_call), timeout=30)
            except asyncio.TimeoutError:
                return "Timed out while generating a response."
            return text

        # If we reach here, we are in pure local mode with tool protocol
        # (rest of ctransformers tool protocol block remains below)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())