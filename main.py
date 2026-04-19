import time
from engine import ChatEngine
from summarizer import compress
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
import os

GEMMA = "./models/gemma-4-E4B-it-Q4_K_M.gguf"
QWEN  = "./models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

console = Console()

if os.path.exists(GEMMA):
    CHAT_MODEL = GEMMA
elif os.path.exists(QWEN):
    console.print("[yellow]Gemma not found — using Qwen 2.5 0.5B[/yellow]")
    CHAT_MODEL = QWEN
else:
    console.print("[red]No model found in ./models/ — run install.sh[/red]")
    exit(1)
COMPRESS_AFTER = 3



with console.status("[bold yellow]Loading Gemma...", spinner="dots"):
    engine = ChatEngine(CHAT_MODEL)

console.clear()
console.print("[bold cyan]❯ Terminal Online.[/bold cyan]")

chat_history: list = []
exchanges: int = 0

while True:
    try:
        user_input = console.input("\n[bold green]❯ [/bold green]").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        chat_history.append({"role": "user", "content": user_input})

        start = time.time()
        full_response = ""

        with console.status("[bold blue]Thinking...", spinner="bouncingBar"):
            stream = engine.generate_response(chat_history)
            try:
                first_chunk = next(stream)
                full_response += first_chunk['choices'][0]['text']
            except StopIteration:
                chat_history.pop()
                continue

        console.print("[bold blue]assistant:[/bold blue]")
        with Live(console=console, refresh_per_second=10) as live:
            for chunk in stream:
                full_response += chunk['choices'][0]['text']
                live.update(Markdown(full_response))

        console.print(f"[dim]Time: {time.time() - start:.2f}s[/dim]")

        chat_history.append({"role": "model", "content": full_response})
        exchanges += 1

        if exchanges >= COMPRESS_AFTER:
            old_summary = ""
            if chat_history and chat_history[0]['role'] == "context":
                old_summary = chat_history[0]['content']

            new_summary = compress(chat_history, old_summary)
            chat_history = [{"role": "context", "content": new_summary}]
            exchanges = 0

            console.print("[dim italic]Memory compressed ✓[/dim italic]")

    except KeyboardInterrupt:
        break
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        continue
