import time
import threading
import queue
import signal
from engine import ChatEngine
from summarizer import compress
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
import os

GEMMA = "./models/gemma-4-E4B-it-Q4_K_M.gguf"
QWEN  = "./models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

console = Console()

# ---------------- Model selection ----------------
if os.path.exists(GEMMA):
    CHAT_MODEL = GEMMA
elif os.path.exists(QWEN):
    console.print("[yellow]Gemma not found — using Qwen 2.5 0.5B[/yellow]")
    CHAT_MODEL = QWEN
else:
    console.print("[red]No model found in ./models/ — run install.sh[/red]")
    exit(1)

COMPRESS_AFTER = 3

with console.status("[bold yellow]Loading model...", spinner="dots"):
    engine = ChatEngine(CHAT_MODEL)

# ---------------- Warmup inference (CPU/GPU agnostic) ----------------
with console.status("[bold yellow]Warming up (first inference)...", spinner="dots"):
    # A dummy prompt that forces internal state initialization
    warmup_messages = [{"role": "user", "content": "Hi"}]
    warmup_stream = engine.generate_response(warmup_messages)
    # Consume a few tokens to fully exercise the pipeline
    for _ in range(5):
        try:
            next(warmup_stream)
        except StopIteration:
            break

console.clear()
console.print("[bold cyan]❯ Terminal Online.[/bold cyan]")

chat_history = []
exchanges = 0

# ---------------- Control flags ----------------
stop_streaming = threading.Event()
exit_all = threading.Event()

def handle_sigint(signum, frame):
    stop_streaming.set()

def handle_sigtstp(signum, frame):
    stop_streaming.set()
    exit_all.set()

signal.signal(signal.SIGINT, handle_sigint)    # Ctrl+C
signal.signal(signal.SIGTSTP, handle_sigtstp)  # Ctrl+Z


def stream_worker(stream_gen, out_queue, stop_event):
    try:
        for chunk in stream_gen:
            if stop_event.is_set():
                break
            out_queue.put(chunk)
    except Exception as e:
        out_queue.put({"__error__": str(e)})
    finally:
        out_queue.put(None)


# ---------------- Main loop ----------------
while True:
    try:
        if exit_all.is_set():
            break

        user_input = console.input("\n[bold green]❯ [/bold green]").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        chat_history.append({"role": "user", "content": user_input})

        stop_streaming.clear()
        q = queue.Queue(maxsize=200)

        # Start timing the response generation
        start_time = time.time()

        stream = engine.generate_response(chat_history)
        worker = threading.Thread(
            target=stream_worker,
            args=(stream, q, stop_streaming),
            daemon=True
        )
        worker.start()

        # ---------------- THINKING PHASE ----------------
        with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
            first_chunk = q.get()

        if first_chunk is None:
            chat_history.pop()
            continue

        if "__error__" in first_chunk:
            console.print(f"[red]Error: {first_chunk['__error__']}[/red]")
            chat_history.pop()
            continue

        full_response = first_chunk["choices"][0]["text"]

        # ---------------- RESPONDING PHASE ----------------
        console.print("[bold blue]assistant:[/bold blue]")

        with Live(Markdown(full_response),
                  console=console,
                  refresh_per_second=10,
                  vertical_overflow="visible") as live:

            while True:
                if exit_all.is_set():
                    stop_streaming.set()
                    break

                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    if stop_streaming.is_set():
                        break
                    continue

                if chunk is None:
                    break

                if "__error__" in chunk:
                    console.print(f"[red]Error: {chunk['__error__']}[/red]")
                    break

                full_response += chunk["choices"][0]["text"]
                live.update(Markdown(full_response))

        # ---------------- INTERRUPTION FEEDBACK ----------------
        if stop_streaming.is_set():
            console.print("[yellow]⚠ Response interrupted (Ctrl+C)[/yellow]")

        # Show elapsed time in seconds (human‑readable duration)
        elapsed = time.time() - start_time
        console.print(f"[dim]Time: {elapsed:.2f}s[/dim]")

        chat_history.append({"role": "model", "content": full_response})
        exchanges += 1

        # ---------------- MEMORY COMPRESSION ----------------
        if exchanges >= COMPRESS_AFTER:
            old_summary = ""
            if chat_history and chat_history[0]["role"] == "context":
                old_summary = chat_history[0]["content"]

            new_summary = compress(chat_history, old_summary)
            chat_history = [{"role": "context", "content": new_summary}]
            exchanges = 0

            console.print("[dim italic]Memory compressed ✓[/dim italic]")

        if exit_all.is_set():
            break

    except EOFError:
        break

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        stop_streaming.set()
        continue

console.print("\n[bold cyan]Goodbye.[/bold cyan]")