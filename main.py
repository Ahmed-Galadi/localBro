import time
import threading
import queue
import signal
import sys
import termios
import atexit
import os
import readline
import argparse
from engine import ChatEngine
from summarizer import compress
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

# Paths
GEMMA = "./models/gemma-4-E4B-it-Q4_K_M.gguf"
QWEN  = "./models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

# ---------------- Argument Parsing ----------------
parser = argparse.ArgumentParser(description="LocalBro CLI")
parser.add_argument("--fast", action="store_true", help="Use the smaller Qwen model")
args = parser.parse_args()

console = Console()

# ---------------- Model Selection ----------------
if args.fast:
    if os.path.exists(QWEN):
        CHAT_MODEL = QWEN
        console.print("[yellow]Fast mode enabled — using Qwen 2.5 0.5B[/yellow]")
    else:
        console.print(f"[red]Error: Qwen model not found at {QWEN}[/red]")
        exit(1)
else:
    if os.path.exists(GEMMA):
        CHAT_MODEL = GEMMA
    elif os.path.exists(QWEN):
        console.print("[yellow]Gemma not found — falling back to Qwen 2.5 0.5B[/yellow]")
        CHAT_MODEL = QWEN
    else:
        console.print("[red]No models found in ./models/ — run install.sh[/red]")
        exit(1)

COMPRESS_AFTER = 3

with console.status("[bold yellow]Loading model...", spinner="dots"):
    engine = ChatEngine(CHAT_MODEL)

# ---------------- Warmup inference ----------------
with console.status("[bold yellow]Warming up...", spinner="dots"):
    warmup_messages = [{"role": "user", "content": "Hi"}]
    warmup_stream = engine.generate_response(warmup_messages)
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

# ---------------- Suppress ^C / ^Z echo ----------------
def _set_echoctl(enable: bool):
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        if enable:
            attrs[3] |= termios.ECHOCTL
        else:
            attrs[3] &= ~termios.ECHOCTL
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        pass

_set_echoctl(False)
atexit.register(_set_echoctl, True)

def handle_sigint(signum, frame):
    stop_streaming.set()

def handle_sigtstp(signum, frame):
    stop_streaming.set()
    exit_all.set()
    _set_echoctl(True)

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

        user_input = input("\n\033[1;32m❯ \033[0m").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        chat_history.append({"role": "user", "content": user_input})

        stop_streaming.clear()
        q = queue.Queue(maxsize=200)

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

        # ---------------- RESPONDING PHASE ----------------
        with console.screen() as screen:
            full_response = first_chunk["choices"][0]["text"]
            
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
                    
                    full_response += chunk["choices"][0]["text"]
                    live.update(Markdown(full_response))

        console.print("[bold blue]assistant:[/bold blue]")
        console.print(Markdown(full_response))

        if stop_streaming.is_set():
            console.print("[yellow]⚠ Response interrupted (Ctrl+C)[/yellow]")

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
