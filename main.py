import time
import threading
import queue
import signal
import sys
import termios
import atexit
import os
import readline
from multiprocessing import Process, Pipe

from engine import ChatEngine
from summarizer import background_summarizer
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

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

COMPRESS_AFTER = 4

with console.status("[bold yellow]Loading model...", spinner="dots"):
    engine = ChatEngine(CHAT_MODEL)

# ---------------- IPC Setup ----------------
parent_conn, child_conn = Pipe()
worker_process = None

console.clear()
console.print("[bold cyan]❯ Terminal Online (Twin-Engine Active).[/bold cyan]")

chat_history = []
exchanges = 0
stop_streaming = threading.Event()
exit_all = threading.Event()

# ---------------- Warmup inference (CPU/GPU agnostic) ----------------
with console.status("[bold yellow]Warming up (first inference)...", spinner="dots"):
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

        # Check if background summarizer finished
        if parent_conn.poll():
            new_summary = parent_conn.recv()
            chat_history = [{"role": "context", "content": new_summary}]
            console.print(f"[dim italic]Memory updated by Qwen: {new_summary[:50]}... ✓[/dim italic]")

        user_input = input("\n\033[1;32m❯ \033[0m").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        chat_history.append({"role": "user", "content": user_input})
        stop_streaming.clear()
        q = queue.Queue(maxsize=200)
        start_time = time.time()

        stream = engine.generate_response(chat_history)
        worker = threading.Thread(target=stream_worker, args=(stream, q, stop_streaming), daemon=True)
        worker.start()

        # ---------------- THINKING PHASE ----------------
        with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
            first_chunk = q.get()

        if first_chunk is None or "__error__" in first_chunk:
            chat_history.pop()
            continue

        # ---------------- RESPONDING PHASE (Alternate Buffer) ----------------
        with console.screen() as screen:
            full_response = first_chunk["choices"][0]["text"]
            with Live(Markdown(full_response), console=console, refresh_per_second=10, vertical_overflow="visible") as live:
                while True:
                    if exit_all.is_set():
                        stop_streaming.set()
                        break
                    try:
                        chunk = q.get(timeout=0.1)
                    except queue.Empty:
                        if stop_streaming.is_set(): break
                        continue
                    if chunk is None: break
                    full_response += chunk["choices"][0]["text"]
                    live.update(Markdown(full_response))

        console.print("[bold blue]assistant:[/bold blue]")
        console.print(Markdown(full_response))

        elapsed = time.time() - start_time
        console.print(f"[dim]Time: {elapsed:.2f}s[/dim]")

        chat_history.append({"role": "model", "content": full_response})
        exchanges += 1

        # ---------------- ASYNC MEMORY COMPRESSION ----------------
        if exchanges >= COMPRESS_AFTER:
            if worker_process and worker_process.is_alive():
                pass # Already summarizing
            else:
                worker_process = Process(target=background_summarizer, args=(child_conn, chat_history, QWEN))
                worker_process.start()
                exchanges = 0

    except EOFError:
        break
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        continue

if worker_process: worker_process.terminate()
console.print("\n[bold cyan]Goodbye.[/bold cyan]")
