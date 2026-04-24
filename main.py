import time
import signal
import sys
import termios
import atexit
import os
import queue
import threading

from engine import ChatEngine
from summarizer import summarize_blocking
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

GEMMA = "./models/gemma-4-E4B-it-Q4_K_M.gguf"
SMOL_MODEL = "./models/smollm2-1.7b-instruct-q4_k_m.gguf"

console = Console()

stop_streaming = threading.Event()
exit_all = threading.Event()


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


def handle_sigint(signum, frame):
    stop_streaming.set()


def handle_sigtstp(signum, frame):
    stop_streaming.set()
    exit_all.set()
    _set_echoctl(True)


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


if __name__ == "__main__":

    # ---------------- Model selection ----------------
    if os.path.exists(GEMMA):
        CHAT_MODEL = GEMMA
    elif os.path.exists(SMOL_MODEL):
        console.print("[yellow]Gemma not found — using Smol[/yellow]")
        CHAT_MODEL = SMOL_MODEL
    else:
        console.print("[red]No model found in ./models/ — run install.sh[/red]")
        exit(1)

    with console.status("[bold yellow]Loading model...", spinner="dots"):
        engine = ChatEngine(CHAT_MODEL)

    console.clear()
    console.print("[bold cyan]❯ Terminal Online.[/bold cyan]")

    # Warmup
    with console.status("[bold yellow]Warming up...", spinner="dots"):
        warmup = [{"role": "user", "content": "Hi"}]
        for _ in engine.generate_response(warmup):
            pass

    console.clear()
    console.print("[bold cyan]❯ Terminal Online.[/bold cyan]")

    _set_echoctl(False)
    atexit.register(_set_echoctl, True)
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTSTP, handle_sigtstp)

    chat_history = []

    # ---------------- Main loop ----------------
    while True:
        try:
            if exit_all.is_set():
                break

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

            # ---------------- Thinking ----------------
            with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
                first_chunk = q.get()

            if first_chunk is None or "__error__" in first_chunk:
                chat_history.pop()
                continue

            # ---------------- Streaming output ----------------
            with console.screen():
                full_response = first_chunk["choices"][0]["text"]

                with Live(Markdown(full_response), console=console, refresh_per_second=10) as live:
                    while True:
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

            elapsed = time.time() - start_time
            console.print(f"[dim]Time: {elapsed:.2f}s[/dim]")

            chat_history.append({"role": "model", "content": full_response})

            # ---------------- MEMORY COMPACTION ----------------
            history_text = "".join([m['content'] for m in chat_history])
            estimated_tokens = len(history_text) // 4

            if estimated_tokens >= 500:
                console.print(f"[dim]⚡ Context at ~{estimated_tokens} tokens. Summarizing...[/dim]")

                summary = summarize_blocking(chat_history, SMOL_MODEL)

                console.print(f"[bold green]✅ Memory compacted.[/bold green]")
                console.print(f"[dim]💾 Summary content: {summary}[/dim]")

                # Keep summary + last exchange
                last_user = None
                last_model = None

                for msg in reversed(chat_history):
                    if msg['role'] == "model":
                        last_model = msg
                    elif msg['role'] == "user":
                        last_user = msg
                        break

                new_history = []

                if summary:
                    new_history.append({
                        "role": "context",
                        "content": summary
                    })

                if last_user:
                    new_history.append(last_user)

                if last_model:
                    new_history.append(last_model)

                chat_history = new_history

        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[bold cyan]Goodbye.[/bold cyan]")
