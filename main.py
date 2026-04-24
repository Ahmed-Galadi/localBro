import time
import threading
import queue
import signal
import sys
import termios
import atexit
import os
import readline
from multiprocessing import Process, Pipe, set_start_method

from engine import ChatEngine
from summarizer import persistent_summarizer
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

GEMMA = "./models/gemma-4-E4B-it-Q4_K_M.gguf"
QWEN  = "./models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

console = Console()

# ---------------- Control flags ----------------
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

def qwen_result_watcher(result_recv):
    while True:
        try:
            summary = result_recv.recv()
            console.print("\n[bold yellow]🔄 Qwen Summary Result:[/bold yellow]")
            console.print(f"[bold white]{summary}[/bold white]\n")
            qwen_busy.clear()
        except EOFError:
            break

if __name__ == "__main__":
    set_start_method("spawn")

    # ---------------- Model selection ----------------
    if os.path.exists(GEMMA):
        CHAT_MODEL = GEMMA
    elif os.path.exists(QWEN):
        console.print("[yellow]Gemma not found — using Qwen 2.5 0.5B[/yellow]")
        CHAT_MODEL = QWEN
    else:
        console.print("[red]No model found in ./models/ — run install.sh[/red]")
        exit(1)

    with console.status("[bold yellow]Loading Gemma...", spinner="dots"):
        engine = ChatEngine(CHAT_MODEL)

    # ---------------- Pipes for Qwen IPC ----------------
    # work pipe: main → Qwen (send chat_history)
    # result pipe: Qwen → main (receive summary)
    work_send, work_recv = Pipe()
    result_send, result_recv = Pipe()

    # ---------------- Spawn Qwen at startup ----------------
    with console.status("[bold yellow]Loading Qwen in background...", spinner="dots"):
        qwen_process = Process(
            target=persistent_summarizer,
            args=(work_recv, result_send, QWEN),
            daemon=True
        )
        qwen_process.start()
        ready_signal = result_recv.recv()  # blocks until Qwen sends "__ready__"

    console.clear()
    console.print("[bold cyan]❯ Terminal Online (Twin-Engine Active).[/bold cyan]")

    # ---------------- Warmup ----------------
    with console.status("[bold yellow]Warming up Gemma...", spinner="dots"):
        warmup_messages = [{"role": "user", "content": "Hi"}]
        warmup_stream = engine.generate_response(warmup_messages)
        for chunk in warmup_stream:
            pass

    console.clear()
    console.print("[bold cyan]❯ Terminal Online.[/bold cyan]")

    _set_echoctl(False)
    atexit.register(_set_echoctl, True)
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTSTP, handle_sigtstp)

    chat_history = []
    qwen_busy = threading.Event() # track if Qwen is currently summarizing


    watcher = threading.Thread(target=qwen_result_watcher, args=(result_recv,), daemon=True)
    watcher.start()
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

            # ---------------- THINKING PHASE ----------------
            with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
                first_chunk = q.get()

            if first_chunk is None or "__error__" in first_chunk:
                chat_history.pop()
                continue

            # ---------------- RESPONDING PHASE ----------------
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

            # ---------------- ASYNC MEMORY COMPRESSION ----------------
            history_text = "".join([m['content'] for m in chat_history])
            estimated_tokens = len(history_text) // 4

            if estimated_tokens >= 500 and not qwen_busy.is_set():
                console.print(f"[dim]⚡ Context at ~{estimated_tokens} tokens. Qwen is summarizing...[/dim]")
                work_send.send(chat_history)  # Qwen is already running, just send the work
                qwen_busy.set()

        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    # ---------------- Shutdown ----------------
    work_send.send(None)  # tell Qwen to exit its loop
    qwen_process.join(timeout=5)
    if qwen_process.is_alive():
        qwen_process.terminate()

    console.print("\n[bold cyan]Goodbye.[/bold cyan]")
