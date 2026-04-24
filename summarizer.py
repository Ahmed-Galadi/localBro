from engine import ChatEngine

def persistent_summarizer(work_conn, result_conn, model_path):
    """
    Loads Qwen once at startup, then waits in a loop for work.
    Receives chat_history, sends back a summary.
    """
    try:
        mem_engine = ChatEngine(model_path, n_threads=2)
        result_conn.send("__ready__")  # signal main that Qwen is loaded

        while True:
            chat_history = work_conn.recv()  # blocks here until work arrives

            if chat_history is None:  # shutdown signal
                break

            user_queries = [m['content'] for m in chat_history if m['role'] == "user"]
            context_text = "\n".join(user_queries)
            context_text = context_text[-3000:]

            prompt = [
                {"role": "user", "content": f"Summarize the following topics discussed into a logical, 3-line summary of facts:\n\n{context_text}"}
            ]

            stream = mem_engine.generate_response(prompt)
            summary = ""
            for chunk in stream:
                summary += chunk["choices"][0]["text"]

            result_conn.send(summary.strip())

    except Exception as e:
        result_conn.send(f"Error during summarization: {e}")
