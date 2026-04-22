from engine import ChatEngine

def background_summarizer(conn, chat_history, model_path):
    """
    This function runs in a separate process.
    It uses Qwen to generate a professional summary of the chat.
    """
    try:
        # Load Qwen inside the child process.
        # We use low threads (2) to avoid starving Gemma of CPU cycles.
        mem_engine = ChatEngine(model_path)
        mem_engine.llm.n_threads = 2
        mem_engine.llm.n_threads_batch = 2

        # Extract only user questions to keep the summary focused
        user_queries = [m['content'] for m in chat_history if m['role'] == "user"]
        context_text = "\n".join(user_queries)

        prompt = [
            {"role": "user", "content": f"Summarize the following topics discussed into a logical, 3-line summary of facts:\n\n{context_text}"}
        ]

        # Generate the summary
        stream = mem_engine.generate_response(prompt)
        summary = ""
        for chunk in stream:
            summary += chunk["choices"][0]["text"]

        # Send back to the main process via Pipe
        conn.send(summary.strip())
    except Exception as e:
        conn.send(f"Error during background compression: {e}")
    finally:
        conn.close()
