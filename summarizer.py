from engine import ChatEngine

def persistent_summarizer(work_conn, result_conn, model_path):
    try:
        mem_engine = ChatEngine(model_path, n_threads=2)
        result_conn.send("__ready__")

        while True:
            chat_history = work_conn.recv()
            if chat_history is None:
                break

            user_queries = [m['content'] for m in chat_history if m['role'] == "user"]
            context_text = "\n".join(user_queries)
            context_text = context_text[:3000:]

            # Raw prompt, no system instruction interference
            raw_prompt = (
                f"<|im_start|>system\n"
                f"You are a summarizer. Output only bullet points. No extra text.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Summarize each topic in one compact sentence using your own words:\n\n"
                f"{context_text}\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"- "
            )

            stream = mem_engine.llm(
                raw_prompt,
                max_tokens=384,
                stream=True,
                stop=["<|im_end|>", "<|endoftext|>"]
            )

            summary = ""
            for chunk in stream:
                summary += chunk["choices"][0]["text"]

            result_conn.send(summary.strip())

    except Exception as e:
        result_conn.send(f"Error: {e}")
