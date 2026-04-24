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
                f"<start_of_turn>user\n"
                f"The user asked about these topics. Write a single compact sentence per topic using your own words. Do not copy the original text.\n\n"
                f"{context_text}\n"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
                f"- "
            )
            stream = mem_engine.llm(
                raw_prompt,
                max_tokens=384,
                stream=True,
                stop=["<end_of_turn>", "<eos>"]
            )

            summary = ""
            for chunk in stream:
                summary += chunk["choices"][0]["text"]

            result_conn.send(summary.strip())

    except Exception as e:
        result_conn.send(f"Error: {e}")
