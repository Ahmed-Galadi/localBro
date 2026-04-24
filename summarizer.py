from engine import ChatEngine

def summarize_blocking(chat_history, model_path):
    # Note: Loading the model from disk every time this function is called is highly inefficient. 
    # You should load this once globally or pass the engine instance to avoid RAM thrashing.
    mem_engine = ChatEngine(model_path, n_threads=2)

    # 1. Build and ACTUALLY use the conversation text
    conversation_lines = []
    for m in chat_history:
        if m['role'] == "user":
            prefix = "User:"
        elif m['role'] == "model":
            prefix = "Assistant:"
        else:
            prefix = "Memory:"
        conversation_lines.append(f"{prefix} {m['content']}")

    # Use the last 2000 characters of the properly formatted dialogue
    context_text = "\n".join(conversation_lines)[-2000:]

    # 2. Use ChatML tags for SmolLM2 and strictly delimit the context
    prompt = (
        "<|im_start|>system\n"
        "You are a strict summarization engine. Do NOT answer questions. ONLY summarize the text provided.<|im_end|>\n"
        "<|im_start|>user\n"
        "Summarize the conversation below in ONE sentence (max 15 words).\n\n"
        "--- CONVERSATION START ---\n"
        f"{context_text}\n"
        "--- CONVERSATION END ---\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    stream = mem_engine.llm(
        prompt,
        max_tokens=32,
        stream=True,
        stop=["<|im_end|>", "<eos>"] # Ensure it stops on SmolLM's end token
    )

    summary = ""
    for chunk in stream:
        summary += chunk["choices"][0]["text"]

    return summary.strip()
