from llama_cpp import Llama


class ChatEngine:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=6,
            n_threads_batch=6,
            n_gpu_layers=-1,
            verbose=False
        )

    def generate_response(self, messages: list):
        # Pinned system instruction
        prompt = (
            "<start_of_turn>user\n"
            "INSTRUCTIONS: You are a knowledgeable offline assistant. "
            "Primary goals: Explain educational concepts clearly and assist with daily tasks. "
            "Be direct, concise, and logical. Use Markdown for clarity. No TEX. No fluff. "
            "Keep answers short and to the point. Max 20 lines unless complexity requires more. "
            "Respond only as the model.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "Understood.\n"
            "<end_of_turn>\n"
        )

        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == "context":
                # Compressed memory injected as a grounding fact, not a hallucinated model utterance.
                prompt += (
                    "<start_of_turn>user\n"
                    f"[Memory from earlier conversations — treat as established fact]:\n{content}\n"
                    "<end_of_turn>\n"
                    "<start_of_turn>model\n"
                    "Memory noted.\n"
                    "<end_of_turn>\n"
                )
            elif role == "user":
                prompt += f"<start_of_turn>user\n{content}\n<end_of_turn>\n"
            else:  # "model"
                prompt += f"<start_of_turn>model\n{content}\n<end_of_turn>\n"

        prompt += "<start_of_turn>model\n"

        return self.llm(
            prompt,
            max_tokens=1024,
            stream=True,
            stop=["<end_of_turn>", "<eos>", "<|end_of_turn|>", "<start_of_turn>"],
            repeat_penalty=1.2,
            temperature=0.2
        )
