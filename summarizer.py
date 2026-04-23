def compress(chat_history: list, old_summary: str = "") -> str:
    """
    Keeps only user questions as memory — they carry all the context.
    Assistant responses are discarded (long, noisy, not needed).
    """
    new_questions = [
        f"- {m['content']}"
        for m in chat_history
        if m['role'] == "user"
    ]

    old_lines = old_summary.strip().splitlines() if old_summary else []

    all_lines = old_lines + new_questions

    # Cap at 20 questions so context block never explodes
    all_lines = all_lines[-20:]

    return "Previous topics discussed:\n" + "\n".join(all_lines)
