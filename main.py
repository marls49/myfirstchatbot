from chatbot import build_chatbot


def run_chat():
    bot = build_chatbot()
    print("Simple kNN-Embedding-Chatbot. Type 'exit' to quit.")
    while True:
        user_inp = input("You: ").strip()
        if user_inp.lower() in {"exit", "quit"}:
            break
        print("Bot:", bot.get_best_answer(user_inp))


if __name__ == "__main__":
    run_chat()
