from qa_engine import answer_question

print("📘 Offline Lecture AI Assistant")
print("Answers are based ONLY on the lecture.")
print("Type 'exit' to quit.\n")

while True:
    question = input("Your question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = answer_question(question)

    print("\nShort Answer:")
    print(result["short"])

    if result["full"]:
        print("\nFrom Lecture (Context):")
        print(result["full"])

    print("-" * 60)
