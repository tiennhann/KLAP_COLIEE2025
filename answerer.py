from extractor.extractor import MDAExtractor
from minh.deepseek import chat
from reasoner.reasoner import Reasoner
import dspy


class Answerer:
    def __init__(self, debug=False):
        self.extractor = MDAExtractor()
        self.reasoner = Reasoner()
        self.debug = debug

    def answer(self, article, query):
        reply = chat(article, query)
        if self.debug:
            print("======== Angelic extraction step: ======== \n", reply)
        facts = self.extractor.forward(text=reply)
        if self.debug:
            print("======== ASP facts conversion step: ======= \n", facts)
        return self.reasoner.reason(facts)


if __name__ == "__main__":
    dspy.settings.configure(
        lm=dspy.LM(
            model="ollama_chat/llama3.3",  # Changed to include provider prefix
            api_base="http://localhost:11435",
            max_tokens=20000,
        )
    )
    text = ""
    with open("./test.txt", "r") as f:
        text = "".join(f.readlines())
    answerer = Answerer(debug=True)
    answer = answerer.answer(text, "In cases where an individual rescues another person from getting hit by a car by pushing that person out of the way, causing the person's luxury kimono to get dirty, the rescuer does not have to compensate damages for the kimono.")
    print(answer)
