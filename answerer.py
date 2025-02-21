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
            print("Deepseek step", reply)
        facts = self.extractor.forward(text=reply)
        if self.debug:
            print("Extractor step", facts)
        return self.reasoner.reason(facts)


if __name__ == "__main__":
    dspy.settings.configure(
        lm=dspy.LM(
            model="ollama_chat/llama3:8b",  # Changed to include provider prefix
            api_base="http://localhost:11434",
            max_tokens=20000,
        )
    )
    text = ""
    with open("./test.txt", "r") as f:
        text = "".join(f.readlines())
    answerer = Answerer(debug=True)
    answer = answerer.answer(text, "If a person under curatorship performs, without getting the consent, an act that requires  getting consent from his/her curator, the curator may ratify that act, but may not rescind that act")
    print(answer)
