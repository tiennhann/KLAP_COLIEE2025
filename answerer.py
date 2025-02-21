from extractor.extractor import MDAExtractor
from reasoner.reasoner import Reasoner

class Answerer:
    def __init__(self):
        self.extractor= MDAExtractor()
        self.reasoner = Reasoner()

    def answer(self, text):
        facts = self.extractor.forward(text=text)
        return self.reasoner.reason(facts)

if __name__ == "__main__":
    text = ""
    with open("./test.txt", "r") as f:
        text = "".join(f.readlines())
    answerer = Answerer()
    answer = answerer.answer(text)
    print(answer)