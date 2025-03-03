from extractor.extractor import MDAExtractor
from minh.deepseek import chat
from reasoner.reasoner import Reasoner
import dspy

class ExtractorException(Exception):
    def __inint__(self, reply, explanation=None):
        super.__init__(reply, explanation)

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
    
    def retry_extractor(self, article, query, failed_answer):
        reply = chat(article, query, "")
        if self.debug:
            print("======== Angelic extraction step: ======== \n", reply)
        facts = self.extractor.forward(text=reply, failed_answer=failed_answer)
        if self.debug:
            print("======== ASP facts conversion step: ======= \n", facts)
        try:
            answer = self.reasoner.reason(facts)
            return reply, facts, answer 
        except Exception as e:
            raise ExtractorException(facts, str(e))

    def retry_chat(self, article, query, failed_answer):
        reply = chat(article, query, failed_answer)
        if self.debug:
            print("======== Angelic extraction step: ======== \n", reply)
        facts = self.extractor.forward(text=reply, failed_answer=None)
        if self.debug:
            print("======== ASP facts conversion step: ======= \n", facts)
        try:
            answer = self.reasoner.reason(facts)
            return reply, answer 
        except Exception as e:
            raise ExtractorException(reply)


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
