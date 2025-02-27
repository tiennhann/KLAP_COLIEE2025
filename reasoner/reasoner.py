from typing import Literal
import clingo
from pathlib import Path

class Reasoner: 
    def __init__(self):
        file_path = Path(__file__).parent / "theory_2.lp"
        with file_path.open("r") as file:
            self.theory = "".join(file.readlines())
    
    def reason(self, facts) -> bool:
        ctl = clingo.Control()
        grounding_parts = [("theory", []), ("facts",[])]

        ctl.add("theory", [], self.theory) 
        ctl.add("facts", [], facts) 
        ctl.ground(grounding_parts, context=Context())
        handle = ctl.solve(yield_=True)
        for m in handle:
            symbols = m.symbols(atoms=True)
            for sym in symbols:
                if str(sym).lower() == "yes":
                    return True 
        return False 
        
class Context:
     def id(self, x):
         return x
     def seq(self, x, y):
         return [x, y]

if __name__ == "__main__":
    reasoner = Reasoner()
    facts = ""
    with open("./test.lp", "r") as f:
        facts = "".join(f.readlines())
    reasoner.reason(facts)
