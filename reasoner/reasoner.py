from typing import Literal
import clingo
from pathlib import Path

class Reasoner: 
    def __init__(self):
        file_path = Path(__file__).parent / "theory.lp"
        with file_path.open("r") as file:
            self.theory = "".join(file.readlines())
    
    def reason(self, factors: dict[str, str], abstract_factors: dict[str, str], roots: dict[str, str], links: dict[tuple[str, str], Literal["support", "attack"]]) -> bool:
        ctl = clingo.Control()
        grounding_parts = [("theory", []), ("factors",[]), ("afactors",[]), ("roots",[]), ("links",[])]
        factors_prog = "".join([f"factor(\"{factor}\")." for factor in factors])
        afactors_prog= "".join([f"afactor(\"{factor}\")." for factor in abstract_factors])
        roots_prog= "".join([f"root(\"{r}\")." for r in roots])
        links_list= [] 
        for nodes, rel in links.items():
            src, dst = nodes 
            rel_val = 0
            print(rel)
            if rel == "attack":
                rel_val = 1
            else: 
                rel_val = 0
            links_list.append(f"link(\"{src}\",{rel_val},\"{dst}\").") 
        
        links_prog = "".join(links_list)

        ctl.add("theory", [], self.theory) 
        ctl.add("factors", [], factors_prog) 
        ctl.add("afactors", [], afactors_prog) 
        ctl.add("roots", [], roots_prog) 
        ctl.add("links", [], links_prog) 
        ctl.ground(grounding_parts, context=Context())
        handle = ctl.solve(yield_=True)
        for m in handle:
            print(m.symbols(atoms=True))
        return True
        

class Context:
     def id(self, x):
         return x
     def seq(self, x, y):
         return [x, y]

if __name__ == "__main__":
    reasoner = Reasoner()
    reasoner.reason({"t0": "test ing factor"}, {"a0": "test afactor"}, {"r1": "test factor"}, {("a0", "t0"): "support", ("a0", "r1"): "attack"})
