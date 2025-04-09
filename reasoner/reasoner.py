from typing import Literal
import clingo
from pathlib import Path

class ASPSolver: 
    def __init__(self, theory_name=""):
        if theory_name == "":
            self.getlink_path = Path(__file__).parent / "theory_4.lp"
            self.getlabel_path = Path(__file__).parent / "theory_3.1.lp"
        else:
            file_path = Path(__file__).parent / theory_name
        # print("use asp encoding to get link:", getlink_path)
        # print("use asp encoding to get label", getlabel_path)
        with self.getlink_path.open("r") as file:
            self.theory_link = "".join(file.readlines())
        with self.getlabel_path.open("r") as file:
            self.theory_label = "".join(file.readlines())
    
    def GetLabel(self, facts) -> bool:
        ctl = clingo.Control(logger=self.log)
        grounding_parts = [("theory", []), ("facts",[])]

        ctl.add("theory", [], self.theory_label)
        ctl.add("facts", [], facts)
        ctl.ground(grounding_parts)
        handle = ctl.solve(yield_=True)
        for m in handle:
            symbols = m.symbols(atoms=True)
            for sym in symbols:
                if str(sym).lower() == "yes":
                    return True 
        # print(m)
        return False 
    
    def log(self, x, y):
        pass
    
    def GetLink(self, facts):
        ctl = clingo.Control(logger=self.log)
        grounding_parts = [("theory", []), ("facts",[])]
        ctl.add("theory", [], self.theory_link)
        ctl.add("facts", [], facts)
        ctl.ground(grounding_parts)
        optimal_model = None
        optimal_cost = None        
        with ctl.solve(yield_=True, async_=True) as handle:
            for model in handle:
                # Get the optimization cost of the current model
                current_cost = model.cost

                # If this is the first model or a better (lower) cost is found, update the optimal model
                if optimal_cost is None or current_cost < optimal_cost:
                    optimal_cost = current_cost
                    optimal_model = model
            
        # print('optimal cost:', optimal_cost)    
        # print('optimal model:', optimal_model)    
            if handle.get().satisfiable == True:
                return " ".join([i + '.' for i in str(optimal_model).split()])
            elif handle.get().satisfiable == False:
                return 'UNSAT'
            else: return 'Unknown'
        
    
if __name__ == "__main__":
    pass
    reasoner = ASPSolver()
    facts = "  node(bfactor(1..5)).query_entail(node(bfactor(1)), ex(1)).root(1, false, ex(1)).query_entail(node(bfactor(2)), ex(2)).query_entail(node(bfactor(3)), ex(2)).root(1, false, ex(2)).query_entail(node(bfactor(4)), ex(3)).root(1, true, ex(3)).query_entail(node(bfactor(5)), ex(4)).root(1, true, ex(4)).query_entail(node(bfactor(1)), ex(5)).query_entail(node(bfactor(2)), ex(5)).query_entail(node(bfactor(3)), ex(5)).query_entail(node(bfactor(4)), ex(5)).root(1, true, ex(5))."
    # with open("./test.lp", "r") as f:
    #     facts = "".join(f.readlines())
    print(reasoner.GetLink(facts))
