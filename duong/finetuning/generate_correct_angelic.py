'''output data from Answerer json in the form:
{  id: ,
    articles: ,
    angelic_st: ,
    link: ,
    hypothesis: ,
    hyp_rl_angelic: ,
    predicted_label: ,
    correct_label: ,

}
'''

from minh.deepseek import chat
from reasoner.reasoner import Reasoner
from extractor.extractor import MDAExtractor
prompt_template = "For the articles {{articles}} and hypothesis {{query}}, you generated the Angelic structure as follows: {{angelic_st}}. However, this structure contains incorrect links that {{predict}} the hypothesis. Please revise these links and generate the correct structure."
'''
if true_label == Y and predicted_label = N ==> {{predict}} = invalidate
if true_label == N and predicted_label = Y ==> {{predict}} = falsely validate
'''
# loop through Answerer data 
    # while=True until predicted_label == correct_label or nb_loop == 5
        # use prompt in prompt_to_label
    # save data as correct angelic_st
class Gen_refined_angelic_st:
    def __init__(self, datas):
        self.angelicExtractor = chat()
        self.labelPredictor = Reasoner()
        self.aspTranslator = MDAExtractor()
        self.refined_output = []
    
    #strat 0
    def refined(self, data):
        refined_count = 0
        # angelic_st = self.angelicExtractor(article, query)
        # facts = self.aspTranslator.forward(text=angelic_st)
        true_label = data["correct_label"]
        predict_label = data["predicted_label"]
        if true_label == "Y" and predict_label =="N":
            predict_prompt = "invalidate"
        if true_label == "N" and predict_label =="Y":
            predict_prompt = "falsely validate"
            
        if true_label != predict_label:
            wrong_label = True
            prompt = prompt_template.replace("{{articles}}", data["articles"]).replace("{{query}}", data["hypothesis"]).replace("{{angelic_st}}", data["angelic_st"]).replace("{predict}", predict_prompt)
        else:
            wrong_label = False
        
        while wrong_label == True and refined_count < 5:  
            refined_angelic_st = self.angelicExtractor(prompt)
            facts = self.aspTranslator(refined_angelic_st)
            predict_label = self.labelPredictor(facts)
            if predict_label == true_label: 
                wrong_label == False
            
            if predict_label != true_label:
                data["angelic_st"] = refined_angelic_st
            self.refined_output.append(data)
            
            refined_count +=1 
    def refined_all(self, datas):
        for data in datas:
            self.refined(data)