from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup,TrainingArguments
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import re

device = 'cuda'

peft_model_id = "snegha24/hindi_pt"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to(device)
model = PeftModel.from_pretrained(model, peft_model_id).to(device)



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", trust_remote_code=True)

pattern = r'\b[1-4]\b'
def getInt(s):
    file = open('output.txt', 'a') 
    file.write(str(s)+"\n")
    file.close() 
    match = re.search(pattern, s)
    if match:
        return int(match.group())
    else:
        return None



def inference(input_prompt, model, tokenizer):

    encodings = tokenizer(input_prompt,return_tensors="pt",return_token_type_ids=False).to(device)
    encodings = encodings.to(device)

    
    outputs = model.generate(**encodings, do_sample=False, max_new_tokens=250)
    output_text = tokenizer.batch_decode(outputs)
    file = open('outputf.txt', 'a') 
    file.write(str(output_text[0])+"\n ******************* \n")
    file.close()
    output_texts = output_text[0][len(input_prompt):]
    return output_texts

l = {}
instr = "### Instruction: \nThe task is to perform reading comprehension task. Given the following passage, query, and answer choices, output the number corresponding to the correct answer.\n"

for lang in ["hin_Deva"]:
    print(lang)
    a = 0
    corr = 0
    tot = 0
    ambig = 0
    dataset = load_dataset("facebook/belebele", lang)
    dataset = dataset['test']
    for inst in dataset:
        a += 1
        prompt = '### Input: \n' + 'Passage: ' + inst["flores_passage"] +  "\n" + 'Query: ' + inst["question"] + "\n"
        choices = "Choices:\n1: " + inst["mc_answer1"] + "\n2: " + inst["mc_answer2"] + "\n3: " + inst["mc_answer3"] + "\n4: " + inst["mc_answer4"]
        inp_prompt = instr + prompt+choices + '\n###Response: \n'
        outputs = inference(inp_prompt,model,tokenizer)
        intv = getInt(outputs)
        if intv == int(inst['correct_answer_num']):
            corr += 1
        elif intv is None:
            ambig += 1
        tot += 1 
      
    acc =  corr/tot*100  
    l[lang] = [corr,ambig,tot, acc]
    print(lang,"Accuracy:",corr/tot*100)


import pandas as pd

df = pd.DataFrame(l)
df = df.T
df.to_csv("output.csv")

