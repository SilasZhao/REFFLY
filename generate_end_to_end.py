import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import json
from tqdm import tqdm
import torch
import pickle
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import sys
from inference_util import *
sys.path.append("/home/songyan/Real_M2L-main/llama/")
from over_sample_reject import *
# from auto_eval import *
DEVICE = "cuda"
# MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
MODEL_NAME = "NousResearch/Llama-2-13b-chat-hf"
# OUTPUT_DIR = "/home/songyan/Real_M2L-main/finetune_llama/Finetune-Llama2-LoRA-main/out_with_negative_13b/checkpoint-23328"
# OUTPUT_DIR = "/data1/songyan/M2L/lyric/llama13b/with_negative/with_gpt_data/with_song_struct/checkpoint-11625/"
OUTPUT_DIR = "zzzsssyyy/Llama13bRevisingWithStruct"
SOURCE_FILE = "/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/song_4_2.json"
DEFAULT_SYSTEM_PROMPT = """
Below is the lyric of a song and the constraint. Rewrite the lyric to satisfy the constraint.
""".strip()
def get_input(f_names):
    all_lyrics = []
    lyric = ""
    for f_name in f_names:
        with open(f_name, 'r') as file:
            for line in file:
                # print(line)
                # Check each line against the pattern
                if "[" in line and "]" in line:
                    all_lyrics.append({"lyric":lyric.strip(),"constraint":eval(line.strip())})
                else:
                    lyric = line
    return all_lyrics
def generate_prompt(input, system_prompt = DEFAULT_SYSTEM_PROMPT):
   return f"""
          ### Instruction: {system_prompt}
   
          ### Input:
          {input}

          ### Response:
          """.strip()


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
)
rephrase_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
)

prefix_prompt_longer = """[INST]<<SYS>> You are a professional writer who is good at rephrasing sentences <</SYS>>
Rewrite a lyric line by adding some trivial details. Make the sentence longer. You should compose two rephrased sentences for the input lyrics. 

Lyrics to be rephrased: 
Dreamers seek mysteries.
[/INST]
Original: "Dreamers seek mysteries."
Rephrased1:"Visionaries pursue enigmas, yearning for discovery. "
Rephrased2: "Wanderers chase after the unknown, drawn by curiosity."

[INST]<<SYS>> You are a professional writer who is good at rephrasing sentences <</SYS>>
Rewrite a lyric line by adding some trivial details. Make the sentence longer. You should compose two rephrased sentences for the input lyrics. 

Lyrics to be rephrased: 
I've spent my life shining shoes at the station.
[/INST]
Original: "I've spent my life shining shoes at the station."
Rephrased1: "My days have been devoted to polishing footwear in the hustle of the terminal."
Rephrased2: "Life passed, buffing boots amidst the commotion of the station."

[INST]<<SYS>> You are a professional writer who is good at rephrasing sentences <</SYS>>
Rewrite a lyric line by adding some trivial details. Make the sentence longer. You should compose two rephrased sentences for the input lyrics. 

Lyrics to be rephrased: 
Began at nine, learning alone.
[/INST]
Original: "Began at nine, learning alone."
Rephrased1: "Started at dawn's light, mastering skills in solitude."
Rephrased2: "Commenced when the clock struck nine, self-taught in silence."

[INST]<<SYS>> You are a professional writer who is good at rephrasing sentences <</SYS>> 
Rewrite a lyric line by adding some trivial details. Make the sentence longer. You should compose two rephrased sentences for the input lyrics. 

Lyrics to be rephrased: 
"""
prefix_prompt_shorter = """[INST]<<SYS>> You are a professional writer who is good at rephrasing sentences <</SYS>>
Rewrite a lyric line by deleting some trivial details. Make the sentence shorter. You should compose two rephrased sentences for the input lyrics. 

Lyrics to be rephrased: 
Many people while away their time fantasizing about uncovering mysteries
[/INST]
Original: "Many people while away their time fantasizing about uncovering mysteries."
Rephrased1:"Many dream of solving mysteries."
Rephrased2: "Dreamers seek mysteries."

[INST]<<SYS>> You are a professional writer who is good at rephrasing sentences <</SYS>>
Rewrite a lyric line by deleting some trivial details. Make the sentence shorter. You should compose two rephrased sentences for the input lyrics. 

Lyrics to be rephrased: 
I've been stationed here at the railway station, polishing footwear
[/INST]
Original: "I've been stationed here at the railway station, polishing footwear"
Rephrased1: "Stationed at the rail, shining shoes."
Rephrased2: "At the station, polishing shoes."

[INST]<<SYS>> You are a professional writer who is good at rephrasing sentences <</SYS>>
Rewrite a lyric line by deleting some trivial details. Make the sentence shorter. You should compose two rephrased sentences for the input lyrics. 

Lyrics to be rephrased: 
I started when I was nine, on my own and taught myself
[/INST]
Original: "I started when I was nine, on my own and taught myself"
Rephrased1: "Started at nine, self-taught."
Rephrased2: "Began at nine, learning alone."

[INST]<<SYS>> You are a professional writer who is good at rephrasing sentences <</SYS>> 
Rewrite a lyric line by deleting some trivial details. Make the sentence shorter. You should compose two rephrased sentences for the input lyrics. 

Lyrics to be rephrased: 
"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
new_tokens = ['/STRESSED/', '/UNSTRESSED/']#/UNSTRESSED/-/STRESSED/
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
tokenizer.add_tokens(list(new_tokens))
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, OUTPUT_DIR)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
def get_result(lyrics,constraint,title,history_info):
    Input = create_input(lyrics,constraint,title) if history_info == "" else create_input(lyrics,constraint,title,history_info)
    print(f'Input created: \n {Input}')
    res = generate(model, generate_prompt(Input)+"\n          We want to generate a lyric with")
    item = {"prompt": Input, "res": res}
    print(res)
    res_sentence, mapping,total_and_incorrect = get_result_from_over_generation(item)
    print(f'res_sentence {res_sentence}, mapping {mapping} total_and_incorrect {total_and_incorrect}')
    return res_sentence,mapping,total_and_incorrect
def get_valid_pair(results):
    paired_lines = []
    for result in results:
        l = result.strip().split('\n')
        
        # print(result)
        # exit()
        # Iterating through the lines to find "Original" and "Rephrased" pairs
        for j in range(len(l) - 2):
            if l[j].strip().startswith('Original:') and l[j + 1].strip().startswith('Rephrased1:') and l[j + 2].strip().startswith('Rephrased2:'):
                original_line = l[j].strip()
                original_line = original_line.replace('"', '')
                original_line = original_line.replace('Original:', '')
                original_line = original_line.strip()
                rephrased_line_1 = l[j + 1].strip()
                rephrased_line_1 = rephrased_line_1.replace('"', '')
                rephrased_line_1 = rephrased_line_1.replace('Rephrased1:', '')
                rephrased_line_1 = rephrased_line_1.strip()
                rephrased_line_2 = l[j + 2].strip()
                rephrased_line_2 = rephrased_line_2.replace('"', '')
                rephrased_line_2 = rephrased_line_2.replace('Rephrased2:', '')
                rephrased_line_2 = rephrased_line_2.strip()
            
                paired_lines.append({"Original":original_line, "Rephrased1":rephrased_line_1,"Rephrased2":rephrased_line_2})
        return paired_lines
def get_rephrased_sentence(original,constraint):
    num_sy,_ = extract(original)
    if num_sy > len(constraint):
        prompt = prefix_prompt_shorter + original +"\n" + "[/INST]"
    else:
        prompt = prefix_prompt_longer+ original +"\n" + "[/INST]"
    results = generate(model, prompt,max_new_tokens=150)
    pair = get_valid_pair(results)
    if pair == []:
        print(f'failed get the rephrased sentence.')
        print(f'ourput {results}')
        return None
    else:
        rephrased1, rephrased2 = pair[0]["Rephrased1"], pair[0]["Rephrased2"]
        num_syllable1,_ = extract(rephrased1)
        num_syllable2,_ = extract(rephrased2)
        if abs(num_syllable1 - len(constraint)) <= abs(num_syllable2 - len(constraint)):
            rephrased = rephrased1
        else:
            rephrased = rephrased2
        return rephrased
def generate(model, prompt,max_new_tokens = 50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,num_beams=16,num_return_sequences = 16,num_beam_groups = 16,diversity_penalty = 1.2)
    # print(f'outputs shape is {outputs.shape}')
    # print([tokenizer.decode(out[inputs_length:], skip_special_tokens=True) for out in outputs])
    # print(tokenizer.decode(outputs[inputs_length:], skip_special_tokens=True))
    
    return [tokenizer.decode(out[inputs_length:], skip_special_tokens=True) for out in outputs]
 
def generate_songs(out_f,iteration = 5):
    with open(SOURCE_FILE,'r') as f:
        data = json.load(f)
    save_res = []
    for song in data:
        constraints = song["constraints"]
        lyrics = song["lyrics"]
        title = song["title"]
        lyrics = lyrics.split("\n")
        history_info = ""
        for num_line in range(len(lyrics)):

            res_sentence, mapping,total_and_incorrect = get_result(lyrics[num_line],constraints[num_line],title,history_info)
            if res_sentence is not None:
                save_res.append({"original":lyrics[num_line],"constraint":constraints[num_line],"result_sentence":res_sentence,"mapping":mapping,"total_and_incorrect":total_and_incorrect,"title":title})
            else:
                i = 0
                rephrased = get_rephrased_sentence(lyrics[num_line],constraints[num_line])
                
                while res_sentence is None and i < iteration:
                    if rephrased == None:
                        print(f'rephrased is None!!')
                        save_res.append({"original":lyrics[num_line],"constraint":constraints[num_line],"result_sentence":None,"mapping":None,"total_and_incorrect":None,"title":title})
                        break

                    print(f'rephrased sentence {rephrased}')
                    res_sentence, mapping,total_and_incorrect = get_result(rephrased,constraints[num_line],title,history_info)
                    i += 1
                    if res_sentence is None:
                        rephrased = get_rephrased_sentence(rephrased,constraints[num_line])
                if res_sentence is not None:
                    save_res.append({"original":lyrics[num_line],"constraint":constraints[num_line],"result_sentence":res_sentence,"mapping":mapping,"total_and_incorrect":total_and_incorrect,"title":title})
            if res_sentence is not None:
                history_info += re.sub(r'[!.,;:?"]+\s*$', '', res_sentence) + ", "
            out_f = out_f
            with open(out_f,'w') as f:
                json.dump(save_res,f,indent = 4)
out_f_rephrasing = "/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/result_4_3_rephrasing.json"
# out_f_no_rephrasing = "/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/result_4_3_no_rephrasing.json"

generate_songs(out_f_rephrasing)
