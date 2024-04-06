import json
import re
import pickle
from datasets import Dataset, load_dataset


data = json.load(open("/home/songyan/Real_M2L-main/data/data_finetune/chatgpt3.5/generated_line_w_structure_neg.json", "r"))
n = len(data)
print(f'there are {n} songs')
train, validation, test = data[:int(n*0.8)], data[int(n*0.8):int(n*0.9)], data[int(n*0.9):]

DEFAULT_SYSTEM_PROMPT = """
Below is the lyric of a song and the constraint. Rewrite the lyric to satisfy the constraint.
""".strip()

def generate_training_prompt(input, output, system_prompt = DEFAULT_SYSTEM_PROMPT):
   return f"""
          ### Instruction: {system_prompt}
   
          ### Input:
          {input}

          ### Response:
          {output}
          """.strip()


def process_data(data):
    dataset = []
    for d in data:
        input = d["Input"]
        output = d["Output"]
        prompt = generate_training_prompt(input, output)
        dataset.append({"prompt": prompt, "input": input, "output": output})
    return dataset

train_dataset = process_data(train)
validation_dataset = process_data(validation)
test_dataset = process_data(test)

train_dataset = Dataset.from_list(train_dataset)
validation_dataset = Dataset.from_list(validation_dataset)
test_dataset = Dataset.from_list(test_dataset)

dataset = {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}

with open('/home/songyan/Real_M2L-main/data/data_finetune/chatgpt3.5/dataset_w_struct_neg.pkl', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

