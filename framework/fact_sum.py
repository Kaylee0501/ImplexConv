import json
import time
import os
import fire
#from openai import OpenAI
import openai
from data_process import dataset_process
from utils import sum_fact

os.environ['OPENAI_API_KEY'] = 'API-KEY'
os.environ['SAMBANOVA_API_KEY'] = 'API-KEY'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(
        home_dir = './datasets',
        dataset_name = 'locomo',
        model_type = 'Meta-Llama-3.1-405B-Instruct',
        output_file = 'summarized_facts.json',
    ):
    # if file exist,  load it
    conversations, qa_pairs = dataset_process(home_dir, dataset_name)
    if os.path.exists(f'{home_dir}/{dataset_name}/{output_file}'):
        with open(f'{home_dir}/{dataset_name}/{output_file}', 'r') as f:
            conversation_list = json.load(f)
        conversations = conversations[len(conversation_list):]        
    else:
        conversation_list = []
    for i, conversation in enumerate(conversations):
        dialog_list = []
        for j, dialog in enumerate(conversation):
            print(dialog)
            dialog_dic = {}
            print(f'Processing conversation {i}, dialog {j}')
            while True:
                try:
                    response, speaker1, speaker2 = sum_fact(dialog, model_type)
                    dialog_dic['summary'] = response.split('<<Summary:>>')[1].split(f'<<{speaker1}:>>')[0].strip()
                    dialog_dic[f'{speaker1}_fact'] = response.split(f'<<{speaker1}:>>')[1].split(f'<<{speaker2}:>>')[0].strip()
                    dialog_dic[f'{speaker2}_fact'] = response.split(f'<<{speaker2}:>>')[1].strip()
                    break
                except Exception as e:
                    if e == KeyboardInterrupt:
                        raise e
                    else:
                        pass            
            dialog_list.append(dialog_dic)
        conversation_list.append(dialog_list)
    
        with open(f'{home_dir}/{dataset_name}/{output_file}', 'w') as f:
            json.dump(conversation_list, f, indent=4)





if __name__ == "__main__":
    fire.Fire(main)