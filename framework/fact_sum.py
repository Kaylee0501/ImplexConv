import json
import time
import os
import fire
#from openai import OpenAI
import openai
from tqdm import tqdm
from data_process import dataset_process
from utils import get_summary, get_all_facts, sum_fact_reasoning

os.environ['OPENAI_API_KEY'] = ' REMOVED'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def main(
        home_dir = './datasets',
        data_dir = 'impConv',
        dataset_name = 'locomo',
        model_type = 'gpt-4o-mini',
        output_file = 'summarized_facts_new.json',
    ):
    # if file exist,  load it
    conversations, qa_pairs = dataset_process(home_dir, dataset_name)
    if os.path.exists(f'{home_dir}/{data_dir}/{output_file}'):
        with open(f'{home_dir}/{data_dir}/{output_file}', 'r') as f:
            conversation_list = json.load(f)
        conversations = conversations[len(conversation_list):]        
    else:
        conversation_list = []
    print(f'conversation length: {len(conversation_list)}')
    for i, conversation in tqdm(enumerate(conversations)):
        dialog_list = []
        for j, dialog in enumerate(conversation):
            dialog_dic = {}
            print(f'Processing conversation {i}, dialog {j}')        
            if dataset_name == 'syn_intent' or dataset_name == 'syn_reasoning':
                print('PASS')
                while True:
                    try:
                        response = sum_fact_reasoning(dialog, model_type)
                        print(response)
                        dialog_dic['summary'] = response.split('<<Summary:>>')[1].split(f'<<Speaker1:>>')[0].strip()
                        dialog_dic['Speaker1_fact'] = response.split(f'<<Speaker1:>>')[1].strip()
                        break
                    except Exception as e:
                        if e == KeyboardInterrupt:
                            raise e
                        else:
                            pass                
            else:
                while True:
                    try:
                        response, speaker1, speaker2 = get_all_facts(dialog, model_type)
                        dialog_dic[f'{speaker1}_fact'] = response.split(f'<<{speaker1}:>>')[1].split(f'<<{speaker2}:>>')[0].strip()
                        dialog_dic[f'{speaker2}_fact'] = response.split(f'<<{speaker2}:>>')[1].strip()
                        summary = get_summary(dialog, model_type)
                        dialog_dic['summary'] = summary.split('<<Summary:>>')[1].strip()
                        break
                    except Exception as e:
                        if e == KeyboardInterrupt:
                            raise e
                        else:
                            pass
            dialog_list.append(dialog_dic)
        conversation_list.append(dialog_list)
    
        with open(f'{home_dir}/{data_dir}/{output_file}', 'w') as f:
            json.dump(conversation_list, f, indent=4)





if __name__ == "__main__":
    fire.Fire(main)