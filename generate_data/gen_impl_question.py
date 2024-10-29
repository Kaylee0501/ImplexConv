import json
import os
import fire
import numpy as np
import tqdm
import random
import openai
import json
from sentence_transformers import SentenceTransformer
os.environ['SAMBANOVA_API_KEY'] = 'KEY'


def user_inq_1(per_info, reason_info, trait_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    
    prompt = f'''
    {per_info} However, they cannot do that activity recently because {reason_info}. Can you come up with a high-level question with an answer other than {trait_info}? 
    The question should be vague.
    Ask the question in the first person.
    The question should not mention that something is wrong.
    The question should not mention that they want something other than their usual activity.
    The question should not include words like "recently, other, new".
    Make sure all the words in the answer and reason do not exist in the question. 
    Please output the inquiry with the format in one line without any additional sentences: 
    User:
    '''
    completion = client.chat.completions.create(
    model='Meta-Llama-3.1-405B-Instruct',
    messages=[
                {"role":"system","content":"You are a helpful assistant"},
                {"role":"user","content": prompt}
              ],
    temperature =  1,
    top_p = 0.9
    )
    response = str(completion.choices[0].message.content)
    print(response)
    # Extract the output from the response if“User” eixits, delete the "User:" and any leading/trailing spaces
    if 'User:' in response:
        response = response.split('User:')[1].strip()
    
    return response

def user_inq_2(per_info, reason_info, trait_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    
    prompt = f'''
    {per_info} However, They cannot do that activity recently because {reason_info}.
    If a person asked a question. You need to give an answer other than "{trait_info}". 
    Make sure all the words in the personal trait and reason do not exist in the question.
    Please output the inquiry in the format of less than 20 words: 
    User:
    '''

    completion = client.chat.completions.create(
    model='Meta-Llama-3.1-405B-Instruct',
    messages=[
                {"role":"system","content":"You are a helpful assistant"},
                {"role":"user","content": prompt}
              ],
    temperature =  1,
    top_p = 0.9
    )
    response = str(completion.choices[0].message.content)
    print(response)
    # Extract the output from the response if“User” eixits, delete the "User:" and any leading/trailing spaces
    if 'User:' in response:
        response = response.split('User:')[1].strip()
    
    return response


def main(
        home_dir = './datasets/impl_reasoning',
        input_file = 'sel_implic_reason1.json',
        output_file = 'sel_implic_question1.jsonl'
    ):
    with open(f'{home_dir}/{input_file}', 'r') as file:
        data = json.load(file)

    for line in data:
        persona = line['persona']
        impli_reas = line['reason_extreme']
        traits = ' '.join(persona.split()[3:])

        print(f'persona: {persona}')
        print(f'reasons: {impli_reas}')
        print(f'traits: {traits}')

        for reason in impli_reas:
            while True:
                try:
                    inquiry1 = user_inq_1(persona, reason, traits)
                    break
                except Exception as e:
                    if e == KeyboardInterrupt:
                        raise e
                    else:
                        pass

            while True:
                try:
                    inquiry2 = user_inq_2(persona, reason, traits)
                    break
                except Exception as e:
                    if e == KeyboardInterrupt:
                        raise e
                    else:
                        pass

            result = {
                'persona' : persona,
                'reason': reason,
                'inquiry1': inquiry1,
                'inquiry2': inquiry2,
            }

            with open(f'{home_dir}/{output_file}', "a") as outfile:
                json.dump(result, outfile)
                outfile.write("\n")



if __name__ == "__main__":
    fire.Fire(main)