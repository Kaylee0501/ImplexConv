import json
import os
import fire
import numpy as np
import tqdm
import random
import openai
import json
from sentence_transformers import SentenceTransformer
os.environ['SAMBANOVA_API_KEY'] = 'API_KEY'


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

def user_inq_2(per_info, reason_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    
    prompt = f'''
    Here's the conversation between a user(speaker 1) and a chatbot assistant.
    Speaker 1 has the following persona trait: {per_info}. However, speaker 1 cannot do the trait since {reason_info}. 
    However, speaker forgots they cannot do the trait, and ask you a question related to the trait. 
    Therfore, your answer should be different with/without the reason. With the reason, you should tell speark 1 they cannot do the trait.
    The trait should be mentioned in the question.
    The question itself should not mention the reason or affect by the reason.
    Questions should be asked in the first person. Include "I".
    The question should not be a yes/no question. Maybe speark 1 can ask assistant for suggestions or ask assistant give some idea about the trait.

    Please only output the question in the format of less than 20 words without any additional sentences.

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

def user_inq_3(per_info, trait_info, reason_info):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    
    prompt = f'''
    Here's the conversation between a user(speaker 1) and a chatbot assistant.
    Speaker 1 has the following persona trait: {per_info}. However, speaker 1 cannot do the trait due to the reason that {reason_info}. 
    Now, speaker 1 ask you a question related to the trait. {reason_info} affect your answer to this question.
    You should tell speark 1 they cannot do the trait due to the reason.
    The trait should be mentioned in the question.
    The question itself should not mention the reason or affect by the reason.
    Questions should be asked in the first person. Include "I".
    The question should not be a yes/no question. 
    The question needs to be diverse.

    Please only output the question in the format of less than 20 words without any additional sentences.

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


# def main(
#         home_dir = './datasets/impl_reasoning',
#         input_file = 'sel_implic_reason1.json',
#         output_file = 'sel_implic_question1.jsonl'
#     ):
#     with open(f'{home_dir}/{input_file}', 'r') as file:
#         data = json.load(file)

#     for line in data:
#         persona = line['persona']
#         impli_reas = line['reason_extreme']
#         traits = ' '.join(persona.split()[3:])

#         print(f'persona: {persona}')
#         print(f'reasons: {impli_reas}')
#         print(f'traits: {traits}')

#         for reason in impli_reas:
#             while True:
#                 try:
#                     inquiry1 = user_inq_1(persona, reason, traits)
#                     break
#                 except Exception as e:
#                     if e == KeyboardInterrupt:
#                         raise e
#                     else:
#                         pass

#             while True:
#                 try:
#                     inquiry2 = user_inq_2(persona, reason, traits)
#                     break
#                 except Exception as e:
#                     if e == KeyboardInterrupt:
#                         raise e
#                     else:
#                         pass

#             result = {
#                 'persona' : persona,
#                 'reason': reason,
#                 'inquiry1': inquiry1,
#                 'inquiry2': inquiry2,
#             }

#             with open(f'{home_dir}/{output_file}', "a") as outfile:
#                 json.dump(result, outfile)
#                 outfile.write("\n")

def main(home_dir = './datasets/impl_reasoning',
        input_file = 'noisy_conversation1.jsonl',
        output_file = 'new_implic_question1.jsonl'
    ):
    all_data = []
    with open(f'{home_dir}/{input_file}', 'r') as file:
        for line in file:
            all_data.append(json.loads(line))

    for item in all_data:
        persona = item['persona']
        impli_reas = item['max_reason']
        traits = ' '.join(persona.split()[3:])
        more_info = ' '.join(persona.split()[2:])

        print(f'persona: {persona}')
        print(f'reasons: {impli_reas}')
        print(f'traits: {traits}')

        while True:
            try:
                inquiry1 = user_inq_3(persona, traits, impli_reas)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass
        print(f'inquiry1: {inquiry1}')

        while True:
            try:
                inquiry2 = user_inq_2(persona, impli_reas)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass

        print(f'inquiry2: {inquiry2}')
        
        result = {
            'persona' : persona,
            'reason': impli_reas,
            'inquiry1': inquiry1,
            'inquiry2': inquiry2,
        }

        with open(f'{home_dir}/{output_file}', "a") as outfile:
            json.dump(result, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    fire.Fire(main)