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

def implic_reason(per_info, traits_info):

    prompt_1 = f'''
        {per_info} However, they have not been able to do it recently. Can you give me at least 30 implicit reasons why that person cannot do it? 
        The reasons should be completely different from each other and belong to different categories.
        The reason should be specific with detailed information, like why it happens.
        The reason cannot include words related to "{traits_info}"
        Please explain the reasoning in only one sentence. Please only output the reasons with the format:
        1: 
        2: 
    '''

    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )

    response = client.chat.completions.create(
        model='Meta-Llama-3.1-405B-Instruct',
        messages=[
                    {"role":"system","content":"You are a helpful assistant"},
                    {"role":"user","content": prompt_1}
                ],
        temperature =  1,
        top_p = 0.9
    )
    reason_result = response.choices[0].message.content
    print(reason_result)

    lines = reason_result.splitlines()

    # Initialize an empty list to store the reasons
    reasons = []

    # Loop through each line
    for line in lines:
        # Split the line at the colon and check if it has a reason part
        if ":" in line:
            reason = line.split(":")[1].strip()  # Extract the part after colon and remove any leading/trailing spaces
            reasons.append(reason)
    top_reason = reasons[1:]

    return top_reason

def user_inq(per_info, reason_info, trait_info):
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

# def scenario_case(per_info, reason_info, trait_info):
#     client = openai.OpenAI(
#         api_key=os.environ.get("SAMBANOVA_API_KEY"),
#         base_url="https://api.sambanova.ai/v1",
#     )
#     prompt = f'''
#     There are two scenarios: in the first, {per_info} and in the second, {reason_info}
#     Can you generate two additional scenarios that build on these without involving {trait_info}, and ensure a smooth multi-session conversation flow based on them? 
#     You can only mention that user has resolved the issue of {reason_info} in the last scenarios depend on the flow of the conversation. It's not a requirment
#     The primary focus should be on making each scenario feel natural and connected.
#     Please only output the Scenario without any explaiation with the format:
#     Scenario 1: 
#     Scenario 2:
#     Scenario 3:  
#     Scenario 4: 
#     '''
#     completion = client.chat.completions.create(
#     model='Meta-Llama-3.1-405B-Instruct',
#     messages=[
#                 {"role":"system","content":"You are a helpful assistant"},
#                 {"role":"user","content": prompt}
#               ],
#     temperature =  1,
#     top_p = 0.9
#     )
#     return str(completion.choices[0].message.content)


def main(
        home_dir = './datasets/impl_reasoning',
        input_file = 'hobbies.json',
        output_file = 'implic_reason1.jsonl'
    ):
    with open(f'{home_dir}/{input_file}', 'r') as file:
        data = json.load(file)
        # data = json.load(file)[10000:20000]

    persona = []
    traits = []
    for line in data:
        if len(line.split()) > 3:
            persona.append(line)
            traits.append(' '.join(line.split()[3:]))

    with open(f'{home_dir}/{output_file}', "a") as outfile:
        for i in range(len(persona)):
            while True:
                try:
                    impli_rea = implic_reason(persona[i], traits[i])
                    break
                except Exception as e:
                    if e == KeyboardInterrupt:
                        raise e
                    else:
                        pass
            result = {'persona' : persona[i],
                    'reason': impli_rea,}
            json.dump(result, outfile)
            outfile.write("\n")



if __name__ == "__main__":
    fire.Fire(main)