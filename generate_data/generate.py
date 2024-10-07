import pandas as pd
import json
import time
import os
import numpy as np
import tqdm
import random
from openai import OpenAI
#os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

import json

# Open and load the JSON file
with open('hobbies.json', 'r') as file:
    data = json.load(file)

# Access or print the loaded data

love_persona = []
love_traits = []
love_persona.append('This person loves sports.')
love_traits.append('sports')
for persona in data:
    if type(persona) == str and persona.find('loves') != -1:
        love_persona.append(persona)
        love_traits.append(persona[(persona.find('loves') + len('loves') + 1):])

random_persona = []
for persona in data:
    if type(persona) == str and persona.find('person is') != -1:
        random_persona.append(persona)

def implic_reason(per_info, traits_info):
    client = OpenAI()
    prompt_1 = f'''
    {per_info} However, they cannot doing it recently. Can you give me at least 10 implicit specific reasons why that person cannot do it recently.
    The reasons should be completely different.
    Please only tell the reasoning in one sentence without include anything related to "{traits_info}. Please only output the reasons with the format:
    1: 
    2: 
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in implicit reason generation."},
            {"role": "user", "content": prompt_1 }
        ]
    )
    reason_result = str(completion.choices[0].message.content)

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
    client = OpenAI()
    prompt = f'''
    {per_info} However, They cannot doing that activity recently because {reason_info} If a person asked a question. You need to give answer other than {trait_info}. 
    {reason_info} should not be include in your question.
    Please output the inquiry with the format with less than 20 words: 
    User:
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in generate general personal question."},
            {"role": "user", "content": prompt }
        ]
    )
    response = str(completion.choices[0].message.content)
    output = response.split('User:')[1].strip()

    return output

def scenario_case(per_info, reason_info, trait_info, random_info):
    client = OpenAI()
    prompt = f'''
    There are two scenarios: in the first, {per_info} and in the second, {reason_info}
    Can you generate three additional scenarios that build on these without involving {trait_info}, and ensure a smooth multi-session conversation flow based on them? 
    You can only mention that user has resolved the issue of {reason_info} in the last scenarios depend on the flow of the conversation. It's not a requirment
    The third Scenario can be related to {random_info} if related.
    The primary focus should be on making each scenario feel natural and connected.
    Please only output the Scenario without any explaiation with the format:
    Scenario 1: 
    Scenario 2:
    Scenario 3:  
    Scenario 4: 
    Scenario 5: 
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in generate different scenarios with instructions."},
            {"role": "user", "content": prompt }
        ]
    )
    return str(completion.choices[0].message.content)

with open('implic_rea_love.jsonl', "a") as outfile:
    for i in range(len(love_persona)):
        while True:
            try:
                impli_rea = implic_reason(love_persona[i], love_traits[i])
                break
            except:
                pass
        for reason in impli_rea:
            random_number = random.randint(1, len(random_persona))
            random_info = random_persona[random_number]
            while True:
                try:
                    result = {
                        'persona' : love_persona[i],
                        'reason': reason,
                        'inquiry': user_inq(love_persona[i], reason, love_traits[i]),
                        'scenario': scenario_case(love_persona[i], reason, love_traits[i], random_info)
                    }
                    break
                except:
                    pass
            print(random_info)
            json.dump(result, outfile)
            outfile.write("\n")