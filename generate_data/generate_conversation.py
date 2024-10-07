import json
import time
import os
from openai import OpenAI
#os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

# Open and load the JSON file
with open('implic_rea_enjoy.jsonl', 'r') as file:
    data = [json.loads(line) for line in file][:200]



def call_gpt(Scenario):
    client = OpenAI()
    content = f'''
    There are two speakers. Speaker1 encounter all the {Scenario}. Speaker 2 is the AI assistant.
    Based on the five Scenario {Scenario}. Can you generate a multi-session conversation with at least 10 turns for each scenario, which mean one scenario have one independent session.
    '''
    completion = client.chat.completions.create(
        # model="gpt-4-turbo-preview",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in conversation generation with specialized scenario."},
            {"role": "user", "content": content }
        ]
    )
    return str(completion.choices[0].message.content)



with open('final_conversation.jsonl', "a") as outfile:
    for result in data:
        scen = result['scenario']
        result = {
            'conversation': call_gpt(scen)
        }
        json.dump(result, outfile)
        outfile.write("\n")