import json
import time
import os
import fire
#from openai import OpenAI
import openai
os.environ['SAMBANOVA_API_KEY'] = 'KEY'



def conversation(scenario):
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1",
    )
    prompt = f'''
    There are two speakers. Speaker 1 encounters the scenario that "{scenario}". Speaker 2 is the AI assistant.
    Based on the information. Can you generate a conversation with at least 10 turns?
    Speaker 1 shouldn't mention the scenario too early. It must be mentioned in the later section.
    Speaker 1 is exactly the person who encounters the scenario.
    The beginning turns should serve as a warm-up to introduce the scenario in a natural way.
    The conversation should be centered around the scenario without any irrelevant or extra information that is not related to the scenario.
    For Spearker 1, please do not start the conversation by saying something similar to "I'm feeling a bit overwhelmed lately." or use the same format as this sentence.
    Include diverse styles like detailed explanations, step-by-step guidance, casual small talk, humor, storytelling, and problem-solving. 
    The conversation should feel realistic and flow naturally. 
    Aim for a balance of formality and informality, capturing nuanced exchanges that go beyond simple responses.
    Please output the conversation in the following format:
    Speaker1: ...
    Assistant: ...
    
    Speaker1: ...
    Assistant: ...
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
    return str(completion.choices[0].message.content)


# def call_gpt(Scenario):
#     client = OpenAI()
#     content = f'''
#     There are two speakers. Speaker1 encounter all the {Scenario}. Speaker 2 is the AI assistant.
#     Based on the five Scenario {Scenario}. Can you generate a multi-session conversation with at least 10 turns for each scenario, which mean one scenario have one independent session.
#     '''
#     completion = client.chat.completions.create(
#         # model="gpt-4-turbo-preview",
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are an expert in conversation generation with specialized scenario."},
#             {"role": "user", "content": content }
#         ]
#     )
#     return str(completion.choices[0].message.content)



def main(
        home_dir = './datasets/impl_reasoning',
        input_file = 'sel_implic_reason1.json',
        output_file = 'final_conversation1.jsonl'
    ):
    with open(f'{home_dir}/{input_file}', 'r') as file:
        data = json.load(file)

    for line in data:
        persona = line['persona']
        impli_reas = line['reason_extreme']

        print(f'persona: {persona}')
        print(f'reasons: {impli_reas}')

        while True:
            try:
                trait_conv = conversation(persona)
                break
            except Exception as e:
                if e == KeyboardInterrupt:
                    raise e
                else:
                    pass

        for reason in impli_reas:
            while True:
                try:
                    reason_conv = conversation(reason)
                    break
                except Exception as e:
                    if e == KeyboardInterrupt:
                        raise e
                    else:
                        pass

            result = {
                'persona' : persona,
                'reason': reason,
                'conversation_trait': trait_conv,
                'conversation_reasoning': reason_conv,
            }

            with open(f'{home_dir}/{output_file}', "a") as outfile:
                json.dump(result, outfile)
                outfile.write("\n")



if __name__ == "__main__":
    fire.Fire(main)