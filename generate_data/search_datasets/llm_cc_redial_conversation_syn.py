from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import torch
import pickle
import os
import json
import random
from tqdm import tqdm
from LLM_Redial_New.Tools import (
    read_dialogue,
    read_jsonl,
    read_json,
    get_conversation_by_id,
)


def get_top_conversation_list(
    conv_list,
    model,
    conversation_embeddings,
    conversations,
    conversation_indices,
    ds,
    inquiries_embeddings=None,
    k=5,
):
    print("getting top")
    top_conversation = []
    top_conversation_score = []
    for conv in tqdm(conv_list):
        conv_embedding = model.encode(conv, convert_to_tensor=True, device="cuda")
        top_similarity_scores = []
        for conversation_emb, conversation, index in zip(
            conversation_embeddings, conversations, conversation_indices
        ):
            for summary_ind, emb in enumerate(conversation_emb):
                similarity = util.cos_sim(conv_embedding, emb).item()
                top_similarity_scores.append(
                    (
                        similarity,
                        conv_embedding,
                        conversation[summary_ind],
                        index,
                        summary_ind,
                    )
                )

        sorted_conversations = sorted(
            top_similarity_scores, key=lambda x: x[0], reverse=True
        )
        top_k = []
        top_k_score = []
        for i in range(k):
            top_score, _, _, top_index, top_summary_ind = sorted_conversations[i]
            dialogue_list = get_top_session(ds, top_index, top_summary_ind)

            # Create a new list to hold formatted strings
            formatted_dialogues = []

            # Iterate over each item in the dialogue list with index
            for i, line in enumerate(dialogue_list):
                if i % 2 == 0:  # Even index, so it's "Speaker1"
                    formatted_dialogues.append(f"Speaker1: {line}")
                else:  # Odd index, so it's "Assistant"
                    formatted_dialogues.append(f"Assistant: {line}\n")

            # Join all formatted strings with an extra newline between entries
            tmp = "\n".join(formatted_dialogues)
            top_k.append(tmp)
            top_k_score.append(top_score)
        top_conversation.append(top_k)
        top_conversation_score.append(top_k_score)
    inquiry_similarities = []
    for conv, conv_score, inquiry_emb in tqdm(
        zip(top_conversation, top_conversation_score, inquiries_embeddings)
    ):
        conv_embedding = model.encode(conv, convert_to_tensor=True, device="cuda")
        inquiry_similarities_per_conv = []
        for conv_emb, score_emb in zip(conv_embedding, conv_score):
            similarity = util.cos_sim(inquiry_emb, conv_emb).item()
            inquiry_similarities_per_conv.append(similarity < score_emb)
        inquiry_similarities.append(inquiry_similarities_per_conv)
    return top_conversation, top_conversation_score, inquiry_similarities


# Helper function to find top session dialogue
def get_top_session(ds, index, summary_ind):
    session_key_map = {
        0: "first_session_dialogue",
        1: "second_session_dialogue",
        2: "third_session_dialogue",
        3: "fourth_session_dialogue",
        4: "fifth_session_dialogue",
    }
    return ds["train"][index].get(session_key_map.get(summary_ind, ""), "")


def get_top_conversation_list_llm_redial(
    conv_list,
    model,
    conversation_embeddings,
    conversations,
    conversation_indices,
    inquiries_embeddings=None,
    k=5,
):
    # print("Retrieving top conversations...")
    top_conversation = []
    top_conversation_score = []

    for conv in tqdm(conv_list):
        conv_embedding = model.encode(conv, convert_to_tensor=True, device="cuda")
        top_similarity_scores = []

        for conversation_emb, conversation, index in zip(
            conversation_embeddings, conversations, conversation_indices
        ):
            similarity = util.cos_sim(conv_embedding, conversation_emb).item()
            top_similarity_scores.append(
                (
                    similarity,
                    conversation,
                    index,
                )
            )

        # Sort and get the highest similarity score
        sorted_conversations = sorted(
            top_similarity_scores, key=lambda x: x[0], reverse=True
        )
        top_k = []
        top_k_score = []
        for i in range(k):
            top_score, top_dialogue, top_index = sorted_conversations[i]

            # Format the conversation
            formatted_text = top_dialogue.replace("User", "Speaker1").replace(
                "Agent", "Assistant"
            )
            formatted_dialogues = formatted_text.split("\n")
            # Join all formatted strings with an extra newline between entries
            tmp = "\n".join(formatted_dialogues)
            top_k.append(tmp)
            top_k_score.append(top_score)
        top_conversation.append(top_k)
        top_conversation_score.append(top_k_score)
    inquiry_similarities = []
    for conv, conv_score, inquiry_emb in tqdm(
        zip(top_conversation, top_conversation_score, inquiries_embeddings)
    ):
        conv_embedding = model.encode(conv, convert_to_tensor=True, device="cuda")
        inquiry_similarities_per_conv = []
        for conv_emb, score_emb in zip(conv_embedding, conv_score):
            similarity = util.cos_sim(inquiry_emb, conv_emb).item()
            inquiry_similarities_per_conv.append(similarity < score_emb)
        inquiry_similarities.append(inquiry_similarities_per_conv)
    return top_conversation, top_conversation_score, inquiry_similarities


def generate_conversation_dataset_cc(
    model_name="dunzhang/stella_en_400M_v5",
    input_file="full_synthetic_conversation_1.json",
    repo_name="jihyoung",
    dataset_name="ConversationChronicles",
    output_file="full_synthetic_conversation_cc_1.json",
    embedding_file="ConversationChronicles_conversation_embeddings.pkl",
    dataset_limit=1000,
):
    # Load pre-trained model
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Load input JSONL file
    data = []

    with open(input_file, "r") as file:
        data = json.load(file)  # This reads the entire JSON array at once
    # print(len(data))
    # print(data[0])

    # Collect random sample of personas and reasons
    # sampled_data = random.sample(data, 2)
    sampled_data = data
    personas = [entry["persona"] for entry in sampled_data]
    reasons = [entry["reason"] for entry in sampled_data]
    noisy_scenarios = [entry["noisy_scenarios"] for entry in sampled_data]

    # Load dataset
    ds = load_dataset(f"{repo_name}/{dataset_name}")

    # Load or create embeddings
    if os.path.exists(embedding_file):
        with open(embedding_file, "rb") as f:
            saved_data = pickle.load(f)
            conversations = saved_data["conversations"]
            conversation_embeddings = saved_data["conversation_embeddings"]
            conversation_indices = saved_data["conversation_indices"]
    else:
        conversations = []
        conversation_embeddings = []
        conversation_indices = []

        for i in tqdm(range(len(ds["train"]))):
            if len(ds["train"][i]["summary"]) > 1:
                conversation = ds["train"][i]["summary"]
                conversations.append(conversation)

                # Encode conversation
                conv_embedding = model.encode(
                    conversation, convert_to_tensor=True, device="cuda"
                )
                conversation_embeddings.append(conv_embedding)
                conversation_indices.append(i)

                if len(conversations) >= dataset_limit:
                    break

        # Save embeddings
        with open(embedding_file, "wb") as f:
            pickle.dump(
                {
                    "conversations": conversations,
                    "conversation_embeddings": conversation_embeddings,
                    "conversation_indices": conversation_indices,
                },
                f,
            )

    # Compute similarity for personas
    print("getting persona conv")
    inquiries = [entry["question"] for entry in sampled_data]
    inquiries_embeddings = model.encode(
        inquiries, convert_to_tensor=True, device="cuda"
    )
    persona_conversation, persona_conversation_score, inquiry_similarities = (
        get_top_conversation_list(
            conv_list=personas,
            model=model,
            conversation_embeddings=conversation_embeddings,
            conversations=conversations,
            conversation_indices=conversation_indices,
            ds=ds,
            inquiries_embeddings=inquiries_embeddings,
        )
    )
    # print("getting reason conv")
    # reason_conversation, reason_conversation_score = get_top_conversation_list(
    #     conv_list=reasons,
    #     model=model,
    #     conversation_embeddings=conversation_embeddings,
    #     conversations=conversations,
    #     conversation_indices=conversation_indices,
    #     ds=ds,
    # )
    # # Use list comprehension to get all conversations and scores in one pass
    # print("getting noisy conv")
    # results = [
    #     get_top_conversation_list(
    #         conv_list=noisy,
    #         model=model,
    #         conversation_embeddings=conversation_embeddings,
    #         conversations=conversations,
    #         conversation_indices=conversation_indices,
    #         ds=ds,
    #     )
    #     for noisy in tqdm(noisy_scenarios)
    # ]

    # # Unpack the results into separate lists for conversations and scores
    # noisy_scenarios_conv, noisy_scenarios_score = zip(*results)

    # Build JSON structure
    i = 0
    for pc, pc_score, inquiry_score in zip(
        persona_conversation, persona_conversation_score, inquiry_similarities
    ):

        json_dict = {
            "cc_noisy_conv": pc,
            "cc_noisy_conv_score": pc_score,
            "cc_inquiry_bool": inquiry_score,
        }
        sampled_data[i].update(json_dict)
        i += 1

    # Save JSON data
    with open(output_file, "w") as outfile:
        json.dump(sampled_data, outfile)
    print(f"Data saved successfully to {output_file}")


def generate_conversation_dataset_llm_redial(
    model_name="dunzhang/stella_en_400M_v5",
    input_file="full_synthetic_conversation_cc_1.json",
    output_file="full_synthetic_conversation_cc_redial_1.json",
    embedding_dir="LLM_Redial_conversation_embeddings",
    dataset_limit=3000,
):
    # Load pre-trained model
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Load input JSON file
    with open(input_file, "r") as file:
        data = json.load(file)
    # data = random.sample(data, 2)

    # Extract persona, reason, and noisy scenarios from input data
    personas = [entry["persona"] for entry in data]
    reasons = [entry["reason"] for entry in data]
    noisy_scenarios = [entry["noisy_scenarios"] for entry in data]
    inquiries = [entry["question"] for entry in data]
    inquiries_embeddings = model.encode(
        inquiries, convert_to_tensor=True, device="cuda"
    )
    # Loop through each item category
    items = ["Movie", "Books", "Electronics", "Sports"]
    for item in items:
        path = f"./LLM_Redial_New/LLM_Redial/{item}"
        final_data_path = f"{path}/final_data.jsonl"
        Conversation_path = f"{path}/Conversation.txt"
        item_map_path = f"{path}/item_map.json"

        # Load data from paths
        final_data = read_jsonl(final_data_path)
        item_map = read_json(item_map_path)
        Conversation = read_dialogue(Conversation_path)

        # Define item-specific embedding file
        embedding_file = f"{embedding_dir}_{item}.pkl"

        # Load or create conversation embeddings for this item
        if os.path.exists(embedding_file):
            print("skipped", item)
            continue
            # with open(embedding_file, "rb") as f:
            #     saved_data = pickle.load(f)
            #     conversations = saved_data["conversations"]
            #     conversation_embeddings = saved_data["conversation_embeddings"]
            #     conversation_indices = saved_data["conversation_indices"]
        else:
            conversations, conversation_embeddings, conversation_indices = [], [], []

            for i in tqdm(
                range(len(final_data)), desc=f"Encoding conversations for {item}"
            ):
                Per_data = json.loads(final_data[i])
                user_id, user_information = next(iter(Per_data.items()))
                Conversation_info = user_information["Conversation"]

                for j, per_conversation_info in enumerate(Conversation_info):
                    conversation_id = per_conversation_info[
                        "conversation_{}".format(j + 1)
                    ]["conversation_id"]
                    dialogue = get_conversation_by_id(Conversation, conversation_id)
                    conversations.append(dialogue)

                    # Encode conversation and store embedding
                    conv_embedding = model.encode(
                        dialogue, convert_to_tensor=True, device="cuda"
                    )
                    conversation_embeddings.append(conv_embedding)
                    conversation_indices.append(i)

                    if len(conversations) >= dataset_limit:
                        break

            # Save embeddings to a file
            with open(embedding_file, "wb") as f:
                pickle.dump(
                    {
                        "conversations": conversations,
                        "conversation_embeddings": conversation_embeddings,
                        "conversation_indices": conversation_indices,
                    },
                    f,
                )

    conversations, conversation_embeddings, conversation_indices = [], [], []

    for item in items:
        embedding_file = f"{embedding_dir}_{item}.pkl"
        if os.path.exists(embedding_file):
            with open(embedding_file, "rb") as f:
                saved_data = pickle.load(f)
                conversations.extend(saved_data["conversations"])
                conversation_embeddings.extend(saved_data["conversation_embeddings"])
                conversation_indices.extend(saved_data["conversation_indices"])

    # Retrieve top conversations for personas, reasons, and noisy scenarios
    print("Retrieving persona conversations...")
    persona_conversation, persona_conversation_score, inquiries_similarities = (
        get_top_conversation_list_llm_redial(
            conv_list=personas,
            model=model,
            conversation_embeddings=conversation_embeddings,
            conversations=conversations,
            conversation_indices=conversation_indices,
            inquiries_embeddings=inquiries_embeddings,
        )
    )

    # print("Retrieving reason conversations...")
    # reason_conversation, reason_conversation_score = (
    #     get_top_conversation_list_llm_redial(
    #         conv_list=reasons,
    #         model=model,
    #         conversation_embeddings=conversation_embeddings,
    #         conversations=conversations,
    #         conversation_indices=conversation_indices,
    #     )
    # )

    # print("Retrieving noisy conversations...")
    # noisy_results = [
    #     get_top_conversation_list_llm_redial(
    #         conv_list=noisy,
    #         model=model,
    #         conversation_embeddings=conversation_embeddings,
    #         conversations=conversations,
    #         conversation_indices=conversation_indices,
    #     )
    #     for noisy in tqdm(noisy_scenarios, desc="Processing noisy scenarios")
    # ]
    # noisy_scenarios_conv, noisy_scenarios_score = zip(*noisy_results)

    # Append retrieved data to each entry in the original data
    for i, (pc, pc_score, inquiry_bool) in enumerate(
        zip(persona_conversation, persona_conversation_score, inquiries_similarities)
    ):
        data[i].update(
            {
                "llmredial_trait_conv": pc,
                "llmredial_cc_trait_conv_score": pc_score,
                "llmredial_inquiry_bool": inquiry_bool,
            }
        )

    # Save the updated data to a new output JSON file
    with open(output_file, "w") as outfile:
        json.dump(data, outfile)

    print(f"Data successfully saved to {output_file}")


# Example usage
generate_conversation_dataset_cc()
generate_conversation_dataset_llm_redial()
