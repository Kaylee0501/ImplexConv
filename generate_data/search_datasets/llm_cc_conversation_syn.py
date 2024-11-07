from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import torch
import pickle
import os
import json
import random
from tqdm import tqdm


def get_top_conversation_list(
    conv_list, model, conversation_embeddings, conversations, conversation_indices, ds
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
        top_score, _, _, top_index, top_summary_ind = sorted_conversations[0]
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
        top_conversation.append(tmp)
        top_conversation_score.append(top_score)
    return top_conversation, top_conversation_score


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


def generate_conversation_dataset_cc(
    model_name="dunzhang/stella_en_400M_v5",
    input_file="full_synthetic_conversation.json",
    repo_name="jihyoung",
    dataset_name="ConversationChronicles",
    output_file="full_synthetic_conversation_cc.json",
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
    persona_conversation, persona_conversation_score = get_top_conversation_list(
        conv_list=personas,
        model=model,
        conversation_embeddings=conversation_embeddings,
        conversations=conversations,
        conversation_indices=conversation_indices,
        ds=ds,
    )
    print("getting reason conv")
    reason_conversation, reason_conversation_score = get_top_conversation_list(
        conv_list=reasons,
        model=model,
        conversation_embeddings=conversation_embeddings,
        conversations=conversations,
        conversation_indices=conversation_indices,
        ds=ds,
    )
    # Use list comprehension to get all conversations and scores in one pass
    print("getting noisy conv")
    results = [
        get_top_conversation_list(
            conv_list=noisy,
            model=model,
            conversation_embeddings=conversation_embeddings,
            conversations=conversations,
            conversation_indices=conversation_indices,
            ds=ds,
        )
        for noisy in tqdm(noisy_scenarios)
    ]

    # Unpack the results into separate lists for conversations and scores
    noisy_scenarios_conv, noisy_scenarios_score = zip(*results)

    # Build JSON structure
    i = 0
    for pc, rc, pc_score, rc_score, noisy_cov, noisy_cov_score in zip(
        persona_conversation,
        reason_conversation,
        persona_conversation_score,
        reason_conversation_score,
        noisy_scenarios_conv,
        noisy_scenarios_score,
    ):
        json_dict = {
            "cc_trait_conv": pc,
            "cc_reasoning_conv": rc,
            "cc_noisy_conv": noisy_cov,
            "cc_trait_conv_score": pc_score,
            "cc_reaosning_conv_score": rc_score,
            "cc_noisy_conv_score": noisy_cov_score,
        }
        data[i].update(json_dict)
        i += 1

    # Save JSON data
    with open(output_file, "a") as outfile:
        json.dump(data, outfile)

    print(f"Data saved successfully to {output_file}")


# Example usage
generate_conversation_dataset_cc()
