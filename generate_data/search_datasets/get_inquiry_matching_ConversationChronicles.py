from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import torch
import pickle
import os
import csv
from tqdm import tqdm

# Load pre-trained model
model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)


inquiries = [
    "What can I do to unwind and clear my mind in a peaceful environment?",
    "What can I do that brings me a sense of fulfillment and energy?",
    "Looking for a party entertainer who can tone down their act for a nervous guest.",
    "What's been preventing you from watching your favorite TV shows lately?",
    "How can I continue learning my favorite language without extra cost or classes?",
]

# Dataset setup
repo_name = "jihyoung"
dataset_name = "ConversationChronicles"
ds = load_dataset(f"{repo_name}/{dataset_name}")

# File to store embeddings and conversations
embedding_file = f"{dataset_name}_conversation_embeddings.pkl"

# Load existing embeddings if available
if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
        conversations = data["conversations"]
        conversation_embeddings = data["conversation_embeddings"]
        conversation_indices = data["conversation_indices"]
else:
    # Process dataset to compute and store embeddings
    conversations = []
    conversation_embeddings = []
    conversation_indices = []
    dataset_size = 0

    for i in tqdm(range(len(ds["train"]))):
        # Filter multi-turn conversations
        if len(ds["train"][i]["summary"]) > 1:
            conversation = ds["train"][i]["summary"]
            conversations.append(conversation)

            # Encode the conversation
            conv_embedding = model.encode(
                conversation, convert_to_tensor=True, device="cuda"
            )
            conversation_embeddings.append(conv_embedding)
            conversation_indices.append(i)
            dataset_size += 1

            if dataset_size >= 1000:  # Limit dataset processing
                break

    # Save embeddings and conversations to a file
    with open(embedding_file, "wb") as f:
        pickle.dump(
            {
                "conversations": conversations,
                "conversation_embeddings": conversation_embeddings,
                "conversation_indices": conversation_indices,
            },
            f,
        )

# Initialize list to store similarity scores
all_similarity_scores = []

for inquiry in inquiries:
    inquiry_embedding = model.encode(inquiry, convert_to_tensor=True, device="cuda")

    for conversation_emb, conversation, index in tqdm(
        zip(conversation_embeddings, conversations, conversation_indices)
    ):
        for summary_ind, emb in enumerate(conversation_emb):
            similarity = util.cos_sim(inquiry_embedding, emb).item()

            # Store similarity scores and conversation details
            all_similarity_scores.append(
                (similarity, inquiry, conversation[summary_ind], index, summary_ind)
            )

# Sort all conversations by highest similarity score across all hobbies
sorted_conversations = sorted(all_similarity_scores, key=lambda x: x[0], reverse=True)

# Prepare data for CSV saving
csv_data = []
for score, inquiry, conversation, index, summary_ind in sorted_conversations:
    if score > 0.45:
        row_data = {
            "Average Similarity Score": score,
            "Inquiry": inquiry,
            "Summary": conversation,
            "Index": index,
        }

        # Append session dialogues based on summary index
        if summary_ind == 1:
            row_data["Session"] = ds["train"][index].get("first_session_dialogue", "")
        elif summary_ind == 2:
            row_data["Session"] = ds["train"][index].get("second_session_dialogue", "")
        elif summary_ind == 3:
            row_data["Session"] = ds["train"][index].get("third_session_dialogue", "")
        elif summary_ind == 4:
            row_data["Session"] = ds["train"][index].get("fourth_session_dialogue", "")
        elif summary_ind == 5:
            row_data["Session"] = ds["train"][index].get("fifth_session_dialogue", "")

        csv_data.append(row_data)

# Save results to CSV
csv_file_path = "five_inquries_cc.csv"
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Average Similarity Score",
            "Inquiry",
            "Summary",
            "Index",
            "Session",
        ],
    )
    writer.writeheader()
    writer.writerows(csv_data)

print("CSV saved successfully.")
