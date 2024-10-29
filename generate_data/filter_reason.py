from sentence_transformers import SentenceTransformer
import os
import fire
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def select_unique_reasons(similarity_matrix, threshold):
    # Number of reasons
    num_reasons = similarity_matrix.shape[0]
    
    # To keep track of selected reasons
    selected_reasons = []
    
    # Create a list of reasons (0 to num_reasons-1)
    remaining_reasons = list(range(num_reasons))
    
    # Loop through all reasons
    while len(remaining_reasons) >= 2:
        # Find the two reasons with the lowest similarity score
        min_score = float('inf')
        best_pair = (None, None)
        
        for i in range(len(remaining_reasons)):
            for j in range(i + 1, len(remaining_reasons)):
                reason1 = remaining_reasons[i]
                reason2 = remaining_reasons[j]
                if similarity_matrix[reason1, reason2] < min_score:
                    min_score = similarity_matrix[reason1, reason2]
                    best_pair = (reason1, reason2)
        
        # Add the best pair to the selected reasons
        selected_reasons.extend(best_pair)

        
        # Remove the reasons that are similar to either of the two selected reasons
        new_remaining_reasons = []
        for reason in remaining_reasons:
            if similarity_matrix[best_pair[0], reason] < threshold and similarity_matrix[best_pair[1], reason] < threshold:
                new_remaining_reasons.append(reason)
        
        remaining_reasons = new_remaining_reasons

    if len(remaining_reasons) == 1:
        selected_reasons.append(remaining_reasons[0])

    print(selected_reasons)
    
    return selected_reasons

def main(
        home_dir = './datasets/impl_reasoning',
        input_file = 'implic_reason1.json',
        output_file = 'sel_implic_reason1.jsonl'
    ):

    with open(f'{home_dir}/{input_file}', 'r') as file:
        data = json.load(file)

    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    with open(f'{home_dir}/{output_file}', "a") as outfile:
        for line in data:
            reasoning = line['reason']
            embeddings = model.encode(reasoning)
            similarities = model.similarity(embeddings, embeddings)
            #print(similarities)
            selected_indices_small = select_unique_reasons(similarities, 0.5)
            selected_reasons_small = [reasoning[i] for i in selected_indices_small]
            selected_indices_large = select_unique_reasons(similarities, 0.55)
            selected_reasons_large = [reasoning[i] for i in selected_indices_large]
            result = {'persona' : line['persona'],
                    'reason': selected_reasons_large,
                    'reason_extreme': selected_reasons_small}
            json.dump(result, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    fire.Fire(main)