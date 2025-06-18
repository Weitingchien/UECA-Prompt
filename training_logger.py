import json
import os
import torch
from transformers import BertTokenizer

def log_predictions(doc_id, x_bert_ids, predicted_y_bert_ids, label_ids, mask_label_ids, gt_emotion, gt_cause, gt_pair, tokenizer: BertTokenizer, output_filepath: str):
    # Ensure all tensor inputs are on the CPU and converted to lists or primitive types before serialization
    
    def decode_and_clean(token_ids_tensor, is_label=False):
        # Detach tensor from graph and move to CPU
        token_ids_tensor = token_ids_tensor.detach().cpu()

        if token_ids_tensor.ndim > 1: # If batched, take the first sample
            token_ids_tensor = token_ids_tensor[0]
        
        if is_label:
            # Filter out -100 values for labels before decoding and for token list
            token_ids_list_for_decode = token_ids_tensor[token_ids_tensor != -100].tolist()
            stored_token_ids_list = token_ids_list_for_decode
        else:
            # For inputs/predictions, decode all tokens, then clean PADs from string
            # Store all original tokens (including PADs)
            token_ids_list_for_decode = token_ids_tensor.tolist()
            stored_token_ids_list = token_ids_tensor.tolist()

        decoded_text = tokenizer.decode(token_ids_list_for_decode, skip_special_tokens=False)
        
        # Remove padding tokens from the decoded string only
        # It's safer to replace by tokenizer.pad_token_id and then decode, 
        # or replace the specific pad token string.
        cleaned_text = decoded_text.replace(tokenizer.pad_token, '').strip()
        # Replace multiple spaces with a single space for cleanliness after pad removal
        cleaned_text = ' '.join(cleaned_text.split())


        return cleaned_text, stored_token_ids_list

    x_bert_decoded, x_bert_tokens = decode_and_clean(x_bert_ids) # is_label=False by default
    predicted_y_bert_decoded, predicted_y_bert_tokens = decode_and_clean(predicted_y_bert_ids) # is_label=False by default
    
    label_decoded, label_tokens = decode_and_clean(label_ids, is_label=True)
    mask_label_decoded, mask_label_tokens = decode_and_clean(mask_label_ids, is_label=True)

    # Prepare the data structure
    log_entry = {
        "doc_id": doc_id, # doc_id is already a string
        "x_bert": {
            "Token IDs": " ".join(map(str, x_bert_tokens)),
            "Decoded": x_bert_decoded
        },
        "predicted_y_bert": { 
            "Token IDs": " ".join(map(str, predicted_y_bert_tokens)),
            "Decoded": predicted_y_bert_decoded
        },
        "label": {
            "Token IDs": " ".join(map(str, label_tokens)), # Should only contain non -100 tokens
            "Decoded": label_decoded
        },
        "mask_label": {
            "Token IDs": " ".join(map(str, mask_label_tokens)), # Should only contain non -100 tokens
            "Decoded": mask_label_decoded
        },
        "gt_emotion": gt_emotion.item() if torch.is_tensor(gt_emotion) else gt_emotion,
        "gt_cause": gt_cause.item() if torch.is_tensor(gt_cause) else gt_cause,
        "gt_pair": gt_pair.item() if torch.is_tensor(gt_pair) else gt_pair
    }

    # Read existing data or initialize if file doesn't exist/is empty
    data_list = []
    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        with open(output_filepath, 'r', encoding='utf-8') as f:
            try:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    data_list = loaded_data
            except json.JSONDecodeError:
                # File is corrupted or not valid JSON, start with an empty list
                pass # data_list is already []

    # Append new entry
    data_list.append(log_entry)

    # Write data back to JSON file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
