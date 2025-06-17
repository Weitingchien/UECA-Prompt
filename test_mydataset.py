#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('.') # Add repo root to allow importing ECPE
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import os

from ECPE import MyDataset # Import MyDataset from ECPE.py

def get_sample_data_as_dict(dataset_instance, tokenizer, sample_idx=0):
    if not (0 <= sample_idx < len(dataset_instance)):
        return {"error": f"Sample index {sample_idx} is out of bounds for dataset of length {len(dataset_instance)}"}

    sample = dataset_instance[sample_idx]
    data_dict = {}
    
    data_dict['doc_id'] = dataset_instance.doc_id[sample_idx]
    
    field_names = ['x_bert', 'y_bert', 'label', 'mask_label', 'gt_emotion', 'gt_cause', 'gt_pair']
    
    for i, field_name in enumerate(field_names):
        data = sample[i]
        if isinstance(data, np.ndarray) and data.ndim == 1 and field_name in ['x_bert', 'y_bert', 'label', 'mask_label']:
            # Convert numpy array to a single space-separated string for Token IDs
            token_ids_str = " ".join(map(str, data.tolist()))
            meaningful_tokens_for_decode = [token_id for token_id in data if token_id != 0]
            decoded_text = tokenizer.decode(meaningful_tokens_for_decode, skip_special_tokens=False)
            data_dict[field_name] = {
                "Token IDs": token_ids_str, # Changed to string
                "Decoded": decoded_text
            }
        elif isinstance(data, np.generic):
             data_dict[field_name] = data.item()
        else:
            data_dict[field_name] = data
            
    return data_dict

def run_and_save_json():
    print("Initializing bert-base-chinese tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print("Tokenizer initialized.")

    train_file_path = 'data_combine_ECPE/fold1_train.txt'
    test_file_path = 'data_combine_ECPE/fold1_test.txt'
    output_json_path = 'mydataset_output.json'

    results = {}

    print(f"Attempting to load training data from: {train_file_path}")
    if not os.path.exists(train_file_path):
        print(f"ERROR: Training file not found at {train_file_path}")
        results['training_data_error'] = f"File not found: {train_file_path}"
    else:
        train_dataset = MyDataset(train_file_path, test=False, tokenizer=tokenizer)
        print("Training dataset loaded.")
        if len(train_dataset) > 0:
            results['first_sample_train_data'] = get_sample_data_as_dict(train_dataset, tokenizer, 0)
        else:
            results['first_sample_train_data'] = {"info": "Training dataset is empty."}
            
    print(f"Attempting to load test data from: {test_file_path}")
    if not os.path.exists(test_file_path):
        print(f"ERROR: Test file not found at {test_file_path}")
        results['test_data_error'] = f"File not found: {test_file_path}"
    else:
        test_dataset = MyDataset(test_file_path, test=True, tokenizer=tokenizer)
        print("Test dataset loaded.")
        if len(test_dataset) > 0:
            results['first_sample_test_data'] = get_sample_data_as_dict(test_dataset, tokenizer, 0)
        else:
            results['first_sample_test_data'] = {"info": "Test dataset is empty."}

    print(f"Writing output to {output_json_path}...")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Successfully wrote output to {output_json_path}")
    except Exception as e:
        print(f"Error writing JSON to file: {e}")

if __name__ == '__main__':
    run_and_save_json()
