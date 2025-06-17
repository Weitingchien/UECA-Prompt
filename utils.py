# utils.py (最終版)

import torch
from transformers import BertTokenizer

# 用於解碼並印出單一批次資料的詳細資訊
def print_data_info(batch_data: tuple, tokenizer: BertTokenizer):
    """
    解碼並格式化印出一個批次中第一個樣本的詳細資訊，用於除錯。
    此版本會解碼 x_bert 和 y_bert 的文本內容，並移除 [PAD] 標記。

    Args:
        batch_data (tuple): 從 DataLoader 取出的單一批次資料。
        tokenizer (BertTokenizer): 用於解碼 Token ID 的分詞器。
    """
    x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = batch_data
    
    print("\n" + "="*20 + " BATCH DATA INFO (First Sample) " + "="*20)
    
    # 印出張量的形狀
    print(f"\n[Shape Info]")
    print(f"x_bert shape:      {x_bert.shape}")
    print(f"y_bert shape:      {y_bert.shape}")
    print(f"label shape:       {label.shape}")
    print(f"mask_label shape:  {mask_label.shape}")
    print(f"gt_emotion shape:  {gt_emotion.shape}")

    # 取出批次中的第一筆資料來觀察
    first_x = x_bert[0]
    first_y = y_bert[0]
    first_label = label[0]
    first_mask_label = mask_label[0]

    print("\n[Content Info - Decoded from Token IDs]")
    
    # 1. 模型輸入 (x_bert) - 解碼文本並移除 [PAD]
    # 我們保留 [CLS], [SEP], [MASK] 等重要特殊符號，只清除 [PAD]
    decoded_x = tokenizer.decode(first_x, skip_special_tokens=False)
    cleaned_x = decoded_x.replace('[PAD]', '').strip()
    print(f"\n--- x_bert (模型看到的輸入，已移除 PAD) ---\n{cleaned_x}\n")

    # 2. 完整答案 (y_bert) - 解碼文本並移除 [PAD]
    decoded_y = tokenizer.decode(first_y, skip_special_tokens=False)
    cleaned_y = decoded_y.replace('[PAD]', '').strip()
    print(f"--- y_bert (完整的參考答案，已移除 PAD) ---\n{cleaned_y}\n")
    
    # 3. 評估標籤 (label) - 維持原樣，因為資訊清晰
    label_tokens = first_label[first_label != -100]
    decoded_label = tokenizer.decode(label_tokens)
    print(f"--- label (評估時的目標答案) ---\nIDs: {label_tokens.tolist()}\nDecoded: '{decoded_label}'\n")

    # 4. 訓練標籤 (mask_label) - 維持原樣
    mask_label_tokens = first_mask_label[first_mask_label != -100]
    decoded_mask_label = tokenizer.decode(mask_label_tokens)
    print(f"--- mask_label (訓練時的目標答案，用於序列學習) ---\nIDs: {mask_label_tokens.tolist()}\nDecoded: '{decoded_mask_label}'\n")

    # 5. 真實數量
    print(f"--- Ground Truth Counts ---")
    print(f"gt_emotion: {gt_emotion[0].item()}, gt_cause: {gt_cause[0].item()}, gt_pair: {gt_pair[0].item()}")

    print("="*66 + "\n")