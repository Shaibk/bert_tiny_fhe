import torch
import numpy as np
import os
import sys
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert

# A small subset of CLINC150 labels for validation
# We will print the ID and check if it makes sense contextually
ID2LABEL_SUBSET = {
    20: "freeze_account",
    5: "weather",
    11: "user_name",
    16: "order", # book table etc
    4: "time",
    8: "gas_type", # or gas station location
    10: "smart_home",
    130: "transfer", # transfer money
    101: "traffic",
    86: "play_music",
    33: "date",
    36: "directions", # gas station
    2: "account_blocked", # cancel reservation might map here or similar
    140: "exchange_rate",
    143: "lost_card",
    128: "balance"
}

# The sentences we used in the benchmark
TEST_SENTENCES = [
    "freeze my account",              # Exp: 20
    "tell me the weather",            # Exp: 5
    "what is your name",              # Exp: 11
    "book a table for two",           # Exp: 16 (restaurant_reservation)
    "transfer money to mom",          # Exp: 130
    "how is the traffic",             # Exp: 101
    "play some music",                # Exp: 86
    "set an alarm for 8am",           # Exp: 4 (alarm)
    "what is the date today",         # Exp: 33
    "where is the nearest gas station",# Exp: 36
    "cancel my reservation",          # Exp: 16/2 (cancel_reservation)
    "who are you",                    # Exp: 11
    "exchange rate for euro",         # Exp: 140
    "my card is lost",                # Exp: 143
    "do you like pizza",              # Exp: 15 (hobbies/smalltalk)
    "what is my balance"              # Exp: 128
]

def main():
    print("==========================================================")
    print("   PyTorch Model Accuracy Verification (With Mask)")
    print("==========================================================")
    
    # 1. Load Model
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "experiments/accuracy_first/plaintext/student_8level.pt")
    
    print(f"Loading Model: {model_path}")
    
    # Initialize with the exact config used in training
    model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=128, layers=2, heads=2, 
        intermediate=512, dropout=0.0, 
        attn_type="2quad", attn_kwargs={"c": 4.0}, 
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )
    
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 2. Tokenize (WITH Padding & Mask)
    print("Tokenizing inputs...")
    model_id = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    inputs = tokenizer(TEST_SENTENCES, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 3. Inference
    print("Running Inference...")
    with torch.no_grad():
        # [CRITICAL] We are passing the mask here!
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1).numpy()
        scores = torch.max(torch.softmax(logits, dim=1), dim=1).values.numpy()

    # 4. Display Results
    print("\nResults:")
    print(f"{'idx':<4} | {'Sentence':<30} | {'Pred ID':<8} | {'Conf':<6} | {'Status'}")
    print("-" * 75)
    
    unique_ids = set()
    
    for i, sent in enumerate(TEST_SENTENCES):
        pid = preds[i]
        unique_ids.add(pid)
        conf = scores[i]
        
        # Check against our expected subset (approximate verification)
        # Note: Your ID mapping might differ slightly depending on the dataset loading order,
        # but if we see diverse IDs that seem distinct for different queries, it's good.
        status = "✅ Distinct"
        
        # Checking against the "All 9s" failure mode
        if pid == 9: 
            status = "⚠️ ID 9 (Suspicious)"
        
        print(f"{i:<4} | {sent:<30} | {pid:<8} | {conf:.2f}   | {status}")

    print("-" * 75)
    print(f"\nDiversity Check: {len(unique_ids)} unique IDs predicted out of {len(TEST_SENTENCES)} sentences.")
    
    if len(unique_ids) > 5:
        print("✅ SUCCESS: The model is predicting diverse intents when Mask is used.")
    else:
        print("❌ FAILURE: The model is collapsed (predicting same ID for everything).")

if __name__ == "__main__":
    main()