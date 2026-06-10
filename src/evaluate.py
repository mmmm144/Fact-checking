import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DATASET = os.path.join(SCRIPT_DIR, "../data/vie/processed/fact_checking_dataset.json")

def evaluate_dataset():
    print("=================== DATASET EVALUATION & STATISTICS ===================")
    
    if not os.path.exists(FINAL_DATASET):
        print(f"Error: {FINAL_DATASET} not found. Run generate_dataset.py first.")
        return
        
    with open(FINAL_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total = len(data)
    print(f"Total entries: {total}")
    
    # 1. Source distribution
    sources = [entry.get("source_type") for entry in data]
    source_counts = Counter(sources)
    print("\nSource Type Distribution:")
    for src, count in source_counts.items():
        percentage = (count / total) * 100
        print(f"  - {src}: {count} ({percentage:.2f}%)")
        
    # 2. Label distribution
    labels = [entry.get("label") for entry in data]
    label_counts = Counter(labels)
    print("\nLabel Distribution:")
    for lbl, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"  - {lbl}: {count} ({percentage:.2f}%)")
        
    # 3. Source Name distribution (top 10)
    src_names = [entry.get("source_name") for entry in data]
    src_name_counts = Counter(src_names)
    print("\nTop Source Names:")
    for name, count in src_name_counts.most_common(10):
        print(f"  - {name}: {count}")
        
    # 4. Check for potential issues
    empty_claims = sum(1 for e in data if not e.get("claim"))
    empty_texts = sum(1 for e in data if not e.get("original_text"))
    missing_dates = sum(1 for e in data if not e.get("publish_date"))
    
    print("\nIntegrity Checks:")
    print(f"  - Empty claims: {empty_claims}")
    print(f"  - Empty original texts: {empty_texts}")
    print(f"  - Missing publish dates: {missing_dates}")
    
    print("\nSample TRUE Entry:")
    true_entries = [e for e in data if e.get("label") == "TRUE"]
    if true_entries:
        sample = true_entries[0]
        print(f"  Source: {sample.get('source_name')} ({sample.get('source_type')})")
        print(f"  Claim: {sample.get('claim')}")
        print(f"  Date: {sample.get('publish_date')}")
    else:
        print("  None found")
        
    print("\nSample FALSE Entry:")
    false_entries = [e for e in data if e.get("label") == "FALSE"]
    if false_entries:
        sample = false_entries[0]
        print(f"  Source: {sample.get('source_name')} ({sample.get('source_type')})")
        print(f"  Claim: {sample.get('claim')}")
        print(f"  Date: {sample.get('publish_date')}")
    else:
        print("  None found")
        
    print("=======================================================================")

if __name__ == "__main__":
    evaluate_dataset()
