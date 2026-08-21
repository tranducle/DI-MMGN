import json
import collections
from data.split_policy import build_splits, assert_no_overlap

def audit():
    unified_pairs_path = "../dataset_pipeline/v2/unified_pairs.jsonl"
    splits = build_splits(unified_pairs_path)
    
    with open("data/splits.json", "w") as f:
        json.dump(splits, f, indent=2)
        
    print("=== LEAKAGE AUDIT ===")
    
    # Check 1: Overlap
    assert_no_overlap(splits)
    print("Check 1: Zero overlap between train, val, test domains: PASS")
    
    # Check 2: Wikipedia in test
    print(f"Check 2: Wikipedia in test: {any('wikipedia.org' in d for d in splits['test'])} (Expected False)")
    
    # Gather stats
    domains_data = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(unified_pairs_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line.strip())
            dom = r["domain"]
            domains_data[dom]["total"] += 1
            if r["label"] == "defaced":
                vec = r.get("attack_type", "unknown")
                domains_data[dom][f"attack_{vec}"] += 1
                
    # Stats per split
    for split_name, doms in splits.items():
        total_pairs = sum(domains_data[d]["total"] for d in doms)
        vectors = collections.Counter()
        for d in doms:
            for k, v in domains_data[d].items():
                if k.startswith("attack_") and v > 0:
                    vectors[k.replace("attack_", "")] += v
        print(f"\nSplit '{split_name}':")
        print(f"  Domains: {len(doms)}")
        print(f"  Total pairs: {total_pairs}")
        print(f"  Attack vectors represented: {dict(vectors)}")

if __name__ == "__main__":
    audit()
