"""Generate a stratified dev/test split of eval/golden.json with a fixed seed."""

import json
import random
from collections import defaultdict

random.seed(42)

with open("eval/golden.json", "r", encoding="utf-8") as f:
    golden_data = json.load(f)

# Stratified split: split each group
# For answerable questions, alternate assignment across sorted categories
answerable_by_cat = defaultdict(list)
for item in golden_data:
    if item.get("answerable"):
        answerable_by_cat[item.get("category", "")].append(item)

# Sort categories for determinism
ans_dev = []
ans_test = []
toggle = True

for cat in sorted(answerable_by_cat.keys()):
    items = list(answerable_by_cat[cat])
    random.shuffle(items)
    for it in items:
        if toggle:
            ans_dev.append(it)
        else:
            ans_test.append(it)
        toggle = not toggle

# For unanswerable easy (10 items): 5 dev, 5 test
easy_items = [it for it in golden_data if it.get("category") == "unanswerable_easy"]
random.shuffle(easy_items)
easy_dev = easy_items[:5]
easy_test = easy_items[5:]

# For unanswerable hard (10 items): 5 dev, 5 test
hard_items = [it for it in golden_data if it.get("category") == "unanswerable_hard"]
random.shuffle(hard_items)
hard_dev = hard_items[:5]
hard_test = hard_items[5:]

dev_items = ans_dev + easy_dev + hard_dev
test_items = ans_test + easy_test + hard_test

dev_ids = [x["id"] for x in dev_items]
test_ids = [x["id"] for x in test_items]

split_data = {
    "random_seed": 42,
    "total_questions": len(golden_data),
    "dev_count": len(dev_items),
    "test_count": len(test_items),
    "dev_ids": dev_ids,
    "test_ids": test_ids,
    "dev": dev_items,
    "test": test_items
}

with open("eval/split.json", "w", encoding="utf-8") as f:
    json.dump(split_data, f, indent=2)

print(f"Generated eval/split.json: {len(dev_items)} dev, {len(test_items)} test")
dev_ans = sum(1 for x in dev_items if x.get("answerable"))
test_ans = sum(1 for x in test_items if x.get("answerable"))
dev_easy = sum(1 for x in dev_items if x.get("category") == "unanswerable_easy")
test_easy = sum(1 for x in test_items if x.get("category") == "unanswerable_easy")
dev_hard = sum(1 for x in dev_items if x.get("category") == "unanswerable_hard")
test_hard = sum(1 for x in test_items if x.get("category") == "unanswerable_hard")
print(f"Dev: {dev_ans} ans, {dev_easy} easy unans, {dev_hard} hard unans")
print(f"Test: {test_ans} ans, {test_easy} easy unans, {test_hard} hard unans")
