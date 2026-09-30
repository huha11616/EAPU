import json
import os
from sklearn.model_selection import train_test_split

SATD = "../data/satd_dataset.jsonl"
NEG = "../data/negative_samples.jsonl"

OUT_E = "../data/expert_lopo"
OUT_A = "../data/anti_lopo"

os.makedirs(OUT_E, exist_ok=True)
os.makedirs(OUT_A, exist_ok=True)


def load(path):
    return [json.loads(l) for l in open(path)]


satd = load(SATD)
neg = load(NEG)

projects = sorted(set(x["project"] for x in satd))


def split(samples, test_project):
    test = [s for s in samples if s["project"] == test_project]
    rest = [s for s in samples if s["project"] != test_project]

    train, val = train_test_split(rest, test_size=0.1, random_state=42)

    return train, val, test


def dump(base, project, name, data):
    d = f"{base}/split_{project}"
    os.makedirs(d, exist_ok=True)

    with open(f"{d}/{name}.jsonl", "w") as f:
        for x in data:
            f.write(json.dumps(x) + "\n")


for p in projects:

    e_train, e_val, e_test = split(satd, p)
    a_train, a_val, a_test = split(neg, p)

    dump(OUT_E, p, "train", e_train)
    dump(OUT_E, p, "val", e_val)
    dump(OUT_E, p, "test", e_test)

    dump(OUT_A, p, "train", a_train)
    dump(OUT_A, p, "val", a_val)
    dump(OUT_A, p, "test", a_test)

print("LOPO split complete.")
