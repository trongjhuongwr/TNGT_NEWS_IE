import json

files = [
    "data\\label_studio\\ouput\\part1.json",
    "data\\label_studio\\ouput\\part2.json",
    "data\\label_studio\\ouput\\part3.json",
    "data\\label_studio\\ouput\\part4.json"
]

merged = []

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        data = json.load(file)
        if isinstance(data, list):
            merged.extend(data)
        else:
            raise ValueError(f"{f} không phải list")

with open("output.json", "w", encoding="utf-8") as out:
    json.dump(merged, out, ensure_ascii=False, indent=2)
