import json

with open("/Users/vansh/PycharmProjects/Text-Extractor/output/rotated/res_0.txt") as f:
    for line in f:
        data = json.loads(line)
        break

print(data['res'])
last_column_positions = [entry['text_region'][-1][0] for entry in data['res']]
max_last_column_position = max(last_column_positions)


