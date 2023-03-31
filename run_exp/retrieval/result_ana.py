import json
import ipdb

result_file_name = ".result/vc_64-4-1-1-1_64-4-1-1-0-1_64-4-1-1-0_64-4-1-0-0.json"

precision = []
filtering_time = []
with open(result_file_name) as f:
    results = json.load(f)
    
for file in results:
    if file.endswith(".json"):
        continue
    print(file)
    for r in results[file]:
        precision.append(results[file][r]["number_of_matched_graphs"]/results[file][r]["candidates_after_filtering"])
        filtering_time.append(results[file][r]["filtering_time"])

avg_p = sum(precision)/len(precision)
print(avg_p) 
avg_p = sum(filtering_time)/len(filtering_time)
print(avg_p) 
