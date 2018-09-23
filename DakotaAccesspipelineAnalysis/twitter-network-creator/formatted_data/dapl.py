import json
import glob

path_lst = glob.glob("E:/archiveteam-twitter-stream-2016-11/11/twitter-network-creator/formatted_data/*.json")
for file in path_lst:
    with open(file,'r') as f:
        data = json.load(f)
        print(len(data))
    