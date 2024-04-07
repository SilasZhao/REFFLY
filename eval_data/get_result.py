import json
import sys
sys.path.append("/home/songyan/Real_M2L-main/finetune_llama/Finetune-Llama2-LoRA-main/")
from inference_util import *
with open('/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/song_4_2.json') as f:
    songs = json.load(f)
with open('/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/result_4_3_rephrasing.json') as f:
    results = json.load(f)
# counter = 0
# with open("/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/First_love's_thrill.json") as f:
#     data = json.load(f)
# for sen in data:
#     print(sen["result_sentence"])
# exit()
def count_num_song():
    for song in songs:
        lyrics = song["lyrics"]
        print(lyrics)
        lyrics = lyrics.split('\n')
        for w in lyrics:
            if w == "":
                continue
            counter += 1

def get_result_sentence(results,line):
    res = []
    for result in results:
        if result["original"] == line:
            res.append(result)
    return res
def catagorize():
    result_song = []
    l_diff = []
    invalid = 0
    all_ly = []
    for song in songs:
        l = []
        # print(song)
        lyrics = song["lyrics"]
        title = song["title"]
        lyrics = lyrics.split('\n')
        result_lyric = []
        result_line = []
        print(lyrics)
        all_ly += lyrics
        for line in lyrics:
            res = get_result_sentence(results,line)
            result_line.append(res[0])
            output = ""
            for r in res:
                if r["result_sentence"] is not None:
                    output = r["result_sentence"]
            if output == "":
                invalid +=1
                num_sy,_ = extract(res[0]["original"])
                if num_sy == -1:
                    # print(f'failed ')
                    result_lyric.append("NOT ABLE TO GENERATE")
                    continue
                diff = abs(len(res[0]["constraint"]) - num_sy)
                print(f'failed to get {res[0]["original"]}. Difference between constraint and original is {diff}')
                l_diff.append(diff)
            result_lyric.append(output)
            with open("/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/"+"_".join(title.split(" "))+".json",'w') as f:
                json.dump(result_line,f,indent=4)
            l.append(output)
        print("=======")
        print(result_line)
        print("=======")
        
        result_song.append({"title":title,"result":" ".join(result_lyric)})
    for i in result_song:
        print(i)
        print("------------")
    print(sum(l_diff)/len(l_diff))
    print(1-invalid/len(all_ly))
    print(len(all_ly))
def analyze_matching_rate(file):
    with open(file) as f:
        data = json.load(f)
    all_total, all_incorrect = 0,0
    for song in data:
        if not isinstance(song["total_and_incorrect"],list):
            try:
                total,_ = extract(song["result_sentence"])
                incorrect = 0
            except:
                continue
        else:
            total,incorrect = song["total_and_incorrect"]
        all_total += total
        all_incorrect += incorrect
    print(f'accuracy {1-all_incorrect/all_total}')
# analyze_matching_rate("/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/result_4_3_rephrasing.json")
# analyze_matching_rate("/home/songyan/Real_M2L-main/data/data_finetune/eval/end-to-end-data/result_4_3_no_rephrasing.json")
catagorize()