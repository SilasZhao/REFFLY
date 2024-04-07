import json
from inference_util import *
import torch

import evaluate
from tqdm import tqdm
from itertools import combinations

bertscore = evaluate.load("bertscore")
def add_elements_combinations(n, k):
    """
    Generate all possible ways to add k elements into a list of length n.
    
    Parameters:
    n (int): The length of the initial list.
    k (int): The number of elements to add.
    
    Returns:
    List of tuples: Each tuple represents a combination of positions 
                    where new elements could be inserted in the list.
    """
    # Generate all positions in the final list of size n+k
    positions = range(n + k)
    
    # Generate all combinations of k positions out of n+k
    all_combinations = list(combinations(positions, k))
    
    return all_combinations

# print(add_elements_combinations(15,2))
def get_sentence_stress(sentence):
    stress_all = []
    for word in sentence.split():
        try:
            stress = get_word_stress(word)
            if not is_important_word(sentence,word):
                stress = [0 for i in range(len(stress))]
            stress_all += stress
            
        except:
            raise ValueError("Input is not valid")
    return stress_all
"""
input: 
sentences:[[sent1],[sent2],...]
reference:
output:
[{'sentence': "Make me laugh, but don't you cry", 'bertscore': [0.7483960390090942]}, {'sentence': "Make me laugh, but don't you cry", 'bertscore': [0.7483960390090942]}...]
"""
def get_bert_score(sentences,reference):
    result = []
    try:
        for sentence in sentences:
            result.append({"sentence":sentence,"bertscore":bertscore.compute(
                predictions=[sentence], references=[reference], model_type="distilbert-base-uncased"
            )["precision"]})
    except Exception as e:
        print(sentences,reference)
        print(e)
        exit()
        
    return result
"""
sentence_stress: list of integers, 1 indicates important word
constraint: list of integers,music constraint
"""
def mapping_result(sentence_stress,constraint):
    incorrect = 0
    for i in range(len(sentence_stress)):
        if sentence_stress[i] == 1 and constraint[i] != 1:
            incorrect += 1
    # print(f'num_stress {num_stress}, incorrect {incorrect}')
    return (sum([i == 1 for i in sentence_stress]),incorrect)
def mapping(sentences,constraint):
    result = []
    for sentence in sentences:
        try:
            sentence_stress = get_sentence_stress(sentence)
            # print(f'inside mapping sentence {sentence}, stress {sentence_stress}')
        except:
            continue
        num_note = len(constraint)
        num_tie = num_note - len(sentence_stress)
        min_incorrect = 1000
        cur_pos = None
        combinations = add_elements_combinations(len(sentence_stress),num_tie)
        # print(combinations)
        
        if combinations[0] == ():
            result.append({"sentence":sentence,"tie_pos":(),"total_and_incorrect":stress_match(sentence,constraint)})
            continue
        for pos in combinations:
            if len(pos) == 0:
                continue
            if pos[0] == 0:
                continue
            if len(pos) > 1:
                invalid = False
                for i in range(len(pos) - 1):
                    if pos[i + 1] - pos[i] == 1:
                        invalid = True
                if invalid: continue
            sentence_stress_temp = sentence_stress.copy()
            for i in pos:
                sentence_stress_temp.insert(i,-1)
            # print(f'pos: {pos}, sentence_stress_temp {sentence_stress_temp}')
            total,incorrect = mapping_result(sentence_stress_temp,constraint)
            # print(f'inside mapping total, music constraint {constraint}, incorrect {total}, {incorrect}; sentence {sentence}, ')
            if incorrect < min_incorrect:
                cur_pos = pos
                min_incorrect = incorrect
        result.append({"sentence":sentence,"tie_pos":cur_pos,"total_and_incorrect":(total,min_incorrect)})
    return result
def get_generated_lyrics_old(output):
    lyrics_match_single_quote = re.search("and the generated lyric is '(.*?)'. The", output)
    lyrics_match_single_quote1 = re.search("and generated lyric is '(.*?)'. The", output)
    if not lyrics_match_single_quote:
        return lyrics_match_single_quote1.group(1) if lyrics_match_single_quote1 else None
    return lyrics_match_single_quote.group(1) if lyrics_match_single_quote else None
"""
"""
def get_result_from_over_generation(item):
    # print(f'item{item}')
    invalid = 0
    item["lyrics"] = [get_generated_lyrics_old(l) for l in item["res"]]
    try:
        original = get_original_lyrics(item["prompt"])
        constraint = get_music_constraint(item["prompt"])
    except:
        invalid+= 1
        return None,None,None
    if len(original.split()) == 0:
        return None,None,None
    sentences = []
    all_match = []
    for l in item["lyrics"]:
        try:
            lyric_length = get_num_syllable_for_sentence(l)
        except:
            continue
        # print(f'lyric: {l} num_sy:{lyric_length}')
        if (l is not None and lyric_length <= len(constraint) and lyric_length >= len(constraint) - 2):
            sentences.append(l)
        if l is not None and lyric_length == len(constraint):
            all_match.append(l)
    if len(sentences) == 0:
        item["selected"] = None
        return None,None,None
    # print(sentences)
    # print(all_match[0])
    # return all_match[0]
    # return all_match[0]
    
    map_result = mapping(sentences=sentences,constraint=constraint)
    potential_sentence = []
    for sentence in map_result:
        if sentence["total_and_incorrect"][1] == 0:
            potential_sentence.append(sentence)
    if len(potential_sentence) <= 1:
        result_sentence = min(map_result, key=lambda x: x['total_and_incorrect'][1])
        # print(result_sentence)
        selected_sentence = result_sentence["sentence"]
        print(f'selected_sentence {selected_sentence}')
        item["selected"] = {"sentence":result_sentence,"bertscore":bertscore.compute(
            predictions=[selected_sentence], references=[original], model_type="distilbert-base-uncased"
        )["precision"]}
        
        return selected_sentence,result_sentence["tie_pos"],result_sentence['total_and_incorrect']
    p = {}
    p_sentences = []
    for s in potential_sentence:
        p[s["sentence"]] = s
        p_sentences.append(s["sentence"])
    # potential_sentence = [s["sentence"] for s in potential_sentence]
    bert_score = get_bert_score(p_sentences,original)
    print(bert_score)
    selected_sentence = max(bert_score, key=lambda x: x['bertscore'])
    print(f'selected_sentence {selected_sentence}')
    selected_sentence["tie_pos"] = p[selected_sentence["sentence"]]['tie_pos']
    item["selected"] = {"sentence":p[selected_sentence["sentence"]],"bertscore":selected_sentence["bertscore"]}
    return selected_sentence["sentence"],selected_sentence["tie_pos"],0
def processing_song(input_f,output_f):
    # path = "/home/songyan/Real_M2L-main/data/data_finetune/negative_samples_result/overgenerate/generated_fine_tuned_iter3_2epoch_result.json"
    f = open(input_f)
    data = json.load(f)
    inqualified = 0
    invalid = 0
    total = 0
    for item in tqdm(data):
        total += 1
        item["lyrics"] = [get_generated_lyrics_old(l) for l in item["res"]]
        # print(item["lyrics"])
        try:
            original = get_original_lyrics(item["prompt"])
            constraint = get_music_constraint(item["prompt"])
        except:
            invalid+= 1
            continue
        if len(original.split()) == 0:
            invalid+=1 
            continue
        sentences = []
        for l in item["lyrics"]:
            try:
                lyric_length = get_num_syllable_for_sentence(l)
                
            except:
                invalid+=1
                continue
            if (l is not None and lyric_length <= len(constraint) and lyric_length >= len(constraint) - 2):
                sentences.append(l)
        if len(sentences) == 0:
            item["selected"] = None
            continue
        
        map_result = mapping(sentences=sentences,constraint=constraint)
        potential_sentence = []
        for sentence in map_result:
            if sentence["total_and_incorrect"][1] == 0:
                potential_sentence.append(sentence)
        if len(potential_sentence) <= 1:
            result_sentence = min(map_result, key=lambda x: x['total_and_incorrect'][1])
            # print(result_sentence)
            selected_sentence = result_sentence["sentence"]
            item["selected"] = {"sentence":result_sentence,"bertscore":bertscore.compute(
                predictions=[selected_sentence], references=[original], model_type="distilbert-base-uncased"
            )["precision"]}
            continue
        p = {}
        p_sentences = []
        for s in potential_sentence:
            p[s["sentence"]] = s
            p_sentences.append(s["sentence"])
        # potential_sentence = [s["sentence"] for s in potential_sentence]
        bert_score = get_bert_score(p_sentences,original)
        selected_sentence = max(bert_score, key=lambda x: x['bertscore'])
        item["selected"] = {"sentence":p[selected_sentence["sentence"]],"bertscore":selected_sentence["bertscore"]}
    print(f'there are {invalid} invalid')
        # print(bertscore)
    with open(output_f,'w') as f:
        json.dump(data,f,indent=4)
def processing_song_without_mapping(input_f,output_f):
    # path = "/home/songyan/Real_M2L-main/data/data_finetune/negative_samples_result/overgenerate/generated_fine_tuned_iter3_2epoch_result.json"
    f = open(input_f)
    data = json.load(f)
    inqualified = 0
    invalid = 0
    total = 0
    for item in tqdm(data):
        total += 1
        item["lyrics"] = [get_generated_lyrics_old(l) for l in item["res"]]
        try:
            original = get_original_lyrics(item["prompt"])
            constraint = get_music_constraint(item["prompt"])
        except:
            invalid+= 1
            continue
        if len(original.split()) == 0:
            invalid+=1 
            continue
        sentences = []
        for l in item["lyrics"]:
            try:
                lyric_length = get_num_syllable_for_sentence(l)
            except:
                invalid+=1
                continue
            if (l is not None and lyric_length == len(constraint)):
                sentences.append(l)
        if len(sentences) == 0:
            item["selected"] = None
        else:
            bert_score = get_bert_score(sentences,original)
            selected_sentence = max(bert_score, key=lambda x: x['bertscore'])
            item["selected"] = selected_sentence
    with open(output_f,'w') as f:
        json.dump(data,f,indent=4)
# m = ["6 syllables, and the generated lyric is 'And the beat goes on, and on, and on'. The corresponding syllables for each word is And(/UNSTRESSED/) the(/UNSTRESSED/)", "6 syllables, and generated lyric is 'In the melody, the heart beats,'. The corresponding syllables for each word is ['In(/UNSTRESSED/)', 'the(/UNSTRESSED", "6 syllables, and the generated lyric is 'The beat, the rhythm, the sound'. The corresponding syllables for each word is The(/UNSTRESSED/) beat,(/STRESSED/) the(/", "6 syllables, and generated lyric is 'The heart beats in time with the rhythm,'. The corresponding syllables for each word is ['The(/UNSTRESSED/)', 'heart(/STRESSED", "6 syllables, and the generated lyric is 'Rhythm is the cure'. The corresponding syllables for each word is Rhythm(/STRESSED/-/UNSTRESSED/) is(/STRESSE", "6 syllables, and the generated lyric is 'Life's a beat, let's take it'. The corresponding syllables for each word is Life's(/STRESSED/) a(/UNSTRESSED", "6 syllables, and generated lyric is 'In tune, the heart beats,'. The corresponding syllables for each word is ['In(/UNSTRESSED/)', 'tune,(/STRESSED", '7 syllables, but the music constraint indicates that generated lyrics should have 6 syllables. Therefore, we should rephrase the original sentence so that generated lyrics have less syllables. The following lyric does not', "6 synds, and generated lyric is 'In tune, the heart beats,'. The corresponding syllables for each word is ['In(/UNSTRESSED/)', 'tune,(/STRESSED/)", '7 syllables, but the music constraint indicates that the generated lyric should have 6 syllables. Therefore, we should rephrase the original sentence so that generated lyrics have less syllables. The following lyric does', "6 syllables, and the generated lyric is 'Dancin' in the moonlight'. The corresponding syllables for each word is Dancin'(/STRESSED/-/UNSTRESSED/) in(/UN", "6 syllables, and generated lyric is 'From the melody, the heart,'. The corresponding syllables for each word is ['From(/STRESSED/)', 'the(/UNSTRESSED/)', '", "7 syllables, because the music constraint indicates that the generated lyrics should have 6 syllables. The generated lyric is 'The heartbeat, the rhythm, the song'. The corresponding syllables for each word", "6 syllables, and generated lyric is 'The rhythm pulses with the heart,'. The corresponding syllables for each word is ['The(/UNSTRESSED/)', 'rhythm(/STRESSED", "6 syllables, and generated lyric is 'Within the melody, the heart beats,'. The corresponding syllables for each word is ['Within(/UNSTRESSED/-/STRESSED/)',", "6 syndyles, and the generated lyric is 'And the night is here in all its gear'. The corresponding syllables for each word is And(/UNSTRESSED/) the(/UNSTRESSED/) night(/"]
# item= {}
# item["res"] = m
# item["prompt"] = "Lyric that needed to be revised based on the music constraint: 'Through the rhythm, the heart weighs,'. Previously generated lyrics are: 'Jazz on a summer's day, swing, sway, embrace the day, And the night is here in all its gear, And the night is here in all its gear, so clear, oh, Each breath in, we're here now, Life's tune, in jazz mode, From our heads to our toes, 'Title is 'Echoes in Jazz3'. The music constraint: S_0: /UNSTRESSED/ S_1: /STRESSED/ S_2: /UNSTRESSED/ S_3: /UNSTRESSED/ S_4: /STRESSED/ S_5: /UNSTRESSED/. The goal is to firstly, match the number of syllables in the music constraint, and secondly, match the important word to the /STRESSED/ syllables.The music constraint indicates that there should be 6 syllables in the generated lyrics. The original sentence has 7 syllables. Therefore, you should rephrasing the original sentence so that generated lyrics have less syllables.The important words in the original lyric is ['rhythm,', 'heart', 'weighs,'], and the syllables for each word is Through(/STRESSED/) the(/UNSTRESSED/) rhythm,(/STRESSED/-/UNSTRESSED/) the(/UNSTRESSED/) heart(/STRESSED/) weighs,(/STRESSED/).Therefore, we want to rephrase the sentence, so that 1, the number of syllables in the generatedlyric is 6 by rephrasing the original sentence so that generated lyrics have less syllables, 2, the stress of each of the important word in the generated lyric matches with the music constraint,and 3, it is fluent, singable, and coherent with the previously generated lyrics."
# print(get_result_from_over_generation(item))
print(mapping(["To this day when I hear that song I think of you"],[
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0
        ]))
# print(get_num_syllable_for_sentence('From our heads to our toes,'))
# # p = "/home/songyan/Real_M2L-main/data/data_finetune/negative_samples_result/overgenerate/generated_fine_tuned_eval_result.json"
# out = "/home/songyan/Real_M2L-main/data/data_finetune/negative_samples_result/overgenerate/song_struct/13b_1epoch_result.json"
# input_f = "/home/songyan/Real_M2L-main/data/data_finetune/negative_samples_result/overgenerate/song_struct/generated_fine_tuned_13b_1epoch_result.json"
# processing_song(input_f,out)
# # processing_song_without_mapping(p,out)
# with open(out) as f:
#     data = json.load(f)
# invalid = 0
# for item in data:
#     if "selected" not in item or item["selected"] is None:
#         invalid += 1

# print(f'successful rate {1 - invalid/len(data)}')
# sentences = ["you are cute","a pretty girl", "beautiful girl","lovely my dear"]
# reference = "you are lovely"
# constraint = [0,1,1,0]
# print(get_bert_score(sentences,reference))
# print(mapping(sentences,constraint))
# [1,0,0,1,1,0,0,1,1,0,0,0]
# [0, ,1,0,1,0,0,1,1,0,1]   
# [0,0,0,1,1,1,1,2,2,2]

# [0,0,0,1,0,0,0,2,0,0,0,0]
# [0,0,1,1,0,0,1,1,0,0,0]





    
        
            
    
                


        # num_syllable +=len(get_uncommen_word_stress(word))
    
# print(f'successful rate {1- inqualified/total}')