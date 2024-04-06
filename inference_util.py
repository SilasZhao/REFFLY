import re 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import json
import math
import matplotlib.pyplot as plt
import pickle
#perplexity
#diversity
file = open("/home/songyan/Real_M2L-main/llama/cmu_dict.pkl",'rb')
cmu_dict = pickle.load(file)
file.close()
# print(cmu_dict)

def get_stress(phone):
    '''
    Takes in a piece of phonetics of a word, and produce the stress (1) and unstress (0).
    Example input: ['AH0 M EY1 Z IH0 NG']
    The phones could be retrived using the following command: phone = pronouncing.phones_for_word('amazing')
    Example output: [0, 1, 0]
    '''
    stress = []
    # print(phone)
    for s in phone.split():
        # print(s)
        if s[-1].isdigit():
            # print()
            if s[-1] == '2':
                stress.append(0)
            else:
                stress.append(int(s[-1]))
    # print("reached the end in get_stress")
    return stress
def get_word_stress(word):
    special_words_1s = [ "aren't","won't","can't","don't",  "weren't","shan't", "ain't", "won't"]
    special_words_2s = ["wouldn't", "hasn't", "didn't", "couldn't", "shouldn't", "isn't", "haven't", "doesn't", "hadn't", "mightn't", "mustn't",  "might've", "haven't", "isn't", "oughtn't", "wasn't", "wouldn't"]
    words = word.split("-")
    all_stress = []
    for word in words:
        word = re.sub(r'^[^a-zA-Z]+', '', word)
        if word in special_words_1s:
            all_stress += [1]
            continue
        if word in special_words_2s:
            all_stress += [1,0]
            continue
        # pattern = r"S+'\S+$"
        word = re.sub(r"\b\w+'\w+", lambda match: re.sub(r"'\w+", '', match.group()), word)
        # print(f'after remove contraction {word}')
        word = re.sub(r'[^\w\s]', '', word)
        word = word.lower().strip()
        phe = cmu_dict[word]
        stress = " ".join(phe[0])
        stress = get_stress(stress)
        all_stress += stress
    return all_stress
def remove_contractions(text):
    # print(text)
    pattern = r"'\w+$"
    text = re.sub(pattern, '', text)
    # text = re.sub(r'[^\w]', ' ', text)
    # print(text)
    # print(text)
    return text
def get_common_word(word):
    for i in common_word:
        if i["word"].lower() == word:
            return i["stress"]
    return None
def get_previous_lyrics(Input):
    pattern = r"Previously generated lyrics are: '\s*(.*?)\s*'Title is"

    # Search for the pattern and extract the content
    match = re.search(pattern, Input, re.DOTALL)  # re.DOTALL to match across multiple lines
    extracted_lyrics = match.group(1) if match else None
    return extracted_lyrics
def get_stress_clean(word):
    cleaned_word = remove_contractions(word)
    cleaned_word = cleaned_word.lower().strip()
    try:
        # print(f'cleaned_word {cleaned_word} {len(cleaned_word.split(" "))}')
    
        stress = cmu_dict[cleaned_word]
        stress = " ".join(cmu_dict[cleaned_word][0])
        return get_stress(stress)
    except:
        stress =  get_common_word(cleaned_word)
        if stress is not None:
            return stress
        raise KeyError(f'couldn"t fine stress info for word "{cleaned_word}"')
    
    
def extract(original):
    sentence = original
    original = original.split(' ')
    num_syllable = 0 
    stress_index = []
    for word in original:
        try:
            # cleaned_word = re.sub(r'[^\w\s]', '', word)
            stress = get_word_stress(word)
            cleaned_word = remove_contractions(word)
            cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
            # stress = get_stress_clean(word)
            if is_noun_or_verb(get_word_pos_in_sentence(sentence,cleaned_word)):
                # print(word)
                for i in range(len(stress)):
                    if stress[i] == 1:
                        stress_index.append(num_syllable + i)
            num_syllable += len(stress)
        except:
            #fail then return -1,[]
            return -1,[]
    # print(num_syllable,stress_index)
    return num_syllable,stress_index

def valid_line(sentence):
    sentence = sentence.split()
    valid = True
    for word in sentence:
        cleaned_word = remove_contractions(word)
        cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
        # print(cleaned_word)
        try: 
            phenom = cmu_dict[cleaned_word.lower()][0]
        except:
            return get_common_word(word) is not None
        # if pronouncing.phones_for_word(cleaned_word.strip()) == []:
        #     valid = False
    return valid 
def get_word_pos_in_sentence(sentence, word):
    # Tokenize and POS tag the sentence
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    # Find and return the POS tag of the specified word
    for token, pos in tagged:
        if token == word:
            # print(pos)
            
            return pos
	
    # print("ERROR!!!!!!: Word not found in the sentence")
    return "Word not found in the sentence"

def is_noun_or_verb(pos_tag):
    """
    Check if the given POS tag is a noun or verb.
    Args:
    pos_tag (str): The POS tag to check.

    Returns:
    int: 1 if the tag is a noun or verb, 0 otherwise.
    """
    # Noun tags
    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}

    # Verb tags
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN','VBP','VBZ'}

    if pos_tag in noun_tags or pos_tag in verb_tags:
        return True
    else:
        return False
def get_word_with_phenom(i):
    stress = get_word_stress(i)
    # stress = pronouncing.phones_for_word(cleaned_word)[0]
    stress = [("/STRESSED/" if i != 0 else "/UNSTRESSED/" )for i in stress]
    return i + "(" + '-'.join(stress)+")"
def is_important_word(sentence,word):
    cleaned_word = remove_contractions(word)
    cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
    # phone = pronouncing.phones_for_word(cleaned_word.strip())[0]
    # stress = get_stress(phone)
    if is_noun_or_verb(get_word_pos_in_sentence(sentence,cleaned_word)):
        return True
    else:
        return False 
def create_input(rephrased,constraint,title,history_info = "No previously generated lyric"):
    if len(constraint) == 0 or not valid_line(rephrased):
        return None
    constraints_processed = []
    for i in range(len(constraint)):
        constraints_processed.append(f"S_{i}: /STRESSED/" if constraint[i] == 1 else f"S_{i}: /UNSTRESSED/")
    # for pair in loaded_pairs:
    rephrased_words = rephrased.split() 
    data_rephrased = []
    important_rephrased = []
    num_syllable_rephrased, important_index_rephrased = extract(rephrased)
    for i in rephrased_words:
        
        data_rephrased.append(get_word_with_phenom(i))
        if is_important_word(rephrased,i):
            important_rephrased.append(i)
    num_syllable = len(constraint)
    data_rephrased = ' '.join(data_rephrased)
    if num_syllable == num_syllable_rephrased:
        modify = "keep the number of syllables the same as in the original sentence"
    elif num_syllable_rephrased > num_syllable:
        modify = "rephrasing the original sentence so that generated lyrics have less syllables"
    else:
        modify = "rephrasing the original sentence so that generated lyrics have more syllables"
    CoT_input = f'The goal is to firstly, match the number of syllables in the music constraint, and secondly, \
match the important word to the /STRESSED/ syllables.\
The music constraint indicates that there should be {num_syllable} syllables in the generated lyrics. The original sentence has {num_syllable_rephrased} syllables. \
Therefore, you should {modify}.\
The important words in the original lyric is {important_rephrased}, and the syllables for each word is {data_rephrased}.\
Therefore, we want to rephrase the sentence, so that 1, the number of syllables in the generated\
lyric is {num_syllable} by {modify}, 2, the stress of each of the important word in the generated lyric matches with the music constraint,\
and 3, it is fluent, singable, and coherent with the previously generated lyrics.'
        
    #deleted the "." before Title.
    Input = f"Lyric that needed to be revised based on the music constraint: '{rephrased}'. Previously generated lyrics are: '{history_info }'\
Title is '{title}'. \
The music constraint: {' '.join(constraints_processed)}. {CoT_input}"
    return Input


output= " The generated lyric is 'With every step, you take, I'm transported on a journey, feeling the rhythm, feeling alive', and the corresponding syllables for each word is With [W IH1 DH](/STRESSED/) every [EH1 V ER0 IY0](/STRESSED/-/UNSTRESSED/-/UNSTRESSED/) step, [S T EH1 P](/STRESSED/) you [Y UW1](/STRESSED/) take, [T EY1 K](/STRESSED/) I'm [AY1](/STRESSED/) transported [T R AE0 N S P AO1 R T IH0 D](/UNSTRESSED/-/STRESSED/-/UNSTRESSED/) on [AA1 N](/STRESSED/) a [AH0](/UNSTRESSED/) journey, [JH ER1 N IY0](/STRESSED/-/UNSTRESSED/) feeling [F IY1 L IH0 NG](/STRESSED/-/UNSTRESSED/) the [DH AH0](/UNSTRESSED/) rhythm, [R IH1 DH AH0 M](/STRESSED/-/UNSTRESSED/) feeling [F IY1 L IH0 NG](/STRESSED/-/UNSTRESSED/) alive [AH0 L AY1 V](/UNSTRESSED/-/STRESSED/). The number of syllables of the generated lyric matches with the total number of syllables in the music constraint. Generated lyric has 21 syllables. The important words in the generated lyric is ['step,', 'take,', 'transported', 'journey,', 'feeling', 'rhythm,', 'feeling']. The position of the stressed syllables of these important words are [4, 6, 11, 12, 15, 17, 18], and S_4, S_6, S_11, S_12, S_15, S_17, S_18 are all '/STRESSED/'. The position of stressed syllable of important words in the generated lyric matches the music constraint. Therefore, the generated lyric satisfies the music constraint. The generated lyric is coherent with the previously generated lyrics."
input_string = "Lyric that needed to be revised based on the music constraint: 'Each step you take takes me on a thrilling journey, filling the air with emotion and confidence'. Previously generated lyrics are: In a desert town where the sun shines bright, there's a funky flamingo who dances all night, With feathers so vibrant, it's a mesmerizing sight, The funky flamingo sets the stage alight, Filling the air with passion and pride, Its graceful movements, like a fiery ballet, Hips swaying freely, in a playful display, With every step, it captivates the crowd, The funky flamingo, dancing so proud, Oh, the Funky Flamingo Flamenco, see it soar, Dancing to the rhythm, it's begging for more., With every move, it takes you on a wild ride, filling the air with passion and pride, With every move, it takes you on a wild ride, filling the air with passion and pride, From dusk till dawn, it keeps the magic alive, spreading its charm and making the world come alive, The funky flamingo's rhythm has no end, A dance that transcends, making our souls mend., Underneath the moonlight, the night comes alive,, With the funky flamingo, it's a joyful vibe, Its wings spread wide, embracing the beat, The funky flamingo dances with so much heat. Title is 'Opera at the Laundromat'. The music constraint: S_0: /STRESSED/ S_1: /UNSTRESSED/ S_2: /UNSTRESSED/ S_3: /UNSTRESSED/ S_4: /STRESSED/ S_5: /UNSTRESSED/ S_6: /STRESSED/ S_7: /UNSTRESSED/ S_8: /UNSTRESSED/ S_9: /UNSTRESSED/ S_10: /UNSTRESSED/ S_11: /STRESSED/ S_12: /STRESSED/ S_13: /UNSTRESSED/ S_14: /UNSTRESSED/ S_15: /STRESSED/ S_16: /UNSTRESSED/ S_17: /STRESSED/ S_18: /UNSTRESSED/ S_19: /UNSTRESSED/ S_20: /STRESSED/. The goal is to firstly, match the number of syllables in the music constraint, and secondly, match the important word to the /STRESSED/ syllables. The music constraint indicates that there should be 21 syllables in the generated lyrics. The important words in the original lyric is ['step', 'take', 'takes', 'thrilling', 'journey,', 'filling', 'air', 'emotion', 'confidence'], and the syllables for each word is Each [IY1 CH](/STRESSED/) step [S T EH1 P](/STRESSED/) you [Y UW1](/STRESSED/) take [T EY1 K](/STRESSED/) takes [T EY1 K S](/STRESSED/) me [M IY1](/STRESSED/) on [AA1 N](/STRESSED/) a [AH0](/UNSTRESSED/) thrilling [TH R IH1 L IH0 NG](/STRESSED/-/UNSTRESSED/) journey, [JH ER1 N IY0](/STRESSED/-/UNSTRESSED/) filling [F IH1 L IH0 NG](/STRESSED/-/UNSTRESSED/) the [DH AH0](/UNSTRESSED/) air [EH1 R](/STRESSED/) with [W IH1 DH](/STRESSED/) emotion [IH0 M OW1 SH AH0 N](/UNSTRESSED/-/STRESSED/-/UNSTRESSED/) and [AH0 N D](/UNSTRESSED/) confidence [K AA1 N F AH0 D AH0 N S](/STRESSED/-/UNSTRESSED/-/UNSTRESSED/). The total number of syllables in original sentence is 24, and that does not match the number of syllables indicated by the music constraint. Therefore, we want to rephrase the sentence, so that 1, the number of syllables in the generated lyric is 21, 2, the stress of each of the important word in the generated lyric matches with the music constraint, and and 3, it is coherent with the previously generated lyrics"
def get_generated_lyrics(output):
    lyrics_match_single_quote = re.search("and the generated lyric is '(.*?)'[.]? The", output)
    lyrics_match_single_quote2 = re.search("and the generated lyric is '(.*?)'Title", output)
    p1 = lyrics_match_single_quote.group(1) if lyrics_match_single_quote else None
    p2 = lyrics_match_single_quote2.group(1) if lyrics_match_single_quote2 else None
    return p1 if p2 is None else p2
def get_generated_lyrics_old(output):
    lyrics_match_single_quote = re.search("The generated lyric is '(.*?)', and", output)
    return lyrics_match_single_quote.group(1) if lyrics_match_single_quote else None

def get_original_lyrics(Input):
    lyrics_match_single_quote = re.search("Lyric that needed to be revised based on the music constraint: '(.*?)'. Previously", Input)
    return lyrics_match_single_quote.group(1) if lyrics_match_single_quote else None

def get_music_constraint(input_string):
    syllable_pattern = r"S_\d+: /STRESSED/|S_\d+: /UNSTRESSED/"
    # Extracting all syllable notations
    syllable_info = re.findall(syllable_pattern, input_string)
    stress_pattern_list = [1 if "/STRESSED/" in syllable else 0 for syllable in syllable_info]
    return stress_pattern_list
def get_title(Input):
    pattern = r"Title is '\s*(.*?)\s*'.\s*The music constraint"

    # Search for the pattern and extract the content
    match = re.search(pattern, Input, re.DOTALL) 
    return match.group(1) if match else None
try:
    
    with open('common_word.pkl', 'rb') as file_handler:
        common_word = pickle.load(file_handler)
except:
    common_word = []
#diff between the lyric and length of constraint
def get_num_syllable_for_sentence(lyrics):
    num_syllable = 0
    for word in lyrics.split():
        try:
            stress = get_word_stress(word)
            # print(f'word {word}, stress {stress}')
            num_syllable += len(stress)
        except:
            raise ValueError("Input is not valid")
            # num_syllable +=len(get_uncommen_word_stress(word))
    return num_syllable

def syllable_match(lyrics,constraint):
    num_syllable = 0
    for word in lyrics.split():
        try:
            stress = get_stress_clean(word)
            num_syllable += len(stress)
        except:
            num_syllable +=len(get_uncommen_word_stress(word))
    return get_num_syllable_for_sentence(lyrics) -len(constraint)

def get_uncommen_word_stress(word):
    while True:
        try:
            return get_stress_clean(word)
        except:
            for i in common_word:
                if i["word"] == word:
                    return i["stress"]
            q = f"Enter the stress of syllables in {word}: "
            input_string_str = input(q)
            try:
                stress = input_string_str.split()
                stress = [int(i) for i in stress]
                common_word.append({"word":word,"stress":stress})
                return stress
                break   
            except ValueError:
                print("That's not a valid list of int. Please enter a list of 1 or 0s.")
        # common_word.append({"word":word,"stress":stress})
        # return stress
"""
lyrics: string
contraint: list of INT
"""
def stress_match(lyrics,constraint):
    _,index = extract(lyrics)
    num_stress = len(index)
    incorrect = 0
    # print(f' inside stress match lyric: {lyrics}, important index {index} music constraint: {constraint}')
    for i in index:
        if constraint[i] != 1:
            incorrect += 1
    # print(f'num_stress {num_stress}, incorrect {incorrect}')
    return (num_stress,incorrect)

    # print("-----------")
    # print(song['res'])
    # print("-----------")
def get_constraint_distribution(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
    #check music_constraint len and number of syllables in the input
    counter = [0 for i in range(60)]
    # print(counter)
    for sentence in data:
        Input = sentence["Input"]
        constraint = get_music_constraint(Input)
        try:
            counter[len(constraint)] += 1
        except:
            print("invalid")
    print(counter)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counter)), counter, color='skyblue')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Number of data points')
    plt.title('Distribution of length of music constraint')
    plt.xticks(range(0, len(counter), 2))
    plt.savefig("dataset_music_constraint_distribution.png")
def get_diff_original_constraint_distribution(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
    #check music_constraint len and number of syllables in the input
    counter = [0 for i in range(60)]
    # print(counter)
    for sentence in data:
        Input = sentence["Input"]
        lyrics = get_original_lyrics(Input)
        cosntraint = get_music_constraint(Input)
        diff = syllable_match(lyrics,cosntraint)
        try:
            counter[diff+20] += 1
        except:
            print(lyrics)
            print(cosntraint)
            continue
    print(counter)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counter)), counter, color='skyblue')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Number of data points')
    plt.title('Difference between original sentence and len(music constraint)')
    plt.xlim(-20, 40)
    x_labels = [i - 20 for i in range(len(counter))]
    plt.xticks(range(0, len(counter), 2), labels=x_labels[::2])
    plt.savefig("dataset_diff_original_constraint_distribution.png")
    
def plot_with_range(data,out_file,xlabel,ylabel,title, drift = 0,counter_length = 60):
    counter = [0 for i in range(counter_length)]
    invalid = 0
    for i in data:
        try:
            counter[i + drift] += 1
        except:
            invalid += 1
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counter)), counter, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(-drift, counter_length - drift)
    x_labels = [i - drift for i in range(len(counter))]
    plt.xticks(range(0, len(counter), 2), labels=x_labels[::2])
    plt.savefig(out_file)
def truncate_data_set(in_path,out_path,length_threshold = 30):
    with open(in_path,'r') as f:
        data = json.load(f)
    new_data = []
    for i in data:
        i["paired_lines"] = i["paired_lines"] if len(i["paired_lines"]) < length_threshold else i["paired_lines"][:length_threshold]
        new_data.append({"title":i["title"],"lyrics":i["lyrics"],"paired_lines":i["paired_lines"]})
    with open(out_path, "w") as file:
        json.dump(new_data, file, indent = 4) 
        
def get_original_sentence_distribution(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
    #check music_constraint len and number of syllables in the input
    counter = [0 for i in range(60)]
    # print(counter)
    for sentence in data:
        Input = sentence["Input"]
        lyrics = get_original_lyrics(Input)
        num_syllable,index = extract(lyrics)
        try:
            counter[num_syllable] += 1
        except:
            pass
        
    print(counter)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counter)), counter, color='skyblue')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Number of data points')
    plt.title('Distribution of number of syllales in sentences')
    plt.xticks(range(0, len(counter), 2))
    plt.savefig("dataset_original_sentence_distribution.png")
# get_constraint_distribution("/home/songyan/Real_M2L-main/llama/generated_fine_tuned.json")
# get_original_sentence_distribution("/home/songyan/Real_M2L-main/llama/generated_fine_tuned.json")
# get_diff_original_constraint_distribution("/home/songyan/Real_M2L-main/llama/generated_fine_tuned.json")
def sentence_in_doc(sentence, file_path):
    try:
        # Open and read the document
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except:
        return False
        # Check if the sentence is in the document
        if sentence in content:
            return True
        else:
            return False
def data_analysis(file_path,need_iter = False,iter_file = None):
    
    with open(file_path,'r') as file:
        data = json.load(file)
    rephrased_total = 0
    rephrased_miss = 0
    original_total = 0
    original_miss = 0
    rephrased_num_syllable_unmatch = []
    original_num_syllable_unmatch = []
    change = []
    invalid = 0
    correct_original = 0
    correct_rephrased = 0
    # print(data[0])
    all_songs_generated = {}
    all_songs_original={}
    iter_data = []
    for song in data:
        
        # lyrics = get_generated_lyrics(song["result"])
        lyrics = get_generated_lyrics(song["res"])
        # print(lyrics)
        if lyrics is None:
            invalid += 1
            continue
        lyrics_original = get_original_lyrics(song["prompt"])
        constraint = get_music_constraint(song["prompt"])
        title = get_title(song["prompt"])
        if title in all_songs_generated:
            all_songs_generated[title] = all_songs_generated[title] +"\n"+ lyrics
        else:
            all_songs_generated[title] = lyrics
        if title in all_songs_original:
            all_songs_original[title] = all_songs_original[title] +"\n"+ lyrics_original
        else:
            all_songs_original[title] = lyrics_original
            
        # print(constraint)
        syllable_diff = syllable_match(lyrics,constraint) 
        original_syllables = get_num_syllable_for_sentence(lyrics_original)
        generated_syllales = get_num_syllable_for_sentence(lyrics)
        change.append(original_syllables - generated_syllales)
        with open("new_result.txt",'a') as f:
            f.write("original: "+lyrics_original + "\n")
            f.write("rephrased: "+lyrics + "\n")
            f.write(str(constraint) + "\n")
            f.write("different: "+str(syllable_diff) + "\n")
        if syllable_diff == 0:
            with open("successful.txt",'a') as f:
                f.write(f'original lyric: {lyrics_original}\n')
                f.write(f'original lyric has {original_syllables} syllables\n')
                f.write(f'music constraint: {constraint}\n')
                f.write(f'music constraint has {len(constraint)} syllables\n')
                f.write(f'generated lyrics: {lyrics}\n')
                f.write("\n")
            correct_rephrased += 1
        if syllable_diff != 0:
            rephrased_num_syllable_unmatch.append(syllable_diff)
            if need_iter:
                history_info = get_previous_lyrics(song["prompt"])
                title = get_title(song["prompt"])
                new_input = create_input(lyrics,constraint,history_info,title)
                iter_data.append({"Input":new_input})
        # if syllable_diff >4:
        #     print(lyrics)
        #     print(constraint)
        #generated syllables <= number of notes
        if syllable_diff <= 0:
            cur_num_stress,cur_miss=stress_match(lyrics,constraint)
            # print(f'cur miss {cur_miss}')
            rephrased_total += cur_num_stress
            rephrased_miss += cur_miss
        else:
            pass
        syllable_diff = syllable_match(lyrics_original,constraint)
        
        if syllable_diff != 0:
            original_num_syllable_unmatch.append(syllable_diff)
        else:
            # print(lyrics_original)
            # print(len(constraint))
            correct_original += 1
        if syllable_diff <= 0:
            cur_num_stress,cur_miss=stress_match(lyrics_original,constraint)
            # print(f'cur miss {cur_miss}')
            original_total += cur_num_stress
            original_miss += cur_miss
    if need_iter:
        out = open(iter_file, "w") 
        print(iter_file)
        json.dump(iter_data, out, indent = 4) 
    print(f'There are in total {len(data)} songs')
    print(f'invalid {invalid}')
    print(f'among lyrics that matches the number of syllables in music constraint: total_important_syllables in rephrased {rephrased_total},missed matched {rephrased_miss}')
    print(f'among lyrics that matches the number of syllables in music constraint: total_important_syllables in original {original_total},missed matched {original_miss}')
    
    print(f'there are {len(rephrased_num_syllable_unmatch)} rephrased lyrics that does not matches the number of syllables in music constraint.')
    print(f'correct_rephrased {correct_rephrased}')
    print(f'original syllable unmatch total: {len(original_num_syllable_unmatch)}')
    plot_with_range(rephrased_num_syllable_unmatch,"/home/songyan/Real_M2L-main/llama/eval_png/rephrased_unmatch_iter3.png","number of Syllable","Number of Datapoints","rephrased_mismatch",15,30)
    plot_with_range(change,"/home/songyan/Real_M2L-main/llama/eval_png/modified_syllables_iter3.png","number of Syllable","Number of Datapoints","modified",20,40)
    plot_with_range(original_num_syllable_unmatch,"/home/songyan/Real_M2L-main/llama/eval_png/original_unmatch_iter3.png","number of Syllable","Number of Datapoints","original_mismatch",15,30)
    print(f'correct original {correct_original}')
    all_song = []
    all_song.append(all_songs_original)
    all_song.append(all_songs_generated)
    out_file = open("/home/songyan/Real_M2L-main/llama/result/all_songs.json", "w") 
    json.dump(all_song, out_file, indent = 4) 
    # l = [0 for i in range(30)]
    # for i in num_syllable_unmatch:
    #     l[i] += 1
    # print(l)
    # print(f'modified syllables is: {change}')
    with open('common_word.pkl', 'wb') as file_handler:
    # Step 4: Use pickle.dump() to serialize the dictionary
        pickle.dump(common_word, file_handler)
# print(get_word_with_phenom("you're"))
    #check music_constraint len and number of syllables in the input
# print(extract("And I'm sorry that I'm not sorry"))
# data_analysis("/home/songyan/Real_M2L-main/data/data_finetune/negative_samples_result/generated_fine_tuned_iter1_result.json",need_iter=True,iter_file="/home/songyan/Real_M2L-main/data/data_finetune/negative_samples_result/generated_fine_tuned_iter2.json")
# print(get_word_stress("'Title"))
# data_analysis("/home/songyan/Real_M2L-main/data/data_finetune/negative_samples_result/generated_fine_tuned_iter3_3epoch_result.json")
# get_constraint_distribution("/home/songyan/Real_M2L-main/data/data_finetune/generated_fine_tuned_scrape.json")
# get_original_sentence_distribution("/home/songyan/Real_M2L-main/data/data_finetune/generated_fine_tuned_scrape.json")
# get_diff_original_constraint_distribution("/home/songyan/Real_M2L-main/data/data_finetune/generated_fine_tuned_scrape.json")
# truncate_data_set("/home/songyan/Real_M2L-main/data/data_finetune/generated_line_all_2.json","/home/songyan/Real_M2L-main/data/data_finetune/generated_line_all_truncated_2.json",length_threshold = 30)
# with open("/home/songyan/Real_M2L-main/data/data_finetune/generated_line_all_truncated.json") as f:
#     data = json.load(f)
#     for i in range(50):
#         print(len(data[i]["paired_lines"]))
# with open("/home/songyan/Real_M2L-main/llama/generated_fine_tuned_result.json",'r') as f:
#     data = json.load(f)
# i = 0
# prompt = data[2]["prompt"]
# generated_lyrics = get_generated_lyrics(data[2]["res"])
# music_constraint = get_music_constraint(data[2]["prompt"])
# diff = syllable_match(generated_lyrics,music_constraint)
# if diff == 0 or i >= 3:
#     finish = True
# title = get_title(prompt)
# history = get_previous_lyrics(prompt)
# Input = create_input(generated_lyrics,music_constraint,history,title)
# if Input == None:
#     finish = True
#     invalid += 1
# print(Input)
# counter = [0 for i in range(60)]
    # # print(counter)
    # for i in change:
    #     try:
    #         counter[i+30] += 1
    #     except:
    #         print(lyrics)
    #         print(constraint)
    #         continue
    # # print(counter)
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(counter)), counter, color='skyblue')
    # plt.xlabel('Number of Syllables')
    # plt.ylabel('Number of data points')
    # plt.title('Difference between original sentence and len(music constraint)')
    # plt.xlim(-30, 30)
    # x_labels = [i - 30 for i in range(len(counter))]
    # plt.xticks(range(0, len(counter), 2), labels=x_labels[::2])
    # plt.savefig("modified_lyrics_distribution.png")

