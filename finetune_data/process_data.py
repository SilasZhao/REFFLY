import json
import random
import pronouncing
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import sys
import re
import pickle
import random
file = open("/home/songyan/Real_M2L-main/llama/cmu_dict.pkl",'rb')
cmu_dict = pickle.load(file)
file.close()
def get_phe(word):
    cleaned_word = remove_contractions(word)
    cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
    return " ".join(cmu_dict[cleaned_word.lower()][0])
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
import nltk
from nltk.corpus import words
from nltk.corpus import cmudict
from tqdm import tqdm
def remove_contractions(text):
    pattern = r"'\w+$"
    return re.sub(pattern, '', text)
def get_data(in_file_name,out_file_name):
    with open(in_file_name) as f:
        loaded_pairs = json.load(f)
    
    for pair in loaded_pairs:
        pair["Music_constraint"] = generate_constraint(pair["Original"])

    with open(out_file_name, 'w') as f:
        json.dump(loaded_pairs,f)

#"Original": "Looking for some education", "Rephrased": "In search of knowledge and enlightenment", "Title: "
"""
Argument:
    original: original sentence, a string
Output:
    number of syllables, index of important syllable (stress syllable of verb/noun)
"""

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
def is_important_word(sentence,word):
    cleaned_word = remove_contractions(word)
    cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
    # phone = pronouncing.phones_for_word(cleaned_word.strip())[0]
    # stress = get_stress(phone)
    if is_noun_or_verb(get_word_pos_in_sentence(sentence,cleaned_word)):
        return True
    else:
        return False 
def get_important_word(original):
    important_word = []
    for word in original:
        cleaned_word = remove_contractions(word)
        cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
        # print(cleaned_word)
        phone = pronouncing.phones_for_word(cleaned_word.strip())[0]
        stress = get_stress(phone)
        if is_noun_or_verb(get_word_pos_in_sentence(sentence,cleaned_word)):
            # print("is noun")
            for i in range(len(stress)):
                if stress[i] == 1:
                    stress_index.append(num_syllable + i)
        num_syllable += len(stress)

def extract(original):
    sentence = original
    original = original.split(' ')
    num_syllable = 0 
    stress_index = []
    # print(original)
    for word in original:
        try:
            # cleaned_word = re.sub(r'[^\w\s]', '', word)
            cleaned_word = remove_contractions(word)
            cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
            # print(cleaned_word)
            phe = cmu_dict[cleaned_word.lower()]
            phenom = " ".join(phe[0])
            stress = get_stress(phenom)
            # stress = get_stress(phone)
            if is_noun_or_verb(get_word_pos_in_sentence(sentence,cleaned_word)):
                # print("is noun")
                for i in range(len(stress)):
                    if stress[i] == 1:
                        stress_index.append(num_syllable + i)
            num_syllable += len(stress)
        except:
            #fail then return -1,[]
            return -1,[]
    # print(num_syllable,stress_index)
    return num_syllable,stress_index

"""
Return [] if could not get the phenom of some words in the input.
"""
def generate_constraint(original):
    num_syllable,stress_index = extract(original)
    threshold = 0.2
    rand = 1 #rand function
    cosntraint = []
    if num_syllable == -1:
        return []
    for i in range(num_syllable):
        cosntraint.append(1 if (i in stress_index or random.random() < threshold) else 0)
    return cosntraint
def extract_phenom(word):
    return pronouncing.phones_for_word(word)
'''
// Important word match with important note
// Step 1. Add syllable for each word in lyrics
// Step 2. CoT: Fill the template 
// CoT: make sure syllable count match with number of notes -> rephase/change word to match music contraints
{
  "Input": ,
  "Output": Oringnal_lyric + syllable
}
// Unimportant word match NOT with important note
// Step 0. Use the first fine-tuned model
// Step 1. Reject Sampling + Evaluator (match rate)
// Step 1. Few-shot ChatGPT to find unimportant words -> generate data for finetune
[
'''
def process_data(pair):#in_file_name,out_file_name
    # with open(in_file_name) as f:
    #     loaded_pairs = json.load(f)

    # for pair in loaded_pairs:
    original = pair["Original"]
    rephrased = pair["Rephrased"]
    constraint = pair["Music_constraint"]
    constraints_processed = []
    for i in range(len(constraint)):
        constraints_processed.append(f"S_{i}: /STRESSED/" if constraint[i] == 1 else f"S_{i}: /UNSTRESSED/")
    
    original_words = original.split()
    rephrased_words = rephrased.split()
    data_original = []
    data_rephrased = []
    important_original = []
    important_rephrased = []
    num_syllable_original, important_index_original = extract(original)
    num_syllable_rephrased, important_index_rephrased = extract(rephrased)
    for i in original_words:
        stress = pronouncing.phones_for_word(remove_contractions(i))[0]
        
        stress = [("/STRESSED/" if i != 0 else "/UNSTRESSED/" )for i in get_stress(stress)]
        data_original.append(i + " [" + str(pronouncing.phones_for_word(i)[0])+"](" + '-'.join(stress)+")")
        if is_important_word(original,i):
            important_original.append(i)
    for i in rephrased_words:
        data_rephrased.append(i +" /" +str(pronouncing.phones_for_word(i)[0])+"/")
        if is_important_word(rephrased,i):
            important_rephrased.append(i)
    num_syllable = len(constraint)
    data_rephrased = ' '.join(data_rephrased)
    Match = "match" if num_syllable == num_syllable_rephrased else "does not match" 
    data_original = ' '.join(data_original) #/EH2 N L AY1 T AH0 N M AH0 N T/ S_1: Stressed, S_2 Unstressed,... [0,1,0]
    CoT_input = f'The goal is to firstly, match the number of syllables in the music constraint, and secondly, \
    match the important word to the /STRESSED/ syllables.\
    The music constraint indicates that there should be {num_syllable} syllables in the generated lyrics.\
    The important words in the original lyric is {important_rephrased}, and the syllables for each word is {data_rephrased}.\
    The total number of syllables in original sentence is {num_syllable_rephrased}, and that {Match} the number \
    of syllables indicated by the music constraint.\
    Therefore, we want to rephrase the sentence, so that firstly, the number of syllables in the generated\
    lyric is {num_syllable}, and secondly, the stress of each of the important word in the generated lyric matches with the music constraint.'
    
    CoT_output = f"The generated lyric is '{original}', and the corresponding syllables for each word is {data_original}, \
    that matches with the total number of syllables in the music constraint. Generated lyric has {num_syllable_original} syllables.\
     The important words in the generated lyric is {important_original}.\
    The position of the stressed syllables of these important words are {important_index_original}, \
    and {', '.join(['S_' + str(i) for i in important_index_original])} are all '/STRESSED/'. \
    The position of stressed syllable of important words matches the music constraint."

    Input = f"Lyric that needed to be revised based on the music constraint: '{rephrased}', \
    music constraint: {' '.join(constraints_processed)}. {CoT_input}"
    Output = CoT_output
    return Input,Output
def get_stress_clean(word):
    cleaned_word = remove_contractions(word)
    cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
    phone = pronouncing.phones_for_word(cleaned_word.strip())[0]
    return get_stress(phone)
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
            return False
        # if pronouncing.phones_for_word(cleaned_word.strip()) == []:
        #     valid = False
    return valid


def get_word_with_phenom_backup(i):
    cleaned_word = remove_contractions(i)
    cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
    stress = pronouncing.phones_for_word(cleaned_word)[0]
    stress = [("/STRESSED/" if i != 0 else "/UNSTRESSED/" )for i in get_stress(stress)]
    return i + " [" + str(pronouncing.phones_for_word(cleaned_word)[0])+"](" + '-'.join(stress)+")"
def get_word_with_phenom(i):
    cleaned_word = remove_contractions(i)
    cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word)
    stress = get_phe(cleaned_word)
    # stress = pronouncing.phones_for_word(cleaned_word)[0]
    stress = [("/STRESSED/" if i != 0 else "/UNSTRESSED/" )for i in get_stress(stress)]
    return i + "(" + '-'.join(stress)+")"

def get_previous_lyrics(Input):
    pattern = r"Previously generated lyrics are: '\s*(.*?)\s*'Title is"

    # Search for the pattern and extract the content
    match = re.search(pattern, Input, re.DOTALL)  # re.DOTALL to match across multiple lines
    extracted_lyrics = match.group(1) if match else None
    return extracted_lyrics

def create_input(rephrased,constraint,history_info,title):
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

def process_lyrics_with_song_structure_neg(dic):
    structure = ["Verse_1","Chorus_1","Verse_2","Chorus_2","Bridge"]
    # history_info = ""
    # print(dic)
    history = []
    data = []
    for struct in structure:
        history.append(f"({' '.join(struct.split('_'))}) ")
        # print("yes")
        lines = dic["lyrics"][struct].split('\n')
    
        lines = [line.strip() for line in lines]
        # print(f'lines {lines}')
        # print("--------------")
        title = re.sub(r'"', '', dic["Title"])
        pair = dic["paired_lines"] 
        invalid = 0
        for line in lines:
            for sentence in pair:
                if sentence["Original"][-1] == ".":
                    sentence["Original"] = sentence["Original"][:-1]
                if sentence["Original"] == line:
                    original = sentence["Original"]
                    rephrased = sentence["Rephrased1"]
                    rephrased2 = sentence["Rephrased2"]
                    # print(f'original: {original}')
                    # print(f'rephrased: {rephrased}')
            try:
                constraint = generate_constraint(original)
            except:
                #didnt find this line, pass
                continue
            if len(constraint) == 0 or not valid_line(original) or not valid_line(rephrased) or not valid_line(rephrased2):
                invalid += 1
                # print(f"didn't get the music constraint: {line}")
                continue
            if original == rephrased and original == rephrased2:
                invalid += 1
                 
                continue
            constraints_processed = []
            for i in range(len(constraint)):
                constraints_processed.append(f"S_{i}: /STRESSED/" if constraint[i] == 1 else f"S_{i}: /UNSTRESSED/")
            
            
            num_syllable_original, important_index_original = extract(original)
            num_syllable_rephrased, important_index_rephrased = extract(rephrased)
            num_syllable_rephrased2, important_index_rephrased2 = extract(rephrased2)
            if abs(num_syllable_original - num_syllable_rephrased) < abs(num_syllable_original - num_syllable_rephrased2):
                rephrased,rephrased2 = rephrased2,rephrased
                num_syllable_rephrased, num_syllable_rephrased2 = num_syllable_rephrased2,num_syllable_rephrased
            #     print("---------")
            #     print(f'original {num_syllable_original}, rephrased {num_syllable_rephrased}, rephrased2 {num_syllable_rephrased2}')
            # else:
            #     print(f'---original {num_syllable_original}, rephrased {num_syllable_rephrased}, rephrased2 {num_syllable_rephrased2}')
            original_words = original.split()
            rephrased_words = rephrased.split()
            rephrased_words2 = rephrased2.split()
            #make sure even negative examples improved some.
            data_original = []
            data_rephrased = []
            data_rephrased2 = []
            important_original = []
            important_rephrased = []
            num_orginal_word_stress = []
            num_rephrased_word_stress = []
            num_rephrased2_word_stress = []
            for i in original_words:
                data_original.append(get_word_with_phenom(i))
                num_orginal_word_stress.append(str(len(get_stress_clean(i))))
                if is_important_word(original,i):
                    important_original.append(i)
            for i in rephrased_words:
                data_rephrased.append(get_word_with_phenom(i))
                num_rephrased_word_stress.append(str(len(get_stress_clean(i))))
                if is_important_word(rephrased,i):
                    important_rephrased.append(i)
            for i in rephrased_words2:
                data_rephrased2.append(get_word_with_phenom(i))
                num_rephrased2_word_stress.append(str(len(get_stress_clean(i))))
            #add negative example
            need_negative = True if random.uniform(0,1) > 0.4 else False 
            # print(random.uniform(0,1) )
            if (num_syllable_original == num_syllable_rephrased2) or rephrased == rephrased2:
                # print("yes")
                need_negative = False
            num_syllable = len(constraint)
            data_rephrased = ' '.join(data_rephrased)
            Match = "match" if num_syllable == num_syllable_rephrased else "does not match" 
                
            history_info = ", ".join(history) if history !=[] else "No previously generated lyric"
            data_original = ' '.join(data_original) #/EH2 N L AY1 T AH0 N M AH0 N T/ S_1: Stressed, S_2 Unstressed,... [0,1,0]
            if num_syllable == num_syllable_rephrased:
                modify = "keep the number of syllables the same as in the original sentence"
            elif num_syllable_rephrased > num_syllable:
                modify = "rephrasing the original sentence so that generated lyrics have less syllables"
            else:
                modify = "rephrasing the original sentence so that generated lyrics have more syllables"
            CoT_input = f'The goal is to firstly, match the number of syllables in the music constraint, and secondly, \
    match the important word to the /STRESSED/ syllables.\
    The music constraint indicates that there should be {num_syllable} syllables in the generated lyrics. The original sentence has {num_syllable_rephrased} syllables. \
    Therefore, you should {modify}. \
    The important words in the original lyric is {important_rephrased}, and the syllables for each word is {data_rephrased}. \
    Therefore, we want to rephrase the sentence, so that 1, the number of syllables in the generated \
    lyric is {num_syllable} by {modify}, 2, the stress of each of the important word in the generated lyric matches with the music constraint, \
    and 3, it is fluent, singable, and coherent with the previously generated lyrics.'
            
            CoT_output = f"We want to generate a lyric with {num_syllable_original} syllables, and the generated lyric is '{original}'. The corresponding syllables for each word is {data_original}. It has {'+'.join(num_orginal_word_stress)} = {num_syllable_original} syllables \
    and matches with the total number of syllables in the music constraint ({num_syllable} syllables). \
    The important words in the generated lyric is {important_original}. \
    The position of the stressed syllables of these important words are {important_index_original}, \
    and {', '.join(['S_' + str(i) for i in important_index_original])} are all '/STRESSED/'. \
    The position of stressed syllable of important words in the generated lyric matches the music constraint."
            
            Input = f"Lyric that needed to be revised based on the music constraint: '{rephrased}'. Previously generated lyrics are: '{history_info }.' It is in {struct.split('_')[0]} section. \
    Title is '{title}'. \
    The music constraint: {' '.join(constraints_processed)}. {CoT_input}"
            CoT_output_negative = f"We want to generate a lyric with {num_syllable_original} syllables, and generated lyric is '{rephrased2}'. The corresponding syllables for each word is {data_rephrased2}. It has {'+'.join(num_rephrased2_word_stress)} = {num_syllable_rephrased2} syllables \
and does not match with the total number of syllables in the music constraint ({num_syllable_original} syllables). Further improvement is needed. "

            Output = CoT_output if not need_negative else CoT_output_negative
            data.append({"Input":Input, "Output":Output})
            # print(Output)
            is_pres = False
            for i in [' '.join(s.split('_')) for s in structure]:
                if i == history[-1]:
                    history[-1] = history[-1] + original
                    is_pres = True
            if not is_pres:
                history.append(original)
            
            
            # print(f'appended lyrics: {original}')
    #         print("--------")
    #         print(f'history: {history_info}')
    #         print("--------")
        
    #     print(f'data: {data}')
    # exit()
    return data
def process_lyrics_with_song_structure(dic):
    structure = ["Verse_1","Chorus_1","Verse_2","Chorus_2","Bridge"]
    history_info = ""
    # print(dic)
    for struct in structure:
        history_info += f"({' '.join(struct.split('_'))}) \n"
        # print(history_info)
        # print("yes")
        lines = dic["lyrics"][struct].split('\n')
    
        lines = [line.strip() for line in lines]
        # print(f'lines {lines}')
        # print("--------------")
        data = []
        title = re.sub(r'"', '', dic["Title"])
        pair = dic["paired_lines"] 
        history = []
        invalid = 0
        for line in lines:
            for sentence in pair:
                if sentence["Original"] == line:
                    original = sentence["Original"]
                    rephrased = sentence["Rephrased1"]
                    rephrased2 = sentence["Rephrased2"]
                    # print(f'original: {original}')
                    # print(f'rephrased: {rephrased}')
            
            constraint = generate_constraint(original)
            if len(constraint) == 0 or not valid_line(original) or not valid_line(rephrased) or not valid_line(rephrased2):
                invalid += 1
                # print(f"didn't get the music constraint: {line}")
                continue
            if original == rephrased and original == rephrased2:
                invalid += 1
                 
                continue
            constraints_processed = []
            for i in range(len(constraint)):
                constraints_processed.append(f"S_{i}: /STRESSED/" if constraint[i] == 1 else f"S_{i}: /UNSTRESSED/")
            
            
            num_syllable_original, important_index_original = extract(original)
            num_syllable_rephrased, important_index_rephrased = extract(rephrased)
            num_syllable_rephrased2, important_index_rephrased2 = extract(rephrased2)
            if abs(num_syllable_original - num_syllable_rephrased) < abs(num_syllable_original - num_syllable_rephrased2):
                rephrased,rephrased2 = rephrased2,rephrased
                num_syllable_rephrased, num_syllable_rephrased2 = num_syllable_rephrased2,num_syllable_rephrased
            #     print("---------")
            #     print(f'original {num_syllable_original}, rephrased {num_syllable_rephrased}, rephrased2 {num_syllable_rephrased2}')
            # else:
            #     print(f'---original {num_syllable_original}, rephrased {num_syllable_rephrased}, rephrased2 {num_syllable_rephrased2}')
            original_words = original.split()
            rephrased_words = rephrased.split()
            rephrased_words2 = rephrased2.split()
            #make sure even negative examples improved some.
            data_original = []
            data_rephrased = []
            data_rephrased2 = []
            important_original = []
            important_rephrased = []
            num_orginal_word_stress = []
            num_rephrased_word_stress = []
            num_rephrased2_word_stress = []
            for i in original_words:
                data_original.append(get_word_with_phenom(i))
                num_orginal_word_stress.append(str(len(get_stress_clean(i))))
                if is_important_word(original,i):
                    important_original.append(i)
            for i in rephrased_words:
                data_rephrased.append(get_word_with_phenom(i))
                num_rephrased_word_stress.append(str(len(get_stress_clean(i))))
                if is_important_word(rephrased,i):
                    important_rephrased.append(i)
            for i in rephrased_words2:
                data_rephrased2.append(get_word_with_phenom(i))
                num_rephrased2_word_stress.append(str(len(get_stress_clean(i))))
            #add negative example
            need_negative = True if random.uniform(0,1) > 0.4 else False 
            # print(random.uniform(0,1) )
            if (num_syllable_original == num_syllable_rephrased2) or rephrased == rephrased2:
                # print("yes")
                need_negative = False
            num_syllable = len(constraint)
            data_rephrased = ' '.join(data_rephrased)
            Match = "match" if num_syllable == num_syllable_rephrased else "does not match" 
            history_info = ", ".join(history) if history !=[] else "No previously generated lyric"
            data_original = ' '.join(data_original) #/EH2 N L AY1 T AH0 N M AH0 N T/ S_1: Stressed, S_2 Unstressed,... [0,1,0]
            if num_syllable == num_syllable_rephrased:
                modify = "keep the number of syllables the same as in the original sentence"
            elif num_syllable_rephrased > num_syllable:
                modify = "rephrasing the original sentence so that generated lyrics have less syllables"
            else:
                modify = "rephrasing the original sentence so that generated lyrics have more syllables"
            CoT_input = f'The goal is to firstly, match the number of syllables in the music constraint, and secondly, \
    match the important word to the /STRESSED/ syllables.\
    The music constraint indicates that there should be {num_syllable} syllables in the generated lyrics. The original sentence has {num_syllable_rephrased} syllables. \
    Therefore, you should {modify}. \
    The important words in the original lyric is {important_rephrased}, and the syllables for each word is {data_rephrased}. \
    Therefore, we want to rephrase the sentence, so that 1, the number of syllables in the generated \
    lyric is {num_syllable} by {modify}, 2, the stress of each of the important word in the generated lyric matches with the music constraint, \
    and 3, it is fluent, singable, and coherent with the previously generated lyrics.'
            
            CoT_output = f"We want to generate a lyric with {num_syllable_original} syllables, and the generated lyric is '{original}'. The corresponding syllables for each word is {data_original}. It has {'+'.join(num_orginal_word_stress)} = {num_syllable_original} syllables \
    and matches with the total number of syllables in the music constraint ({num_syllable} syllables). \
    The important words in the generated lyric is {important_original}. \
    The position of the stressed syllables of these important words are {important_index_original}, \
    and {', '.join(['S_' + str(i) for i in important_index_original])} are all '/STRESSED/'. \
    The position of stressed syllable of important words in the generated lyric matches the music constraint."
            
            Input = f"Lyric that needed to be revised based on the music constraint: '{rephrased}'. Previously generated lyrics are: '{history_info }.' It is in {struct.split('_')[0]} section. \
    Title is '{title}'. \
    The music constraint: {' '.join(constraints_processed)}. {CoT_input}"
            Output = CoT_output
            data.append({"Input":Input, "Output":Output})
            # print(Output)
            history.append(original)
    # print(f'data: {data}')
    return data
# process_data({"Original": "Looking for some education", "Rephrased": "In search of knowledge and enlightenment", "Music_constraint": [1, 1, 0, 0, 0, 0, 1, 0]}) 
def process_lyrics_with_context(dic):#in_file_name,out_file_name
    # with open(in_file_name) as f:
    #     loaded_pairs = json.load(f)
    data = []
    title = re.sub(r'"', '', dic["Title"])
    pair = dic["paired_lines"] 
    history = []
    invalid = 0
    for line in pair:
        original = line["Original"]
        rephrased = line["Rephrased1"]
        rephrased2 = line["Rephrased2"]
        
        constraint = generate_constraint(original)
        if len(constraint) == 0 or not valid_line(original) or not valid_line(rephrased) or not valid_line(rephrased2):
            invalid += 1
            # print(f"didn't get the music constraint: {line}")
            continue
        if original == rephrased and original == rephrased2:
            invalid += 1
            continue
        constraints_processed = []
        for i in range(len(constraint)):
            constraints_processed.append(f"S_{i}: /STRESSED/" if constraint[i] == 1 else f"S_{i}: /UNSTRESSED/")
         
        
        num_syllable_original, important_index_original = extract(original)
        num_syllable_rephrased, important_index_rephrased = extract(rephrased)
        num_syllable_rephrased2, important_index_rephrased2 = extract(rephrased2)
        if abs(num_syllable_original - num_syllable_rephrased) < abs(num_syllable_original - num_syllable_rephrased2):
            rephrased,rephrased2 = rephrased2,rephrased
            num_syllable_rephrased, num_syllable_rephrased2 = num_syllable_rephrased2,num_syllable_rephrased
        #     print("---------")
        #     print(f'original {num_syllable_original}, rephrased {num_syllable_rephrased}, rephrased2 {num_syllable_rephrased2}')
        # else:
        #     print(f'---original {num_syllable_original}, rephrased {num_syllable_rephrased}, rephrased2 {num_syllable_rephrased2}')
        original_words = original.split()
        rephrased_words = rephrased.split()
        rephrased_words2 = rephrased2.split()
        #make sure even negative examples improved some.
        data_original = []
        data_rephrased = []
        data_rephrased2 = []
        important_original = []
        important_rephrased = []
        num_orginal_word_stress = []
        num_rephrased_word_stress = []
        num_rephrased2_word_stress = []
        for i in original_words:
            data_original.append(get_word_with_phenom(i))
            num_orginal_word_stress.append(str(len(get_stress_clean(i))))
            if is_important_word(original,i):
                important_original.append(i)
        for i in rephrased_words:
            data_rephrased.append(get_word_with_phenom(i))
            num_rephrased_word_stress.append(str(len(get_stress_clean(i))))
            if is_important_word(rephrased,i):
                important_rephrased.append(i)
        for i in rephrased_words2:
            data_rephrased2.append(get_word_with_phenom(i))
            num_rephrased2_word_stress.append(str(len(get_stress_clean(i))))
        #add negative example
        need_negative = True if random.uniform(0,1) > 1 else False 
        # print(random.uniform(0,1) )
        if (num_syllable_original == num_syllable_rephrased2) or rephrased == rephrased2:
            # print("yes")
            need_negative = False
        num_syllable = len(constraint)
        data_rephrased = ' '.join(data_rephrased)
        Match = "match" if num_syllable == num_syllable_rephrased else "does not match" 
        history_info = ", ".join(history) if history !=[] else "No previously generated lyric"
        data_original = ' '.join(data_original) #/EH2 N L AY1 T AH0 N M AH0 N T/ S_1: Stressed, S_2 Unstressed,... [0,1,0]
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
        
        CoT_output = f"We want to generate a lyric with {num_syllable_original} syllables, and the generated lyric is '{original}'. The corresponding syllables for each word is {data_original}. It has {'+'.join(num_orginal_word_stress)} = {num_syllable_original} syllables \
and matches with the total number of syllables in the music constraint ({num_syllable} syllables). \
The important words in the generated lyric is {important_original}.\
The position of the stressed syllables of these important words are {important_index_original}, \
and {', '.join(['S_' + str(i) for i in important_index_original])} are all '/STRESSED/'. \
The position of stressed syllable of important words in the generated lyric matches the music constraint."
        
        CoT_output_negative = f"We want to generate a lyric with {num_syllable_original} syllables, and generated lyric is '{rephrased2}'. The corresponding syllables for each word is {data_rephrased2}. It has {'+'.join(num_rephrased2_word_stress)} = {num_syllable_rephrased2} syllables \
and does not match with the total number of syllables in the music constraint ({num_syllable_original} syllables). Further improvement is needed. "

        Input = f"Lyric that needed to be revised based on the music constraint: '{rephrased}'. Previously generated lyrics are: '{history_info }.'\
Title is '{title}'. \
The music constraint: {' '.join(constraints_processed)}. {CoT_input}"
        Output = CoT_output if not need_negative else CoT_output_negative
        data.append({"Input":Input, "Output":Output})
        # print(Output)
        history.append(original)
    return data
# process_data({"Original": "Looking for some education", "Rephrased": "In search of knowledge and enlightenment", "Music_constraint": [1, 1, 0, 0, 0, 0, 1, 0]}) 
def process_fine_tune_data(in_file_name,out_file_name):
    with open(in_file_name) as f:
        songs = json.load(f)
    all_data = []
    print(f'there are {len(songs)} songs')
    # exit()
    for song in tqdm(songs, desc="Processing Songs"):
        try:
            cur_data = process_lyrics_with_song_structure_neg(song)
            for data in cur_data:
                # print(data)
                all_data.append(data)
            # print(all_data)
            # exit()
        except Exception as e:
            print(e)
            # exit()
            continue
        # exit()
    with open(out_file_name, 'w') as f:
        json.dump(all_data,f,indent=4)

def get_data(in_file_name,out_file_name):
    with open(in_file_name) as f:
        loaded_pairs = json.load(f)
    cleaned_Data = []
    valid = 0
    invalid = 0
    for pair in loaded_pairs:
        pair["Music_constraint"] = generate_constraint(pair["Original"])
        try:
            Input,Output = process_data(pair)
            cleaned_Data.append({"Input":Input, "Output": Output})
            valid += 1
        except:
            invalid += 1
            # print(pair)
            continue
        
    with open(out_file_name, 'w') as f:
        json.dump(cleaned_Data,f)
#/home/songyan/Real_M2L-main/llama/generated_fine_tuned.json
if __name__ == "__main__":
    process_fine_tune_data("/home/songyan/Real_M2L-main/data/data_finetune/chatgpt3.5/generated_line_w_structure.json","/home/songyan/Real_M2L-main/data/data_finetune/chatgpt3.5/generated_line_w_structure_neg.json")
# get_data("processed_data.json","finetune_data.json")
    # with open("/home/songyan/Real_M2L-main/data/data_finetune/generated_fine_tuned_scrape.json") as f:
    #     loaded_pairs = json.load(f)
    #     print(len(loaded_pairs))

# process_data("paired_lines.json","processed_data.json")
# paired_lines.json