from re import T
from music21 import converter, midi, stream, metadata,meter,note,interval
from nltk.sem.evaluate import _TUPLES_RE
# from inference.infer_lyrics import compile_melody
import numpy as np
# from inference.generation_utils import *
#change it to average interval jump.
import json
import os
from tqdm import tqdm
import glob
import random
puncts = ['.', ',', "?", "!", ';']
def detect_punc(token):
    for p in puncts:
        if p in token:
            return True
    return False
def get_avg_interval_jump(s):
    notes = [n for n in s.recurse() if isinstance(n, note.Note)]

    # Calculate absolute intervals between consecutive notes
    intervals = [abs(interval.notesToChromatic(notes[i], notes[i+1]).semitones)
                for i in range(len(notes) - 1)]

    # Compute the average interval size
    average_interval = sum(intervals) / len(intervals)
    return average_interval
def get_avg_duration(s):
    all_duration = []
    notes = s.recurse().notes
    for i,note in enumerate(notes):
        try:
            note.duration
            note.pitch
        except: #when the note is a rest, a chord progression etc.
            continue
        all_duration.append(note.duration.quarterLength)
    avg_d = np.mean(all_duration)
    return avg_d

def get_vocal(s,output_file):
    # if s is None:
    #     print("WARNING! S IS NONE")
    # print(s.parts)
    for part in s.parts:
        n = part.recurse().notes
        counter = 0
        for i,note in enumerate(n):
            if note.lyric:
                counter += 1
            if counter > 15:
                return part
    # new_score = s.parts[0]
    return None
# new_score = get_vocal(s,"")
#use this result to go to chatGPT and check.
def contain_number(lyrics):
    import re
    pattern = r"\d"
    if re.search(pattern, lyrics):
        return True
    else:
        return False
def get_lyric(s):
    new_s = get_vocal(s,"")
    all_l = []
    l = []
    for n in new_s.flat.notes:
        if n.lyrics:
            # print(n.lyrics[0].text)
            if contain_number(n.lyric):
                import re
                cleaned = n.lyrics[0].text
                cleaned = re.sub(r'\d+', '', cleaned)
                cleaned_sentence = re.sub(r'[^\w\s]', '', cleaned)
                # cleaned_sentence = cleaned.split()[0]
                if len(cleaned_sentence) == 0:
                    continue
                l.append(cleaned_sentence)
                if detect_punc(cleaned):
                    all_l.append(l)
                    l = []
            else:
                # print(n.lyric.split()[0])
                l.append(n.lyrics[0].text)
                if detect_punc(n.lyric):
                    all_l.append(l)
                    l = []
    if len(l) > 0:
        all_l.append(l)
    return all_l

# print(get_lyric(new_score))
# print(get_lyric(new_score))
# time_signatures = first_part.recurse().getElementsByClass(meter.TimeSignature)
def get_stress(time_signature):
    # print(time_signature)
    if str(time_signature) == "4/4" or str(time_signature) == "2/2":
        return [0,2],1
    if str(time_signature) == "3/4" or time_signature == "2/4" or time_signature == "3/8":
        return [0],1
    if str(time_signature) == "6/8":
        return [0,3],0.5
    if str(time_signature) == "9/8":
        return [0,3,6], 0.5
    if str(time_signature) == "12/8":
        return [0,3,6,9],0.5

def meter_stressed_note(note,time_signature):
    stressed,beat_duration = get_stress(time_signature)
    # print(note.offset/beat_duration)
    # print(stressed)
    if note.offset/beat_duration in stressed and not note.isRest:
        return True
    else:
        return False
puncts = ['.', ',', "?", "!", ';',':']

def detect_punc(token):
    for p in puncts:
        if p in token:
            return True
    return False

def detect_part(s):
    return s
def is_syncopation(current,previous):
    if previous is None:
        # print("previous is None")
        return False
    # print("previous.duration")
    # print(previous.duration)
    # print("current.duration.quarterLength")
    # print(current.duration.quarterLength)
    if previous.duration.quarterLength < current.duration.quarterLength:
        return True
    else:
        return False
def is_real_syncopation(current,previous,is_meter):
    if previous is None:
        return False
    if previous.duration.quarterLength < current.duration.quarterLength and not is_meter:
        return True
    else:
        return False
def print_note(s):
    notes = s.recurse().notes
    for n in notes:
        # if n.pitch:
        try:
            print(n.pitch)
            print(f'len(n.lyrics) {n.lyrics}')
            if n.lyrics[0].number == 2:
                pass
            else:
                
                print(n.lyrics[0].text)
        except Exception as e:
            print(e)
            print(f'not able to print {n}')
def prepare_input_for_song_mass(s):
    notes = s.recurse().notes
    all_note_tuple = []
    note_tuple = []
    error = []
    all_lyrics = []
    lyrics = []
    for i,n in enumerate(notes):
        try:
            n.duration
            n.pitch
            n.lyric
            note_tuple.append((n.pitch.ps,float(n.duration.quarterLength),0))
            lyrics.append(n.lyric)
            if detect_punc(n.lyric):
                all_note_tuple.append(note_tuple)
                note_tuple = []
                all_lyrics.append(lyrics)
                lyrics = []
        except Exception as e:
            if e not in error:
                error.append(e)
            if isinstance(note, note.Rest):
                if note_tuple == 0:
                    continue
                else:
                    note_tuple[-1][2] = n.duration.quarterLength
            continue  
    # print(error)  
    all_note_tuple.append(note_tuple)   
    all_lyrics.append(lyrics)      
    return all_note_tuple,all_lyrics
def song_mass_process(dir,out_f):
    files = get_files(dir,post = '.mxl')
    ret = []
    for f in files:
        # print(f)
        s = converter.parse(f)
        score,lyrics = prepare_input_for_song_mass(s)
        ret.append({"score_name":f.split('/')[-1],'score':score,"lyrics":lyrics})
    with open(out_f,'w') as f:
        json.dump(ret,f,indent = 4)
def get_important_note_and_duration_copy(s,log = False):
    # is_tied = False
    # tied_duration = 0
    # previous_pitch = None
    JUMP_INTERVAL = get_avg_interval_jump(s) + 1
    all_patterns = []
    all_duration = []
    duration = []
    pattern = []
    fake_all_patterns = []
    #fake pattern is the baseline, which only use duration as indicator
    fake_pattern = []
    new_s = get_vocal(s,"")
    new_s.show()
    avg_duration = get_avg_duration(new_s)
    # part_name = new_s.partName or "Unnamed Part"
    # print(part_name)
    measures = new_s.getElementsByClass('Measure')
    # print(s.parts[0].getElementsByClass('Measure'))
    #in case it does not have a tie, but indead a tie
    fake_tie_flag = False
    previous = None
    cur_duration = 0
    for measure in measures:
    
        time_signature = measure.timeSignature or new_s.getTimeSignatures()[0]
        if time_signature:
            for n in measure.notesAndRests:
                cur_lyric = ""
                if n.lyric is not None and len(n.lyrics) != 0:
                    if n.lyrics[0].number == 1:
                        cur_lyric = n.lyrics[0].text
                if n.isNote and n.tie and (n.tie.type == "stop" or n.tie.type == "continue") :
                    # if previous is not None:
                    #     previous.duration.quarterLength += n.duration.quarterLength
                    if (n.tie and (n.tie.type == "stop" or n.tie.type == "continue")) or n.lyric is None or len(n.lyrics[0].text) == 0:
                        if n.lyric:
                            if detect_punc(n.lyrics[0].text):
                                all_patterns.append(pattern)
                                all_duration.append(duration)
                                duration = []
                                pattern = []
                                if log:
                                    print("detected punctuation!")
                        continue
                if n.isRest or not isinstance(n, note.Note):

                    if n.isRest and len(all_patterns) > 0 and len(all_patterns[-1]) != 0 and n.duration.quarterLength >= max(avg_duration,1):

                        all_patterns.append(pattern)
                        all_duration.append(duration)
                        duration = []
                        pattern = []
                        if log:
                            print(f" max(avg_duration,1) {max(avg_duration,1)}")
                            print("detected long rest, new phrase!")
                    if n.isRest and len(pattern) != 0:
                        duration.append(-1 * float(n.duration.quarterLength))
                        duration
                    continue
                if n.lyric is None or len(n.lyrics[0].text) == 0:
                    fake_tie_flag = True
                    if log:
                        print(f"note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {lyric}")
                        print(f"start fake tie, previously was {cur_duration}, now it is {cur_duration + n.duration.quarterLength}")
                    previous.duration.quarterLength += n.duration.quarterLength
                    
                    cur_duration += n.duration.quarterLength
                    continue 
                #fake tie end, reset fake tie flag
                if fake_tie_flag == True:
                    if len(pattern) == 0 and len(all_patterns) >= 2:
                        all_patterns[-2][-1] = 1
                    else:
                        if len(all_patterns) >= 2:
                            pattern[-1] = 1
                    fake_tie_flag = False
                if n.duration.quarterLength > avg_duration:
                    fake_pattern.append(1)
                else:
                    fake_pattern.append(0)
                if meter_stressed_note(n,time_signature.ratioString) or is_jump(n,previous,JUMP_INTERVAL) or is_real_syncopation(n,previous,float(n.offset/get_stress(time_signature.ratioString)[1]).is_integer()):
                    pattern.append(1) 
                    if log: 
                        if n.isNote and n.tie:
                            print("it is a tie!")
                            print(n.tie.type)
                        if meter_stressed_note(n,time_signature.ratioString):
                            print("meter_stressed_note")
                        if is_jump(n,previous,JUMP_INTERVAL):
                            print("is_Jump")
                        if (n.tie and n.tie.type == "start"):
                            print("tie")
                        if is_syncopation(n,previous) and not float(n.offset/get_stress(time_signature.ratioString)[1]).is_integer():
                            print(float(n.offset/get_stress(time_signature.ratioString)[1]))
                            print("syncopation")      
                        lyric = (n.lyrics[0].text).split('\n')[0]                 
                        print(f"Stressed note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {lyric}")
                    
                else:
                    pattern.append(0)
                    if log:
                        print(f"Weaked note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {n.lyric}")
                # if is_jump(n,previous): print("Jump!")
                cur_duration += n.duration.quarterLength
                if log:
                    print(f"duration added: {cur_duration}")
                    print("-------------")

                duration.append(cur_duration)
                cur_duration = 0

                previous = n
                if n.lyric is not None:
                    # print(n.lyric)
                    if detect_punc(n.lyric):
                        # print("measure")
                        # print(measure.number)
                        # print(n.pitch)
                        all_patterns.append(pattern)
                        pattern = []
                        fake_all_patterns.append(fake_pattern)
                        fake_pattern = []
                        all_duration.append(duration)
                        duration = []
                

    if len(pattern) > 0:
        all_patterns.append(pattern)
    if len(fake_pattern) > 0:
        fake_all_patterns.append(fake_pattern)
    if len(duration) > 0:
        all_duration.append(duration)
    # print(fake_all_patterns)
    return all_patterns,all_duration
"""
params:
    s: s = converter.parse(file.mxl)
output:
    all patterns: list of list of int. Each list represents a musical phrase. 1 represents important note, 0 represents unimportant note
    all durations: List of list of int. Each list represents a musical phrase. Value represents note/rest length in quaternote. Negative representes rest.
"""
def get_important_note_and_duration(s,log = False):
    JUMP_INTERVAL = get_avg_interval_jump(s) + 1
    all_patterns = []
    all_duration = []
    duration = []
    pattern = []
    fake_all_patterns = []
    #fake pattern is the baseline, which only use duration as indicator
    fake_pattern = []
    new_s = get_vocal(s,"")
    
    avg_duration = get_avg_duration(new_s)

    measures = new_s.getElementsByClass('Measure')
    #in case it does not have a tie, but indead a tie
    fake_tie_flag = False
    previous = None
    cur_duration = 0
    for measure in measures:
        time_signature = measure.timeSignature or new_s.getTimeSignatures()[0]
        if time_signature:
            for n in measure.notesAndRests:
                cur_lyric = ""
                if n.lyric is not None and len(n.lyrics) != 0:
                    if n.lyrics[0].number == 1:
                        cur_lyric = n.lyrics[0].text
               
                if n.isRest or not isinstance(n, note.Note):
                    if (n.isRest and n.duration.quarterLength >= max(1,avg_duration) and pattern != []):
                        all_patterns.append(pattern)
                        all_duration.append(duration)
                        duration = []
                        pattern = []
                        if log:
                            print(f'max(1,avg_duration) {max(1,avg_duration)}, n.duration.quarterLength  {n.duration.quarterLength }')
                            print("detected long rest!")
                    if n.isRest and len(pattern) > 0:
                        duration.append(-1*float(n.duration.quarterLength))
                    continue
                #if there is no lyric, then deem it as a tie
                if cur_lyric == "":
                    #song not started, just the instrumental intro
                    if previous is None:
                        continue
                    fake_tie_flag = True
                    if log:
                        print(f"note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {cur_lyric}")
                        print(f"start fake tie, previously was {cur_duration}, now it is {cur_duration + n.duration.quarterLength}")
                    previous.duration.quarterLength += n.duration.quarterLength
                    continue 
                #fake tie end, reset fake tie flag, change the value of previous pattern as well as duration
                if fake_tie_flag == True:
                    
                    if log:
                        print(f"fake tie ended, appended previous {previous.duration.quarterLength}")
                    if len(pattern) == 0 and len(all_patterns) >= 2:
                        if len(all_duration[-2]) >= 2 and previous.duration.quarterLength > all_duration[-2][-2]:
                            all_patterns[-1][-1] = 1
                        all_duration[-1][-1] = previous.duration.quarterLength
                    else:
                        if len(pattern) >= 1:
                            if len(duration) >= 2 and previous.duration.quarterLength > duration[-2]:
                                pattern[-1] = 1
                            duration[-1] = previous.duration.quarterLength
                    cur_duration = 0
                    fake_tie_flag = False
                if n.duration.quarterLength > avg_duration:
                    fake_pattern.append(1)
                else:
                    fake_pattern.append(0)
                if meter_stressed_note(n,time_signature.ratioString) or is_jump(n,previous,JUMP_INTERVAL) or (n.tie and n.tie.type == "start") or is_real_syncopation(n,previous,float(n.offset/get_stress(time_signature.ratioString)[1]).is_integer()):
                    pattern.append(1) 
                    lyric = (n.lyrics[0].text).split('\n')[0]       
                    if log: 
                        if n.isNote and n.tie:
                            print("it is a tie!")
                            print(n.tie.type)
                        if meter_stressed_note(n,time_signature.ratioString):
                            print("meter_stressed_note")
                        if is_jump(n,previous,JUMP_INTERVAL):
                            print("is_Jump")
                        if (n.tie and n.tie.type == "start"):
                            print("tie")
                        if is_syncopation(n,previous):
                            print("syncopation")                
                        print(f"Stressed note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {cur_lyric}")
                else:
                    pattern.append(0)
                    if log:
                        print(f"Weaked note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {cur_lyric}")
                # if is_jump(n,previous): print("Jump!")
                if log:
                    print(f"duration added: {n.duration.quarterLength}")
                    print("-------------")

                duration.append(n.duration.quarterLength)
                cur_duration = 0

                previous = n
                if n.lyric is not None:
                    # print(n.lyric)
                    if len(n.lyrics) != 0:
                        if n.lyrics[0].number == 1:
                            
                            if detect_punc(n.lyrics[0].text):
                                all_patterns.append(pattern)
                                pattern = []
                                fake_all_patterns.append(fake_pattern)
                                fake_pattern = []
                                all_duration.append(duration)
                                duration = []
    if len(pattern) > 0:
        all_patterns.append(pattern)
    if len(fake_pattern) > 0:
        fake_all_patterns.append(fake_pattern)
    if len(duration) > 0:
        all_duration.append(duration)
    duration = []
    for i in all_duration:
        duration.append([float(j) for j in i])
    return all_patterns,duration
def get_important_note_and_duration_with_lyrics(s,log = False,for_eval=False):
    # is_tied = False
    # tied_duration = 0
    # previous_pitch = None
    JUMP_INTERVAL = get_avg_interval_jump(s) + 1
    all_patterns = []
    all_duration = []
    duration = []
    pattern = []
    all_lyrics = []
    lyrics = []
    fake_all_patterns = []
    #fake pattern is the baseline, which only use duration as indicator
    fake_pattern = []
    new_s = get_vocal(s,"")
    
    avg_duration = get_avg_duration(new_s)
    # part_name = new_s.partName or "Unnamed Part"
    # print(part_name)
    measures = new_s.getElementsByClass('Measure')
    # print(s.parts[0].getElementsByClass('Measure'))
    #in case it does not have a tie, but indead a tie
    fake_tie_flag = False
    previous = None
    cur_duration = 0
    for measure in measures:
        # print('yes')
        time_signature = measure.timeSignature or new_s.getTimeSignatures()[0]
        # print(time_signature)
        if time_signature:
            for n in measure.notesAndRests:
                cur_lyric = ""
                if n.lyric is not None and len(n.lyrics) != 0:
                    if n.lyrics[0].number == 1:
                        cur_lyric = n.lyrics[0].text
                    if for_eval:
                        cur_lyric = str(n.lyric.strip('\n'))
                        # print(cur_lyric)
               
                if n.isRest or not isinstance(n, note.Note):
                    if (n.isRest and n.duration.quarterLength >= max(1,avg_duration) and pattern != []):
                        all_patterns.append(pattern)
                        all_duration.append(duration)
                        duration = []
                        pattern = []
                        if log:
                            print(f'max(1,avg_duration) {max(1,avg_duration)}, n.duration.quarterLength  {n.duration.quarterLength }')
                            print("detected long rest!")
                    if n.isRest and len(pattern) > 0:
                        duration.append(-1*float(n.duration.quarterLength))
                    continue
                #if there is no lyric, then deem it as a tie
                # if n.lyric is None or len(n.lyrics[0].text) == 0:
                if cur_lyric == "":
                    # print('here')
                    
                    #song not started, just the instrumental intro
                    if previous is None:
                        continue
                    fake_tie_flag = True
                    if log:
                        print(f"note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {cur_lyric}")
                        print(f"start fake tie, previously was {cur_duration}, now it is {cur_duration + n.duration.quarterLength}")
                    previous.duration.quarterLength += n.duration.quarterLength
                    continue 
                #fake tie end, reset fake tie flag, change the value of previous pattern as well as duration
                if fake_tie_flag == True:
                    
                    if log:
                        print(f"fake tie ended, appended previous {previous.duration.quarterLength}")
                    if len(pattern) == 0 and len(all_patterns) >= 2:
                        if len(all_duration[-2]) >= 2 and previous.duration.quarterLength > all_duration[-2][-2]:
                            all_patterns[-1][-1] = 1
                        all_duration[-1][-1] = previous.duration.quarterLength
                        # print(all_duration)
                    else:
                        if len(pattern) >= 1:
                            if len(duration) >= 2 and previous.duration.quarterLength > duration[-2]:
                                pattern[-1] = 1
                            duration[-1] = previous.duration.quarterLength
                    cur_duration = 0
                    fake_tie_flag = False
                if n.duration.quarterLength > avg_duration:
                    fake_pattern.append(1)
                else:
                    fake_pattern.append(0)
                if n.lyrics[0].number == 1:
                    cur_lyric = n.lyrics[0].text
                    if not for_eval:
                        lyrics.append(cur_lyric)
                if for_eval:
                    # print('here')
                    counter = 0
                    for i in range(len(n.lyrics)):
                        if n.lyrics[i].text.strip() != '':
                            counter = i
                            break
                    lyrics.append((n.lyrics[counter].text,n.lyrics[counter].syllabic))
                if meter_stressed_note(n,time_signature.ratioString) or is_jump(n,previous,JUMP_INTERVAL) or (n.tie and n.tie.type == "start") or is_real_syncopation(n,previous,float(n.offset/get_stress(time_signature.ratioString)[1]).is_integer()):
                    pattern.append(1) 
                    
                    lyric = (n.lyrics[0].text).split('\n')[0]       
                    if log: 
                        if n.isNote and n.tie:
                            print("it is a tie!")
                            print(n.tie.type)
                        if meter_stressed_note(n,time_signature.ratioString):
                            print("meter_stressed_note")
                        if is_jump(n,previous,JUMP_INTERVAL):
                            print("is_Jump")
                        if (n.tie and n.tie.type == "start"):
                            print("tie")
                        if is_syncopation(n,previous):
                            print("syncopation")                
                        print(f"Stressed note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {cur_lyric}")
                else:
                    pattern.append(0)
                    if log:
                        print(f"Weaked note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number} with lyric: {cur_lyric}")
                # if is_jump(n,previous): print("Jump!")
                if log:
                    print(f"duration added: {n.duration.quarterLength}")
                    print("-------------")

                duration.append(n.duration.quarterLength)
                cur_duration = 0

                previous = n
                if n.lyric is not None:
                    # print(n.lyric)
                    if len(n.lyrics) != 0:
                        if not for_eval:
                            if n.lyrics[0].number == 1:
                                if detect_punc(n.lyrics[0].text):
                                    all_patterns.append(pattern)
                                    pattern = []
                                    fake_all_patterns.append(fake_pattern)
                                    fake_pattern = []
                                    all_duration.append(duration)
                                    duration = []
                                    all_lyrics.append(lyrics)
                                    lyrics = []
                        else:
                            if detect_punc(n.lyric.strip('\n')):
                                all_patterns.append(pattern)
                                pattern = []
                                fake_all_patterns.append(fake_pattern)
                                fake_pattern = []
                                all_duration.append(duration)
                                duration = []
                                all_lyrics.append(lyrics)
                                lyrics = []
    if len(pattern) > 0:
        all_patterns.append(pattern)
    if len(fake_pattern) > 0:
        fake_all_patterns.append(fake_pattern)
    if len(duration) > 0:
        all_duration.append(duration)
    if len(lyrics) > 0:
        all_lyrics.append(lyrics)
    # print(fake_all_patterns)
    duration = []
    # duration = []
    for i in all_duration:
        duration.append([float(j) for j in i])
    # print(duration)
    if not for_eval:
        all_lyrics = [''.join(i) for i in all_lyrics]
    return all_patterns,duration,all_lyrics

def get_all_info_for_gpt(s,with_lyrics = False,num_lines = 10):
    new_s = get_vocal(s,"")
    try:
        measures = new_s.getElementsByClass('Measure')
    except:
        return None, None
    print_time = False
    counter = 0
    num_sentence = 0
    Finished = False
    lyric_l = []
    ret_sentence = ""
    for measure in measures:
        if Finished:
            break
        time_signature = measure.timeSignature or new_s.getTimeSignatures()[0]
        if print_time == False:
            # print(f"time_signature is {time_signature }")
            print_time = True
            ret_sentence += f"time_signature is {time_signature }" + "\n"
        if time_signature:
            if len(measure.notesAndRests) == 0:
                return None,None
            for n in measure.notesAndRests:
                # print(n.lyrics)
                cur_lyrics = n.lyrics[0].text if n.lyrics != [] and n.lyrics[0].number == 1 else ''
                if detect_punc(cur_lyrics):
                    num_sentence += 1
                if not with_lyrics:
                    ret_sentence += f"Measure: {measure.number} {counter}'th note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} duration {n.duration.quarterLength}" + "\n"
                    # print(f"Measure: {measure.number} {counter}'th note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} duration {n.duration.quarterLength}")
                else:
                    lyric_l.append(n.lyrics[0].text if n.lyrics != [] else '')
                counter += 1
                if num_sentence > num_lines:
                    Finished = True
                    if with_lyrics:
                        # print(f"There are {len(lyric_l)} notes, so the lyric list length is {len(lyric_l)}, each entry of the list represents the lyric of the note of its index")
                        ret_sentence += lyric_l
                        # print(lyric_l)
                    break
            
    return ret_sentence,num_sentence

def is_jump(current, previous,JUMP_INTERVAL):
    if previous is None or not current.isNote or not previous.isNote: return False
    return current.pitch.ps - previous.pitch.ps >= JUMP_INTERVAL

'''
Heuristic:
Phrase > measure > beat
top k for the desired key word --> same meaning, better corresponds the melody.

word importance -- i am crazy?
'''

def importance_analysis(s):
    all_patterns = []
    pattern = []
    for measure in s.getElementsByClass('Measure'):
        time_signature = measure.timeSignature or s.getTimeSignatures()[0]
        previous = None
        if time_signature:
            for n in measure.notesAndRests:
                if n.isNote and n.tie and (n.tie.type == 'stop' or n.tie.type == 'continue'):
                    continue
                if n.isRest:
                    continue
                if meter_stressed_note(n,time_signature.ratioString) or is_jump(n,previous) or (n.tie and n.tie.type == "start") or is_syncopation(n,previous):
                    pattern.append(1)
                    print(f"Stressed note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number}")
                else:
                    pattern.append(0)
                    print(f"Weaked note: {n.nameWithOctave if isinstance(n, note.Note) else 'Rest'} at offset {n.offset} in Measure: {measure.number}")
                # if is_jump(n,previous): print("Jump!")
                previous = n
                if n.lyric is not None:
                    # print(n.lyric)
                    if detect_punc(n.lyric):
                        all_patterns.append(pattern)
                        pattern = []
    return all_patterns

def compare_notes_in_words(note_list,lyric_list,ref,log):
    total = 0
    correct = 0
    for i in range(len(note_list)):
        n_l = note_list[i]
        for j in range(len(n_l)):
            if n_l[j] == 1:
                total += 1
                if lyric_list[i][j] == 1:
                    correct += 1
                else:
                    if log:
                        print("i,j = ",i,j)
                        print(lyric_list[i])
                        print(ref[i][j])
    return total, correct/total

#note_list is lyric_list, mistake here
def compare_words_in_notes(note_list,lyric_list,ref,log):
    total = 0
    correct = 0
    for i in range(len(note_list)):
        n_l = note_list[i]
        for j in range(len(n_l)):
            # if log:
            #     print("i,j = ",i,j)
            #     print(ref[i][j])
            if lyric_list[i][j]  == 1:
                total += 1
                if n_l[j]== 1:
                    correct += 1
                else:
                    if log:
                        print("i,j = ",i,j)
                        print(ref[i][j])
    return total, correct/total
def get_important_lyrics(l):
    l_all = []
    s = []
    for sentence in l:
        for w in sentence:
            if '0' in w:
                s.append(0)
            if '1' in w:
                s.append(1)
        l_all.append(s)
        s = []
    return l_all
def get_files(directory,post = ""):
    file_paths = []
    for root, directories, files in os.walk(directory):
        # print(files)
        for filename in files:
            file_path = os.path.join(root, filename)
            if post != "":
                if not file_path.endswith(post):
                    continue
            
            file_paths.append(file_path)    
    return file_paths
def get_annotated_xml(pdf_dir,source_dir):
    base_dir = os.path.abspath(source_dir)
    base_dir = os.path.dirname(base_dir)
    pdf_files = get_files(pdf_dir)
    mxl_files = get_files(source_dir)
    pdf_files = list(filter(lambda x: x.endswith('.pdf'), pdf_files))
    mxl_files = list(filter(lambda x: x.endswith('.mxl'), mxl_files))
    
    ret = []
    for file in pdf_files:
        for i in mxl_files:
            if file.split("/")[-1].split(".")[0] == i.split("/")[-1].split(".")[0]:
                ret.append(i)
        else:
            print(f"didn't find {file}")
    print(f'there are {len(ret)} extracted songs')
    return ret

def important_note_from_file(pdf_dir,source_dir,out_f):
    files = get_annotated_xml(pdf_dir,source_dir)
    ret = []
    for file in files:
        important_note = get_important_note(converter.parse(file))
        ret.append({"name":file.split("/")[-1],"extracted":important_note,"annotated":""})
    with open(out_f,'w') as f:
        json.dump(ret,f,indent=4)
def get_lyric_draft(draft,num_sentences):
    potential = []
    for data in draft:
        if data["num_line"] >= num_sentences:
            potential.append(data)
    res = random.choice(potential)
    res = res["draft"]
    res = res.split("\n")[:num_sentences]
    return "\n".join(res)
    
def prepare_gpt_baseline(base_dir,draft_f,out_f):
# Loop through every file ending with .xml in the base directory and its subdirectories
    ret_l = []
    with open(draft_f) as f:
        lyric_draft = json.load(f)
    mxl_files = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        # Find all files with .mxl extension
        for filename in filenames:
            if filename.lower().endswith(".mxl"):
                # Get the full path of the .mxl file
                mxl_files.append(os.path.join(dirpath, filename))
    print(f'there are total {len(mxl_files)} of files')
    for f in mxl_files:
        # print(f)
        s = converter.parse(f)
        score,num_sentence = get_all_info_for_gpt(s)
        if score is not None:
            draft = get_lyric_draft(lyric_draft,num_sentence)
            ret_l.append({"score":score,"file":f,"draft":draft})
    with open(out_f,'w') as f:
        json.dump(ret_l,f,indent = 4)
def calculate_eval(eval_score_dir,annotated_lyric_dir):
    
    n_song = 0
    sum_note_in_word = 0
    sum_word_in_note = 0
    # Read the JSON file
    with open(annotated_lyric_dir, 'r') as file:
        data = json.load(file)
    
    for song in data:
        
        score_filename = song["score_filename"]

        # print(score_filename)
        annotated_lyric = song["annotated_lyric"]
        f = eval_score_dir + score_filename
        s = converter.parse(f)
        important_note_list = get_important_note(s)
        important_word_list = get_important_lyrics(annotated_lyric)
        # print(annotated_lyric)
        # print(important_note_list)
        # print("important notes that is noun/verb")
        note_in_word = compare_words_in_notes(important_word_list,important_note_list,annotated_lyric,False)
        # print(note_in_word)
        # print("noun/verb that is important notes")
        word_in_note = compare_notes_in_words(important_word_list,important_note_list,annotated_lyric,False)
        print(word_in_note[1])
        n_song += 1
        sum_note_in_word += note_in_word[1]
        sum_word_in_note += word_in_note[1]
    avg_note_in_word = sum_note_in_word / n_song
    avg_word_in_note = sum_word_in_note / n_song

    print(f"there are {n_song} songs.")
    # print(f"average important notes that is noun/verb is {avg_note_in_word}")
    print(f"average noun/verb that is important notes is {avg_word_in_note}")
def prepare_input_in_batch(source_dir,out_f):
    f_ps = get_files(source_dir,'.mxl')

    ret = []
    for f in f_ps:
        s = converter.parse(f)
        important_note, duration = get_important_note_and_duration(s)
        du = []
        baseline = []
        for i in duration:
            du.append([float(j) for j in i])
            
        ret.append({"generated_score":important_note,"annotated_score":[],"duration":du,"original_file": f})
        # ret.append({"constraints_extractor":important_note,"constraints_expert":[],"duration":du,"original_file": f})
    with open(out_f,'w') as f:
        json.dump(ret,f,indent=4)

def process_translation(base_dir,out_f):
    fs = get_files(base_dir,'.mxl')
    for f in fs:
        s = converter.parse(f)
        constraint,duration,lyrics = get_important_note_and_duration_with_lyrics(s)
        print(lyrics)