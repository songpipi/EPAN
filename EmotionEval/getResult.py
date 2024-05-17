import io
from EmotionEval.getEmationVocab import get_Emotion
from EmotionEval.getRefEmotion import getRef
import csv

import math
import pandas as pd
import numpy as np


def clean(str):
    from getEmationVocab import is_letter
    string = ''
    for char in str:
        if is_letter(char):
            string += char
    return string

def get_result(pred_dict, emotion_path, ref_dict):

    Emotion_list, Words_list  = get_Emotion(emotion_path)
    result = []
    have_score = 0
    right_score = 0
    wrong_score = 0
    counter = 0
    contain_one_right = 0
    contain_right_and_wrong = 0
    Ref = getRef(Emotion_list, Words_list, ref_dict)


    vids = pred_dict.keys()
    for vid in vids:
        dict = {}
        dict['vid_id'] = vid
        dict['caption right emotion'] = []
        dict['caption wrong emotion'] = []  
        dict['Reference emotion'] = Ref[vid]
        Ref_emotion = Ref[vid]
        caption = pred_dict[vid]
        if len(caption)==1: caption = caption[0]
        tokens = caption.split()
        have_emotion = False
        contain_right = False
        contain_wrong = False
        counter += 1

        for token in tokens:
            for j in range(len(Words_list)):
                if token == Words_list[j].context[0]:
                    # ipdb.set_trace()
                    have_emotion = True

                    label = Emotion_list[Words_list[j].label_id].label
                    Words_list[j].times += 1
                    Emotion_list[Words_list[j].label_id].times += 1

                    if label in Ref_emotion:
                        contain_right = True
                        if label not in dict['caption right emotion']:
                            dict['caption right emotion'].append(label)
                            right_score += 1
                    
                    else:
                        contain_wrong = True
                        if label not in dict['caption wrong emotion']:
                            dict['caption wrong emotion'].append(label)
                            wrong_score += 1
                    break

        if have_emotion:
            have_score += 1
            dict['have emotion'] = True
            if contain_right:
                contain_one_right += 1
                if contain_wrong:
                    contain_right_and_wrong += 1
        else:
            dict['have emotion'] = False
        result.append(dict)

    N_r = right_score
    N_w = wrong_score
    phi = counter - have_score
    Acc_sw =  N_r / (N_r + N_w + phi) 
    
    N_r_1= contain_one_right
    N_h = contain_right_and_wrong
    alpha = 1e-4
    sp = math.exp(N_h/(N_h - N_r_1 + alpha))
    Acc_c = sp * (N_r_1/ counter)

    return Acc_sw, Acc_c
