import os

class EmotionVocab():
    def __init__(self):
        self.label = ''  
        self.words_id = [] 
        self.times = 0 

class EmotionWord():
    def __init__(self):
        self.context = '' 
        self.label_id = 0 
        self.times = 0
       

def get_Emotion(emotion_path):

    Emotion_list = []
    Words_list = []
    main_path = emotion_path
    dir = os.listdir(main_path) 
    dir.sort()
    for filename in dir:
        emotion = EmotionVocab()
        emotion.label = filename.split('.')[0]
        open_path = os.path.join(main_path, filename)
        f = open(open_path, 'r')
        for line in f.readlines():
            word = line.split()
            if word==[]: continue
            emotion_word = EmotionWord()
            emotion_word.label_id = len(Emotion_list)
            emotion_word.context = word
            emotion.words_id.append(len(Words_list))
            Words_list.append(emotion_word)
        Emotion_list.append(emotion)

    return Emotion_list, Words_list

