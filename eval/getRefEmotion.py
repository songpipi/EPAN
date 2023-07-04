
def getRef(Emotion_list, Words_list, ref_dict):
    vids = ref_dict.keys()
    Ref = {}   
    for vid in vids:
        em_list = []     
        GTs = ref_dict[vid]
        for GT in GTs:

            tokens = GT.split()
            for token in tokens:
                for w in Words_list:
                    if token == w.context[0]:
                        if Emotion_list[w.label_id].label not in em_list:
                            em_list.append(Emotion_list[w.label_id].label)
        Ref[vid]=em_list
    return Ref

