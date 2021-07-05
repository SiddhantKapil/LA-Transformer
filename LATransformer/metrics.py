def rank1(label, output):
    if label==output[1][0][0]:
        return True
    return False

def rank5(label, output):
    if label in output[1][0][:5]:
        return True
    return False

def rank10(label, output):
    if label in output[1][0][:10]:
        return True
    return False

def calc_map(label, output):
    count = 0
    score = 0
    good = 0
    for out in output[1][0]:
        count += 1
        if out==label:
            good += 1            
            score += (good/count)
    if good==0:
        return 0
    return score/good