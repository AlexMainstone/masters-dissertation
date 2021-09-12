import json
import operator
import numpy as np
import time
import random

from scipy import optimize
from train_statistical import load_corpus
from ignite.metrics.nlp import Bleu
from scipy.optimize import minimize


class Node:
    phrase = ""
    start = 0
    end = 0
    coverage = []
    total_score = 0
    reorder_total = 0
    score = 1
    ngram_prob = 1
    probability = 1

def compute_ngram_prob(ngram, src, penalty = 0.1):
    words = src.split(" ")
    if len(words) < 1:
        return 1
    words = ["&start;", "&start;"] + words
    prob = 0
    for i in range(2, len(words)):
        if not words[i-2] in ngram:
            prob*=penalty
            continue
        if not words[i-1] in ngram[words[i-2]]:
            prob*=penalty
            continue
        if not words[i] in ngram[words[i-2]][words[i-1]]:
            prob*=penalty
            continue
        prob += ngram[words[i-2]][words[i-1]][words[i]]
    return prob

def prune_stacks(stacks): # histogram pruning
    start = time.time()
    for s in stacks:
        # While there are too many nodes in a stack, remove the node from the stack
        while len(s) > 100:
            worstidx = 0
            worstval = 1
            for i in range(len(s)):
                if s[i].probability < worstval:
                    worstval = s[i].probability
                    worstidx = i
            del s[worstidx]
    
    return stacks

def prune_stack(stack): # histogram pruning
    start = time.time()
    stack.remove(min(stack, key=operator.attrgetter("probability")))
    return stack

def reorder_distance(alpha, start_curr, end_prev):
    return alpha**abs(start_curr - end_prev - 1)

# 
def translate(ttable, lm, src, w1=1, w2=1, w3=0.5, unknown_penalty=0.9): # beam search decoding
    def get_phrases(splitsrc, ttable):
        # Get all possible phrase pairs
        for i in range(len(splitsrc)):
            for j in range(i+1, len(splitsrc)+1):
                # if longer than the longest phrases in lookup table
                if len(splitsrc[i:j]) > 5:
                    break
                
                phrase = " ".join(splitsrc[i:j])
                
                # if not in the phrase table
                if not phrase in ttable:
                    continue
                
                # yield an answer
                for result, score in ttable[phrase]:
                    yield ((i, j), result, score)
    def overlaps(p1, p2):
        for i in p1:
            if i >= p2[0] and i <= p2[1]:
                return True
        return False

    print("Translation Started...")
    splitsrc = src.split(" ")
    
    # if the whole sentance is in the translation table we can just return that
    if src in ttable:
        n = Node()
        n.phrase = ttable[src][0][0]
        return n

    start_node = Node()
    # for i in splitsrc:
    #     start_node.translated.append(0)
    stacks = [[start_node]] # Each stack represents no. of words translated
    for i in range(len(splitsrc)):
        stacks.append([])
    
    phrasetable = {}
    for i in get_phrases(splitsrc, ttable):
        if not i[0][0] in phrasetable:
            phrasetable[i[0][0]] = {}
        if not i[0][1] in phrasetable[i[0][0]]:
            phrasetable[i[0][0]][i[0][1]] = []
        phrasetable[i[0][0]][i[0][1]].append((i[1], i[2]))
    
    
    for idx, s in enumerate(stacks):
        for idx2, n in enumerate(s):
            if idx2 % 10 == 0:
                print("STACK: %s, COL: %s" % (idx, idx2), end='\r')
            for start, cols in phrasetable.items():
                for end in cols:
                    if not overlaps(n.coverage, (start, end)): #TODO: this needs fixing
                        for (result, score) in phrasetable[start][end]:
                            if score < 0.001:
                                # Filter out the really bad predictions
                                continue
                            # Create hypothesis 
                            hypothesis = Node()
                            hypothesis.start = start
                            hypothesis.end = end
                            hypothesis.coverage = n.coverage + list(range(start, end ))#NOTE:+1 to end was here
                            hypothesis.phrase = n.phrase + " " + result
                            hypothesis.score = score
                            hypothesis.total_score += score
                            hypothesis.reorder_total += reorder_distance(0.5, start, n.end)

                            # non-log-linear
                            # hypothesis.probability = ((n.score + score) * reorder_distance(0.5, start, n.end)) * hypothesis.ngram_prob

                            # # LOG-LINEAR 
                            hypothesis.ngram_prob = compute_ngram_prob(lm, hypothesis.phrase, unknown_penalty)
                            if hypothesis.ngram_prob == 0: hypothesis.ngram_prob = 0.001
                            hypothesis.probability = np.exp((w1*np.log(n.total_score + score)) + (w2 * np.log(hypothesis.ngram_prob)) + (w3 * np.log(hypothesis.reorder_total)))

                            stackpos = len(hypothesis.coverage)
                            while len(stacks) <= stackpos:
                                stacks.append([])
                            stacks[stackpos].append(hypothesis)
                            
                            #NOTE: recombination would go here

                            # prune
                            # stacks = prune_stacks(stacks) #NOTE: pruning does not change stack position as it adds to future stacks
                            if len(stacks[stackpos]) > 100:
                                stacks[stackpos] = prune_stack(stacks[stackpos])
    
    laststack = []
    for i in stacks:
        i.sort(key=operator.attrgetter('probability'))
        if len(i) != 0:
            laststack = i
        # for j in i:
            # print(j.phrase + " (NGRAM_PROB:({}), SCORE:({}), TOTAL_PROB:({}))".format(j.ngram_prob, j.total_score, j.probability))
    return laststack

def translate_word(ttable, src):
    # Word based translation
    out = ""
    for i in src.split(" "):
        if i in ttable:
            out += " " + ttable[i][0][0]
        else:
            out += " " + i
    return out

def random_grid_search(lm, phrase_table, src, trgt, grid, n_search=10, n_translations=3):
    best_score = 0
    best_search = []

    bleu = Bleu(ngram=4, smooth="no_smooth")
    for i in range(n_search):
        print("Search %s/%s" % (i+1, n_search))
        # Choose random parameters
        w1 =  random.choice(grid[0])
        w2 =  random.choice(grid[1])
        w3 =  random.choice(grid[2])
        unknown_penalty = random.choice(grid[3])
        print("WEIGHTS = (w1 : %s, w2 : %s, w3 : %s, unknown_penalty : %s)" % (w1, w2, w3, unknown_penalty))

        for j in range(n_translations):
            rand = 0
            while True:
                rand = random.randint(0, len(src)-1)
                if len(cy[rand].split()) < 25:
                    break
            t = translate(phrase_table, lm, src[rand], w1, w2, w3, unknown_penalty)[-1].phrase
            print("> %s" % (src[rand].replace("\n", "")))
            print("= %s" % (trgt[rand].replace("\n", "")))
            print("< %s\n" % (t))
            bleu.update((t.split(), [trgt[rand].split()]))

        # Check to see if new hyperparameters are best
        score = bleu.compute().item()
        if score > best_score:
            best_score = score
            best_search = [w1, w2, w3, unknown_penalty]
        bleu.reset()
        print(str(best_search) + "\n")
    
    return best_search


if __name__ == "__main__":

    ## DECODING
    with open("models/cy-en/statistical/english-language-model.json", 'r') as json_file:
        langmodel = json.load(json_file)

    with open("models/cy-en/statistical/phrase-table.json", 'r') as json_file:
        dlist = json.load(json_file)

    target = "rydym yma heddiw i sicrhau cefnogaeth ar draws y byd i ein problemau genedlaethol"
    print("WORD TRANSLATION: " + translate_word(dlist, target))
    
    print("TESTING MODEL")
    (cy, en) = load_corpus("corpuses/cy-en/test/test.cy", "corpuses/cy-en/test/test.en")
    f = open("output/statistical/output.cy-en.txt", "w", encoding="utf-8")
    for c, e in zip(cy, en):
        best = translate(dlist, langmodel, c, 7.35346861, 4.50340169, 6.49193613, 0.9)[-1].phrase
        f.write(best + "\n")
    f.close()

    print("TUNING MODEL...")
    # Load tuning corpuses
    # We don't need the alignment
    (cy, en) = load_corpus("corpuses/fr-en/validate/validate.fr", "corpuses/fr-en/validate/validate.en")
    weights = random_grid_search(langmodel, dlist, cy, en, [[x / 2.0 for x in range(1,21)], [x / 2.0 for x in range(1, 21)], [x / 2.0 for x in range(1, 21)], [x / 10.0 for x in range(5, 10)]])
    unknown_word = weights[3]
    weights = weights[:-1]

    def optimize_translate(weights):
        print(weights)
        bleu = Bleu(ngram=4)
        for i in range(3):
            rand = 0
            while True:
                rand = random.randint(0, len(cy)-1)
                if len(cy[rand].split()) < 25:
                    break
            best = translate(dlist, langmodel, cy[rand], weights[0], weights[1], weights[2], unknown_word)[-1]
            print("> %s" % (cy[rand].replace("\n", "")))
            print("= %s" % (en[rand].replace("\n", "")))
            print("< %s \n" % (best.phrase))
            bleu.update((best.phrase.split(), [en[rand].split()]))
        return bleu.compute().item() * -1

    res = minimize(optimize_translate, weights, method='nelder-mead', options={"maxiter" : 100, "disp":True})
    print(res)
    print("TUNED!")


    start = time.time()
    print("PHRASE TRANSLATION: " + translate(dlist, langmodel, target, w1, w2, w3, unknown_penalty))
    print("Time taken %s" % (time.time() - start))

    print("(w1 (phrase-prob)=%f, w2 (ngram)=%f, w3 (reorder model)=%f, unknown_penalty=%f)" % (w1, w2, w3, unknown_penalty))
