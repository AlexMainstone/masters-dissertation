import json
import re
import numpy as np

# Loads the corpuses & alignment


def load_corpus(src, trgt, align=None):
    # Load first corpus
    srcfile = open(src, 'r', encoding='utf-8')
    srclines = srcfile.readlines()

    # Load second corpus
    trgtfile = open(trgt, 'r', encoding='utf-8')
    trgtlines = trgtfile.readlines()

    # Check if we need to load the alignment
    if align == None:
        return (srclines, trgtlines)

    # Load alignment
    alignfile = open(align, 'r', encoding='utf-8')
    alignlines = alignfile.readlines()

    # load into 2D tuple array from "Pharaoh format"
    alignarr = []
    for l in range(len(alignlines)):
        alignarr.append([])

        # Split into individual alignments
        for p in alignlines[l].split(" "):
            if not p.strip():
                continue
            i = p.split("-")
            alignarr[l].append((int(i[0].replace("\n", "")),
                               int(i[1].replace("\n", ""))))

    # return lists
    return (srclines, trgtlines, alignarr)


def split_and_save(path, src, trgt, align, train_size=400000, test_size=100000):
    # open files to save to
    src_train = open(path + "/train/src-lang-train.src", 'w', encoding='utf-8')
    trgt_train = open(path + "/train/trgt-lang-train.trgt",
                      'w', encoding='utf-8')
    align_train = open(path + "/train/alignment-train.txt",
                       'w', encoding='utf-8')

    # Save each line to file
    for i in range(train_size):
        src_train.write(src[i])
        trgt_train.write(trgt[i])
        out = ""
        for j in align[i]:
            out += " " + "-".join(map(str, j))
        align_train.write(out + "\n")

    # Close files
    src_train.close()
    trgt_train.close()
    align_train.close()

    # Repeat
    src_test = open(path + "/test/src-lang-test.src", 'w', encoding='utf-8')
    trgt_test = open(path + "/test/trgt-lang-test.trgt", 'w', encoding='utf-8')
    align_test = open(path + "/test/alignment-test.txt", 'w', encoding='utf-8')

    for i in range(train_size, train_size + test_size):
        src_test.write(src[i])
        trgt_test.write(trgt[i])
        out = ""
        for j in align[i]:
            out += " " + "-".join(map(str, j))
        align_test.write(out + "\n")

    src_test.close()
    trgt_test.close()
    align_test.close()

# src, trgt, align = load_corpus("corpus/cy-en/raw_corpus/clean.cy", "corpus/cy-en/raw_corpus/clean.en", "corpus/cy-en/raw_corpus/phrase-alignment.txt")
# split_and_save("corpus/cy-en", src, trgt, align)


def generate_ngram(src):
    # Generate language model
    model = {}
    iterator = 0
    for i in src:
        iterator += 1
        if iterator % 100 == 0:
            print("{} / {} LINES PROCESSED FOR N-GRAMS".format(iterator, len(src)))

        # Do some pre-processing on the text
        i = i.lower()  # Lowercase
        i = re.sub(r'[^\w]', ' ', i)  # remove symbols
        i = re.sub(r'0-9', '', i)  # remove numbers

        # Iterate through each word
        words = i.split(' ')
        words = ["&start; "] + ["&start;"] + words + \
            ["&end;"]  # Add start and end characters
        for w in range(2, len(words)):
            if not words[w-2] in model:
                model[words[w-2]] = {}
            if not words[w-1] in model[words[w-2]]:
                model[words[w-2]][words[w-1]] = {}

            # Add to model or increment word probability
            if words[w] in model[words[w-2]][words[w-1]]:
                model[words[w-2]][words[w-1]][words[w]] += 1
            else:
                model[words[w-2]][words[w-1]][words[w]] = 1

    # Normalize model values to be between 0 and 1
    for k1 in model.keys():
        for k2 in model[k1].keys():
            minimum = np.min(list(model[k1][k2].values()))
            total = np.sum(list(model[k1][k2].values()))
            # print("min:{}, sum:{}".format(minimum, total))
            for k3 in model[k1][k2].keys():
                model[k1][k2][k3] = (model[k1][k2][k3])/(total)
    return model

# Saves a formatted corpus for awesome-align


def save_formatted(corpus, path):
    save_file = open(path, 'w', encoding='utf-8')
    cleanfr = open("corpus/fr-en/raw_corpus/clean.fr", 'w', encoding='utf-8')
    cleanen = open("corpus/fr-en/raw_corpus/clean.en", 'w', encoding='utf-8')

    i = 0
    pos = -1
    while i < 500000:
        pos += 1
        if not corpus[pos][0].strip() or not corpus[pos][1].strip():
            print(corpus[pos][0] + "|||" + corpus[pos][1])
            continue  # One of the translation options are empty
        save_file.write((corpus[pos][0] + " ||| " +
                        corpus[pos][1]).replace("\n", "") + "\n")
        cleanfr.write(corpus[pos][0])
        cleanen.write(corpus[pos][1])
        i += 1
    save_file.close()

# src, trgt, align = load_corpus("corpus/fr-en/raw_corpus/europarl-v7.fr-en.fr", "corpus/fr-en/raw_corpus/europarl-v7.fr-en.en", "corpus/cy-en/raw_corpus/phrase-alignment.txt")
# save_formatted(list(zip(src, trgt)), "corpus/fr-en/raw_corpus/formatted-corp.txt")


def phrase_extraction(srctext, trgtext, alignment):
    def extract(f_start, f_end, e_start, e_end):
        if f_end < 0:  # 0-based indexing.
            return {}
        # Check if alignement points are consistent.
        for e, f in alignment:
            if ((f_start <= f <= f_end) and
               (e < e_start or e > e_end)):
                return {}

        # Add phrase pairs (incl. additional unaligned f)
        # Remark:  how to interpret "additional unaligned f"?
        phrases = set()
        fs = f_start
        # repeat-
        while True:
            fe = f_end
            # repeat-
            while True:
                # add phrase pair ([e_start, e_end], [fs, fe]) to set E
                # Need to +1 in range  to include the end-point.
                src_phrase = " ".join(srctext[i]
                                      for i in range(e_start, e_end+1))
                trg_phrase = " ".join(trgtext[i] for i in range(fs, fe+1))
                # Include more data for later ordering.
                phrases.add(((e_start, e_end+1), src_phrase, trg_phrase))
                fe += 1  # fe++
                # -until fe aligned or out-of-bounds
                if fe in f_aligned or fe == trglen:
                    break
            fs -= 1  # fe--
            # -until fs aligned or out-of- bounds
            if fs in f_aligned or fs < 0:
                break
        return phrases

    # Calculate no. of tokens in source and target texts.
    srctext = srctext.split()   # e
    trgtext = trgtext.split()   # f
    srclen = len(srctext)       # len(e)
    trglen = len(trgtext)       # len(f)
    # Keeps an index of which source/target words are aligned.
    f_aligned = [j for _, j in alignment]

    bp = set()  # set of phrase pairs BP
    # for e start = 1 ... length(e) do
    # Index e_start from 0 to len(e) - 1
    for e_start in range(srclen):
        # for e end = e start ... length(e) do
        # Index e_end from e_start to len(e) - 1
        for e_end in range(e_start, srclen):
            # // find the minimally matching foreign phrase
            # (f start , f end ) = ( length(f), 0 )
            # f_start ∈ [0, len(f) - 1]; f_end ∈ [0, len(f) - 1]
            f_start, f_end = trglen-1, -1  # 0-based indexing
            # for all (e,f) ∈ A do
            for e, f in alignment:
                # if e start ≤ e ≤ e end then
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            phrases = extract(f_start, f_end, e_start, e_end)
            if phrases:
                bp.update(phrases)
    return bp


def create_dict(srctext, trgtext, alignment, max_phrase_length=5):
    dlist = {}
    for i in range(int(np.floor(len(srctext)))):
        if i % 100 == 0:  # printing is very slow, only do it every 100 iterations
            print("{0}/{1} LINES PROCESSED FOR PHRASES".format(i, len(srctext)))

        phrases = phrase_extraction(srctext[i], trgtext[i], alignment[i])
        for p, a, b in phrases:
            a = a.lower()
            b = b.lower()
            # Enforce phrase length
            if len(b.split(" ")) > max_phrase_length:
                continue

            if a in dlist:
                if b in dlist[a]:
                    dlist[a][b] += 1
                else:
                    dlist[a][b] = 1
            else:
                dlist[a] = {b: 1}
    # Process list
    for a in list(dlist):
        total = sum(list(dlist[a].values()))
        for j in dlist[a].keys():
            dlist[a][j] /= total

        dlist[a] = list(dlist[a].items())
        dlist[a].sort(key=lambda x: x[1], reverse=True)

    return dlist


def distance_reorder(alpha, start, end):
    return pow(alpha, start - end - 1)


if __name__ == "__main__":
    # LOAD CORPUSES
    (srctext, trgtext, alignment) = load_corpus("corpuses/fr-en/train/train.fr",
                                                "corpuses/fr-en/train/train.en", "corpuses/fr-en/train/train.align")

    # ENGLISH LANGUAGE MODEL
    langmodel = generate_ngram(trgtext)

    # Save language model to json
    with open('models/english-language-model.json', 'w') as fp:
        json.dump(langmodel, fp)

    # ## TRANSLATION TABLE
    dlist = create_dict(srctext, trgtext, alignment)

    # # Save translation table to json
    with open('models/statistical-data-lower-test.json', 'w') as fp:
        json.dump(dlist, fp)
