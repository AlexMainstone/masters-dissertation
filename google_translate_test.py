from lib.google_trans_new.google_trans_new import google_translator

translator = google_translator()

        # words = re.findall(r"[\w']+|[.,!?;]", p.lower().replace(" &apos;", "'").replace("&amp;", "&"))
        # for i, wrd in enumerate(words):
            # words[i] = wrd.replace("'", " &apos;").replace("&", "&amp;")
#NOTE: Google will time out if you do too many requests, 100k is not achievable
file = open("corpuses/fr-en/test/test.fr", "r", encoding="utf-8") #welsh-english
lines = file.readlines()
file.close()

out = []
count = 0
for l in lines[:1000]: # first 5k in test set
    count +=1
    try: # Google has a cap for requests ~1200, we'll take what we can
        t = translator.translate(l.replace("&apos;", "'").replace("&amp;", "&"), lang_src='fr')
    except:
        print("COULD NOT FETCH, SAVING.")
        break
    out.append(t)
    print("{}) {}".format(count, t))

file = open("output/google_translate/output.2.fr-en.txt", "w", encoding="utf-8")
for l in out:
    file.write(l.replace("&", "'") + "\n")
file.close()