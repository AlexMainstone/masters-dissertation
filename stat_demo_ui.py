from tkinter import *
from tkinter import ttk
from evaluate_statistical import *

# init window
root = Tk()
root.title("Statistical Translator")

w1val = StringVar()
w1val.set("7.35346861")
w2val = StringVar()
w2val.set("4.50340169")
w3val = StringVar()
w3val.set("6.49193613")
langchoice = StringVar()
textbox = Text(root, height = 5)
textboxout = Text(root, height = 5)

with open("models/cy-en/statistical/english-language-model.json", 'r') as json_file:
    langmodel = json.load(json_file)

with open("models/cy-en/statistical/phrase-table.json", 'r') as json_file:
    dlist = json.load(json_file)

def translate_button():
    print("TRANSLATING...")
    translation = translate(dlist, langmodel, textbox.get(1.0, "end-1c"), float(w1val.get()), float(w2val.get()), float(w3val.get()), 0.9)[-1]
    textboxout.delete(1.0, "end") # Clear textbox
    textboxout.insert(1.0, translation.phrase)

def changed_combo(e):
    lang = "cy"
    if langchoice.get() == "French":
        lang = "fr"

    global langmodel
    with open("models/%s-en/statistical/english-language-model.json" % (lang), 'r') as json_file:
        langmodel = json.load(json_file)

    global dlist
    with open("models/%s-en/statistical/phrase-table.json" % (lang), 'r') as json_file:
        dlist = json.load(json_file)
    


langlabel = Label(root, text="SOURCE LANG:")
combobox = ttk.Combobox(root, width = 10, textvariable = langchoice)
combobox["values"] = ("Welsh", "French")
combobox.bind('<<ComboboxSelected>>', changed_combo)
combobox.current(0) 
langlabel.grid(row=0, column=0)
combobox.grid(row=0, column=1)

w1label = Label(root, text="PHRASE WEIGHT:")
w1entry = Entry(textvariable=w1val)
w1label.grid(row=1,column=0)
w1entry.grid(row=1,column=1)

w2label = Label(root, text="NGRAM WEIGHT:")
w2entry = Entry(textvariable=w2val)
w2label.grid(row=1,column=3)
w2entry.grid(row=1,column=4)

w3label = Label(root, text="REORDER WEIGHT:")
w3entry = Entry(textvariable=w3val)
w3label.grid(row=1,column=5)
w3entry.grid(row=1,column=6)

label1 = Label(root, text = "Source Text")
transbtn = Button(root, height = 2, text="Translate", command = lambda:translate_button())
label1.grid(row=2, column=0, columnspan=7)
textbox.grid(row=3, column=0, columnspan=7)
transbtn.grid(row=4, column=0, columnspan=7)

label2 = Label(root, text = "Target Text")
label2.grid(row=5, column=0, columnspan=7)
textboxout.grid(row=6, column=0, columnspan=7)

root.mainloop()