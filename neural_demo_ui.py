from tkinter import *
from tkinter import ttk
from neural import *

# init window
root = Tk()
root.title("Neural Translator")

langchoice = StringVar()
textbox = Text(root, height = 5)
textboxout = Text(root, height = 5)


lang = "cy"
src_lang = Field(tokenize=tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

trg_lang = Field(tokenize=tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

src_train = open("corpuses/%s-en/train/train.%s" % (lang, lang), 
                "r", encoding="utf-8").readlines()[:-200000]
tgt_train = open("corpuses/%s-en/train/train.en" % (lang),
                "r", encoding="utf-8").readlines()[:-200000]
train_data = create_dataset(src_train, tgt_train, (src_lang, trg_lang))
src_lang.build_vocab(train_data, min_freq=2)
trg_lang.build_vocab(train_data, min_freq=2)

INPUT_DIM = len(src_lang.vocab)
OUTPUT_DIM = len(trg_lang.vocab)


enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

SRC_PAD_IDX = src_lang.vocab.stoi[src_lang.pad_token]
TRG_PAD_IDX = trg_lang.vocab.stoi[trg_lang.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load(f"models/cy-en/neural/model.cy-en.pt"))

def translate_button():
    print("TRANSLATING...")
    translation, _ = translate_sentence(textbox.get(1.0, "end-1c"), src_lang, trg_lang, model, device)
    textboxout.delete(1.0, "end") # Clear textbox
    textboxout.insert(1.0, " ".join(translation).replace("<eos>", ""))

def changed_combo(e):
    lang = "cy"
    if langchoice.get() == "French":
        lang = "fr"
    elif langchoice.get() == "Danish":
        lang = "da"
    elif langchoice.get() == "Spanish":
        lang = "es"
    
    src_train = open("corpuses/%s-en/train/train.%s" % (lang, lang), 
                    "r", encoding="utf-8").readlines()[:-200000]
    tgt_train = open("corpuses/%s-en/train/train.en" % (lang),
                    "r", encoding="utf-8").readlines()[:-200000]
    
    global src_lang
    src_lang = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)
    
    global trg_lang
    trg_lang = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    train_data = create_dataset(src_train, tgt_train, (src_lang, trg_lang))

    src_lang.build_vocab(train_data, min_freq=2)
    trg_lang.build_vocab(train_data, min_freq=2)

    INPUT_DIM = len(src_lang.vocab)
    OUTPUT_DIM = len(trg_lang.vocab)


    global enc
    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device)

    global dec
    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device)

    SRC_PAD_IDX = src_lang.vocab.stoi[src_lang.pad_token]
    TRG_PAD_IDX = trg_lang.vocab.stoi[trg_lang.pad_token]

    global model
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load(f"models/{lang}-en/neural/model.{lang}-en.pt"))

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load(f"models/cy-en/neural/model.cy-en.pt"))


langlabel = Label(root, text="SOURCE LANG:")
combobox = ttk.Combobox(root, width = 10, textvariable = langchoice)
combobox["values"] = ("Welsh", "French", "Danish", "Spanish")
combobox.bind('<<ComboboxSelected>>', changed_combo)
combobox.current(0) 
langlabel.grid(row=0, column=0)
combobox.grid(row=0, column=1)

label1 = Label(root, text = "Source Text")
transbtn = Button(root, height = 2, text="Translate", command = lambda:translate_button())
label1.grid(row=2, column=0, columnspan=7)
textbox.grid(row=3, column=0, columnspan=7)
transbtn.grid(row=4, column=0, columnspan=7)

label2 = Label(root, text = "Target Text")
label2.grid(row=5, column=0, columnspan=7)
textboxout.grid(row=6, column=0, columnspan=7)

root.mainloop()
