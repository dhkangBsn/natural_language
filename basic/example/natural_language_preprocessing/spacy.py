import spacy

def main():
    en_text = "A Dog Run back corner near spare bedrooms"
    spacy_en = spacy.load('en')

    def tokenize(en_text):
        return [tok.text for tok in spacy_en.tokenizer(en_text)]

    print(print(tokenize(en_text)))

    return

if __name__ == '__main__':
    main()