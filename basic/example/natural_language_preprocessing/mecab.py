import spacy
spacy_en = spacy.load('en')
def main():
    komoran = Mecab()

    kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
    print(kor_text.split())

    return

if __name__ == '__main__':
    main()

