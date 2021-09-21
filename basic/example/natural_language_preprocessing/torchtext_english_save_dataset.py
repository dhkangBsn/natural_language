from torchtext.legacy.data import TabularDataset
from torchtext import data # torchtext.data 임포트
import urllib.request
import pandas as pd
def main():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/LawrenceDuan/"\
        "IMDb-Review-Analysis/master/IMDb_Reviews.csv",
        filename="data/IMDb_Reviews.csv")

    df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
    print(df.head())
    print('전체 샘플의 개수 : {}'.format(len(df)))

    train_df = df[:25000]
    test_df = df[25000:]

    train_df.to_csv("data/train_data.csv", index=False)
    test_df.to_csv("data/test_data.csv", index=False)

if __name__ == '__main__':
    main()