import torch

def main():
    train_data = 'you need to know how to code'
    word_set = set(train_data.split()) # 중복을 제거한 단어들의 집합인 단어 집합 생성.
    vocab = {word: i+2 for i, word in enumerate(word_set)} # 단어 집합의 각 단어에 고유한 정수 맵핑.
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    print(vocab)
    # 단어 집합의 크기만큼의 행을 가지는 테이블 생성.
    embedding_table = torch.FloatTensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.2, 0.9, 0.3],
        [0.1, 0.5, 0.7],
        [0.2, 0.1, 0.8],
        [0.4, 0.1, 0.1],
        [0.1, 0.8, 0.9],
        [0.6, 0.1, 0.1]])
    # 이제 임의의 문장 'you need to run'에 대해서 룩업 테이블을 통해 임베딩 벡터들을 가져와보겠습니다.
    sample = 'you need to run'.split()
    idxes = []
    # 각 단어를 정수로 변환
    for word in sample:
        try:
            idxes.append(vocab[word])
        except KeyError: # 단어 집합에 없는 단어일 경우 <unk>로 대체한다.
            idxes.append(vocab['<unk>'])
            print('unk', word)
    idxes = torch.LongTensor(idxes)
    # 룩업 테이블
    lookup_result = embedding_table[idxes,:] # 각 정수를 인덱스로 임베딩 테이블에서 값을 가져온다.
    print(lookup_result)

    # 임베딩 층 사용하기 => nn.Embedding()
    # 전처리는 동일한 과정을 거칩니다.
    train_data = 'you need to know how to code'
    word_set = set(train_data.split()) # 중복을 제거한 단어들의 집합인 단어 집합 생성
    vocab = {tkn : i+2 for i, tkn in enumerate(word_set)}
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    # 이제 nn.Embedding()을 사용하여 학습 가능한 임베딩 테이블을 만든다.
    import torch.nn as nn
    embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                                   embedding_dim=3,
                                   padding_idx=1)
    # nn.Embedding은 크게 두 가지 인자를 받는데 각각 num_embeddings과 embedding_dim입니다.
    # - num_embeddings : 임베딩을 할 단어들의 개수. 다시 말해 단어 집합의 크기임
    # - embedding_dim: 임베딩 할 벡터의 차원. 사용자가 정해주는 하이퍼파라미터
    # - padding_idx : 선택적으로 사용하는 인자. 패딩을 위한 토큰의 인덱스를 알려 줌
    print(embedding_layer.weight)
if __name__ == '__main__':
    main()