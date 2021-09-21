# https://wikidocs.net/60314
from torchtext import data # torch.data 임포트

def main():

    # 필트 정의

    # 텍스트
    TEXT = data.Field(sequential=True,
                      use_vocab=True,
                      tokenize=str.split,
                      lower=True,
                      batch_first=True,
                      fix_length=20)

    # 라벨
    LABEL = data.Field(sequential=False,
                       use_vacab=False,
                       batch_first=False,
                       is_target=True)

    # sequential : 시퀀스 데이터 여부 ( True가 기본값 )
    # use_vocab : 단어 집합을 만들 것인지 여부 ( True가 기본값)
    # tokenize : 어떤 토큰화 함수를 사용할 것인지 지정 (string.split이 기본값)
    # lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
    # batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부.
    # (False가 기본값)
    # is_target : 레이블 데이터 여부 (False가 기본값)
    # fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다.

    # 주의할 점은 위 필드는 어떻게 전처리를 진행할 것인지를 정의한 것이고,
    # 실제 훈련 데이터에 대해서 전처리는 진행하지 않았습니다.
    # 사실, 아직 훈련 데이터를 다운로드 받지도 않았습니다.

if __name__ =='__main__':
    main()