# 1. 기본셋팅
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def main():
    # 현재 실습하고 있는 파이썬 코드를 재실행해도
    # 다음에도 같은 결과가 나오도록 랜덤 시드(random see)를 줍니다.
    torch.manual_seed(1)
    # 실습을 위한 기본적인 셋팅이 끝났습니다.
    # 이제 훈련 데이터인 x_train과 y_train을 선언합니다.

    # 2. 변수 선언
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])
    # 실습을 위한 기본적인 셋팅이 끝났습니다. 이제 훈련 데이터인 x_train과
    # y_train을 선언합니다.
    print(x_train)
    print(x_train.shape)

    # 3. 가중치와 편향의 초기화
    # 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
    W = torch.zeros(1, requires_grad=True)
    #
    print(W)
    # 가중치 W가 0으로 초기화 되어 있으므로 0이 출력된 것을 확인할 수 있다.
    # 위에서 requires_grad = True가 인자로 주어진 것을 확인할 수 있음
    # 이 변수는 학습을 통해 계속 값이 변경되는 변수임을 의미
    # 마찬가지로 편향 b도 0으로 초기화하고, 학습을 통해 값이 변경되는 변수임을
    # 명시
    b = torch.zeros(1, requires_grad=True)
    print(b)

    # 현재 가중치 W와 b 둘 다 0이므로 현 직선의 방정식은 다음과 같습니다.
    # y = 0 * x + b
    # 지금 상태에선 x에 어떤 값이 들어가도 가설은 0을 예측하게 된다.
    # 즉, 아직 적절한 W와 b의 값이 아니다.

    # 4. 가설 세우기
    # 파이토치 코드 상으로 직선의 방정식에 해당되는 가설을 선언합니다.
    # H(x) = Wx + b
    # hypothesis = x_train * W + b
    # print(hypothesis)

    # 5. 비용 함수 선언하기
    # 파이토치 코드 상으로 선형 회귀의 비용 함수에 해당되는 평균 제곱 오차를 선언한다.
    # cost(w,b)
    # torch.mean()을 이용한 평균 계산 적용
    # cost = torch.mean((hypothesis - y_train) ** 2)
    # print(cost)

    # 경사하강법의 일종
    # optimizer = optim.SGD([W,b], lr=0.01)

    # gradient를 0으로 초기화
    # optimizer.zero_grad()

    # 비용 함수를 미분하여 gradient 계산
    # cost.backward()

    # W와 b를 업데이터
    # optimizer.step()

    # 7. 전체코드드
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])
    # 모델 초기화
    W = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # optimizer 설정
    optimizer = optim.SGD([W,b], lr=0.01)

    nb_epochs = 1999 # 원하는 만큼 경사 하강법을 반복
    for epoch in range(nb_epochs + 1):

        # H(x) 계산
        hypothesis = x_train * W + b

        # cost 계산
        cost = torch.mean((hypothesis - y_train) ** 2)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 100 번 마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))

if __name__ == '__main__':
    main()