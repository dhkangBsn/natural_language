import torch
import torch.nn as nn
def main():
    # 배치 크기 x 채널 x 높이(height) x 너비(weight)의 크기의 텐서를 선언
    inputs = torch.Tensor(1, 1, 28, 28)
    print('텐서의 크기 : {}'.format(inputs.shape))

    conv1 = nn.Conv2d(1, 32, 3, padding=1)
    print(conv1)

    conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    print(conv2)

    pool = nn.MaxPool2d(2)
    print(pool)

    out = conv1(inputs)
    print(out.shape)

    out = pool(out)
    print(out.shape)

    # out의 첫번째 차원이 몇인지 출력해 보겠습니다.
    print(out.size(0))

    # out의 첫번째 차원은 1입니다. 두번째 차원이 몇인지 출력해 보겠습니다.
    print(out.size(1))

    # 마찬가지로 out의 네번째 차원을 출력해 보겠습니다.
    print(out.size(3))

    # 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼처라
    out = out.view(out.size(0), -1)
    print(out.shape)
    # 배치 차원을 제외하고 모두 하나의 차원으로 통합된 것을 볼 수 있다.
    # 이제 이제 대해서 전결합층(Fully-Connected layer)를 통과시켜 보겠습니다.
    # 출력층으로 10개의 뉴런을 배치하여 10개 차원의 텐서로 변환한다.
    fc = nn.Linear(3136, 10) # input_dim = 3,3136, output_dim = 10
    out = fc(out)
    print(out.shape)



if __name__ == '__main__':
    main()