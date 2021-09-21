# https://wikidocs.net/60572
import torch
import torch.nn.functional as F

def main():
    z = torch.rand(3,5, requires_grad=True)
    hypothesis = F.softmax(z, dim=1)
    # low level
    print(torch.log(hypothesis))

    # 파이토치에서는 두 개의 함수를 결합한 F.log_softmax()라는 도구를 제공한다.
    print(F.log_softmax(z, dim=1))
    # 위 두 출력 결과가 동일한 것을 볼 수 있다.

    # low level
    y = torch.randint(5, (3,)).long()
    print(y)

    y_one_hot = torch.zeros_like(hypothesis)
    print(y_one_hot)
    print( y.unsqueeze(1))
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    print(y_one_hot)

    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    # high level
    print((y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean())
    print(F.nll_loss(F.log_softmax(z, dim=1), y))
    print(F.cross_entropy(z, y))
    return

if __name__ == '__main__':
    main()