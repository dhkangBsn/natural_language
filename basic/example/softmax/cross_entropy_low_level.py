# https://wikidocs.net/60572
import torch
import torch.nn.functional as F

def main():
    torch.manual_seed(1)

    z = torch.FloatTensor([1,2,3])

    hypothesis = F.softmax(z, dim=0)
    print(hypothesis)
    print(hypothesis.sum())

    z = torch.rand(3,5, requires_grad=True)
    print(z)

    hypothesis = F.softmax(z, dim=1)
    print(hypothesis)
    print(hypothesis.sum(dim=1))

    # torch.randint(범위, 갯수)
    # print(torch.randint(50, ()).long())
    y = torch.randint(5, (3,)).long()
    print(y)

    y_one_hot = torch.zeros_like(hypothesis)
    print(y_one_hot)
    print( y.unsqueeze(1))
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    print(y_one_hot)

    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    print(cost)

if __name__ == '__main__':
    main()