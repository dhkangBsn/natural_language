import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

def main():

    # 3. CNN으로 MNIST 분류하기
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 랜덤 시드 고정
    torch.manual_seed(777)

    # GPU 사용 가능일 경우 랜덤 시드 고정
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.001
    training_epoch = 15
    batch_size = 100

    # 데이터로더를 사용하여 데이터를 다루기 위해서 데이터셋을 정의해 준다.
    mnist_train = dsets.MNIST(root='MNIST_data/',  # 다운로드 경로 지정
                              train=True,  # True를 지정하면 훈련 데이터로 다운로드
                              transform=transforms.ToTensor(),  # 텐서로 변환
                              download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',  # 다운로드 경로 지정
                             train=False,  # False를 지정하면 테스트 데이터로 다운로드
                             transform=transforms.ToTensor(),  # 텐서로 변환
                             download=True)

    # 데이터로더를 사용하여 배치 크기를 지정해준다.
    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size = batch_size,
                                              shuffle=True,
                                              drop_last = True)
    class CNN(torch.nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # 첫번째층
            # ImgIn shape=(?,28,28,1)
            # Conv -> (?, 28, 28, 32)
            # Pool -> (?, 14, 14, 32)
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )


            # 두번째층
            # ImgIn shape = (?, 14, 14, 32)
            # Conv -> (?, 14, 14, 64)
            # Pool -> (?, 7, 7, 64)
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

            # 전결합층 7x7x64 inputs -> 1- outputs
            self.fc = torch.nn.Linear(7*7*64, 10, bias=True)





if __name__ == '__main__':
    main()