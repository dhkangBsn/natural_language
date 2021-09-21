# https://wikidocs.net/60683
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
def main():
    x = np.arange(-5.0, 5.0, 0.1)
    print(x)
    y = sigmoid(x)
    print(y)
    plt.plot(x,y)
    plt.plot([0,0],[1.0,0.0], ':')

    plt.title('Sigmoid Function')
    plt.show()

    return

if __name__ == '__main__':
    main()