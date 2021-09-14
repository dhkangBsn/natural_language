import torch
def check_gpu(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    temp = torch.cuda.get_device_name(0)
    print(temp)


if __name__ == '__main__':
    check_gpu('check_gpu start')
