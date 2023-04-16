from train import CNN
import torch
from torchvision import transforms
from PIL import Image
import draw
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":
    category = os.listdir("./dataset")
    print(category)

    model = CNN().cuda()
    model.load_state_dict(torch.load("model32.pth"))

    tfm = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])


    draw.work()
    img = Image.open("./tmp/tmp.png")
    img = tfm(img)
    img[img < 0.95] = 00.
    img[img >= 0.95] = 1.
    img = 1 - img
    img = img.unsqueeze(0)


    plt.imshow(img[0, 0])
    plt.show()

    pred = model(img.cuda())
    _, pred = torch.max(pred, 1)
    print("answer:", category[pred])
