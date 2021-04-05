import torch as pt
import torch.nn as tnn
import cv2
import csv
import numpy as np
from tqdm import tqdm


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [
        conv_layer(
            in_list[i],
            out_list[i],
            k_list[i],
            p_list[i]) for i in range(
            len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


def preprocess(image):
    image = (np.reshape(image, (128, 128)) * 255).astype(np.uint8)  # reshaping
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # converting to RGB values
    # denoising makes colors more fade
    image = cv2.fastNlMeansDenoising(image, None, 60, 7, 20)
    for x in range(len(image)):
        for y in range(len(image[x])):
            for z in range(len(image[x][y])):
                if image[x][y][z] < 70:  # RGB values 0 to 85 are close to total back
                    # replacing the black by white (=255); the back is the
                    # digits
                    image[x][y][z] = 255
    for x in range(len(image)):
        for y in range(len(image[x])):
            for z in range(len(image[x][y])):
                if image[x][y][z] != 255:  # now replace everything that's not white by black
                    image[x][y][z] = 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def inference(model, test_data, file_path):
    model.eval()
    device = pt.device('cpu')
    if pt.cuda.is_available():
        device = pt.device('cuda')
    model.to(device)
    inferences = [["ID", "Category"]]
    for i, img in enumerate(tqdm(test_data)):
        input = pt.tensor(img).unsqueeze(0).unsqueeze(0)
        input = input.to(device)
        out = int(pt.argmax(model.forward(input)))
        inferences.append([i, out])

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(inferences)


def inference_with_preprocessing(model, test_data, file_path):
    model.eval()
    device = pt.device('cpu')
    if pt.cuda.is_available():
        device = pt.device('cuda')
    model.to(device)
    inferences = [["ID", "Category"]]
    for i, img in enumerate(tqdm(test_data)):
        img = preprocess(img)
        input = pt.tensor(img).unsqueeze(0).unsqueeze(0)
        input = input.to(device)
        out = int(pt.argmax(model.forward(input)))
        inferences.append([i, out])

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(inferences)
