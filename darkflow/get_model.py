from darkflow.defaults import argHandler
from darkflow.dark import darknet
import numpy as np

if __name__ == "__main__":
    args = [
        '',
        '--model', '/home/moribo/darknet/cfg/tiny_face_two_label.cfg',
        '--load', '/home/moribo/darknet/backup_two_labels/tiny_face_two_label_final.weights'
    ]
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)
    darknet = darknet.Darknet(FLAGS)
    weight_list = []
    for layer in darknet.layers:
        if layer.type == 'convolutional':
            k = layer.w['kernel']
            b = layer.w['biases']
            # U, sigma, V = np.linalg.svd(k)
            # print(k.shape)
            # print(U.shape, V.shape, sigma.shape)
            weight_list.append([k, b])
    # print(weight_list)