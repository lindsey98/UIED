from os.path import join as pjoin
import cv2
import os
import UIED.detect_compo.ip_region_proposal as ip
from UIED.cnn.CNN import CNN


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def main(input_path_img, output_root):

    resized_height = resize_height_by_longest_edge(input_path_img)

    os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
    # switch of the classification func
    classifier = {}
    classifier['Elements'] = CNN('Elements')
    ip.compo_detection(input_path_img, output_root, classifier=classifier,
                       resize_by_height=resized_height, show=False)

if __name__ == '__main__':

    # set input image path
    input_path_img = 'D:\\ruofan\\Knowledge_graph_website\\data\\Facebook\\FP\\allsecurelocksmithandsecurity.com\\shot.png'
    output_root = 'data/output'

    main(input_path_img, output_root)