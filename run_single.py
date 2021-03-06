from os.path import join as pjoin
import cv2.cv2 as cv2
import os
import UIED.detect_compo.ip_region_proposal as ip

def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def main(classifier, input_path_img, output_root):

    resized_height = resize_height_by_longest_edge(input_path_img)
    os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
    org = ip.compo_detection(input_path_img, output_root, resize_by_height=resized_height, classifier=classifier, show=False)
    return org

if __name__ == '__main__':

    # set input image path
    input_path_img = './data/layout_testset/Facebook/FP/actionrehabandsupply.com/shot.png'
    output_root = './data/output'

    main(input_path_img, output_root)

    # for b in boxes:
    #     cv2.rectangle()