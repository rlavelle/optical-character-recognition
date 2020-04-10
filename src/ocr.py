from image_preprocess import PreProcess
from line_segmentation import LineSegmentation
import cv2 as cv

if __name__ == "__main__":
    file = '../inputs/input.jpg'

    # pre process the image
    preproc = PreProcess(file)
    preproc.resize(540,960)
    preproc.rotate()
    preproc.show()
    img = preproc.get_image()

    # segment by line
    line_seg = LineSegmentation(img)
    line_seg.prep()
    lines = line_seg.segment()

    for line in lines:
        cv.imshow("input", line)
        cv.waitKey()
