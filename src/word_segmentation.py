import cv2 as cv
import numpy as np
from line_segmentation import LineSegmentation
from image_preprocess import PreProcess

debug = False


class WordSegmentation:
    def __init__(self,line):
        self.line = line
        self.words = []
        self.bw = None

    def prep(self):
        # grey scale and blurr the image
        gray = cv.cvtColor(self.line, cv.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv.filter2D(gray, -1, kernel)

        if debug:
            cv.imshow("gray", gray)
            cv.waitKey()

        # binarize the image
        self.bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 9)
        self.bw = 255 - self.bw

        if debug:
            cv.imshow("bw",self.bw)
            cv.waitKey()

    def segment(self):
        # dilate each component of the image vertically and slightly horizontally
        # so that each word becomes a single connected component for bounding boxes
        kernel = np.ones((100, 20), np.uint8)
        dilate = cv.dilate(self.bw, kernel, iterations=1)

        if debug:
            cv.imshow("dilate", dilate)
            cv.waitKey()

        # find components
        components, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # sort contours
        components = sorted(components, key=lambda c: cv.boundingRect(c)[0])

        for c in components:
            # skip small boxes
            if cv.contourArea(c) < 600:
                continue
            # Get bounding box
            x, y, w, h = cv.boundingRect(c)
            # Getting word image
            self.words.append(self.line[y:y + h, x:x + w])
            if debug: cv.rectangle(self.line, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if debug:
            cv.imshow("boxed", self.line)
            cv.waitKey()

        return self.words


if __name__ == "__main__":
    file = '../inputs/hello.jpg'

    # pre process the image
    preproc = PreProcess(file)
    preproc.resize(600, 1000)
    preproc.rotate()
    # preproc.show()
    img = preproc.get_image()

    # segment by line
    line_seg = LineSegmentation(img)
    line_seg.prep()
    lines = line_seg.segment()

    line = lines[0]

    # segment by word
    word_seg = WordSegmentation(line)
    word_seg.prep()
    words = word_seg.segment()
    word = words[0]

    cv.imshow("word",word)
    cv.waitKey()
