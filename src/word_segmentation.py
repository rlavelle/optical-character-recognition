import cv2 as cv
import numpy as np
from line_segmentation import LineSegmentation
from image_preprocess import PreProcess

debug = False
write = False


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
            if write: cv.imwrite('../outputs1/gray.jpg',gray)
            cv.imshow("gray", gray)
            cv.waitKey()

        # binarize the image
        self.bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 7)
        self.bw = 255 - self.bw

        if debug:
            if write: cv.imwrite('../outputs1/bw.jpg',self.bw)
            cv.imshow("bw",self.bw)
            cv.waitKey()

        # remove noise from image (necessary for high quality camera)
        kernel = np.ones((4,4))
        self.bw = cv.morphologyEx(self.bw, cv.MORPH_OPEN, kernel)

        if debug:
            if write: cv.imwrite('../outputs1/denoise.jpg',self.bw)
            cv.imshow("de noise", self.bw)
            cv.waitKey()

    def segment(self):
        # dilate each component of the image vertically and slightly horizontally
        # so that each word becomes a single connected component for bounding boxes
        kernel = np.ones((100, 25), np.uint8)
        dilate = cv.dilate(self.bw, kernel, iterations=1)

        if debug:
            if write: cv.imwrite('../outputs1/dilate.jpg',dilate)
            cv.imshow("dilate", dilate)
            cv.waitKey()

        # find components
        components, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # sort components y axis top to bottom
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
            if write: cv.imwrite('../outputs1/img.jpg',self.line)
            cv.imshow("boxed", self.line)
            cv.waitKey()

        return self.words


if __name__ == "__main__":
    file = '../inputs/sample.jpg'

    # pre process the image
    preproc = PreProcess(file)
    preproc.resize(1000, 1400)
    preproc.rotate()
    # preproc.show()
    img = preproc.get_image()

    # segment by line
    line_seg = LineSegmentation(img)
    line_seg.prep()
    lines = line_seg.segment()

    line = lines[1]

    # segment by word
    word_seg = WordSegmentation(line)
    word_seg.prep()
    words = word_seg.segment()
    i = 0
    for word in words:
        if write: cv.imwrite(f'../outputs/word{i}.jpg', word)
        cv.imshow("word",word)
        cv.waitKey()
        i+=1
