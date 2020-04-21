import cv2 as cv
import numpy as np
from line_segmentation import LineSegmentation
from word_segmentation import WordSegmentation
from image_preprocess import PreProcess
from cnn import CNN
from data import label_to_letter
import math

debug = False


class CharSegmentation:
    def __init__(self, word):
        self.word = word
        self.chars = []
        self.bw = None

    def prep(self):
        # grey scale and blurr the image
        gray = cv.cvtColor(self.word, cv.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.float32) / 4
        gray = cv.filter2D(gray, -1, kernel)

        if debug:
            cv.imshow("gray", gray)
            cv.waitKey()

        # binarize the image
        self.bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 14)
        self.bw = 255 - self.bw

        if debug:
            cv.imshow("bw",self.bw)
            cv.waitKey()

    def segment(self):
        # dilate each component of the image vertically so that each character
        # becomes a single connected component for bounding boxes
        kernel = np.ones((15, 2), np.uint8)
        dilate = cv.dilate(self.bw, kernel, iterations=1)

        if debug:
           cv.imshow("dilate", dilate)
           cv.waitKey()

        # find components
        components, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # sort components y axis top to bottom
        components = sorted(components, key=lambda c: cv.boundingRect(c)[0])

        for c in components:
            # skip small boxes
            if cv.contourArea(c) < 100:
                continue
            # Get bounding box
            x, y, w, h = cv.boundingRect(c)
            # Getting char image
            self.chars.append(self.word[y:y + h, x:x + w])
            cv.rectangle(self.word, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if debug:
            cv.imshow("boxed", self.word)
            cv.waitKey()

        return self.chars

    def clean_char(self, char):
        # gray scale
        char = cv.cvtColor(char, cv.COLOR_RGB2GRAY)

        # blurr image and threshold
        kernel = np.ones((2, 2), np.float32) / 4
        char = cv.filter2D(char, -1, kernel)
        char = np.uint8((char>char.mean())*255)
        char = 255 - char

        # if its a skinny rectangle
        if char.shape[0]/char.shape[1] > 3:
            # add a padding around the char so it doesnt touch the edges
            char = cv.copyMakeBorder(char, 3, 3, 3, 3, cv.BORDER_CONSTANT,0)
            # reshape to keep same width but cut height to 28
            char = cv.resize(char, (char.shape[1], 28))
            # reshape to have the width be 28 by evenly padding both sides
            char = cv.copyMakeBorder(char, 0, 0, math.ceil(14-char.shape[1]/2), math.ceil(14-char.shape[1]/2), cv.BORDER_CONSTANT,0)
            char = cv.resize(char, (28,28))
        else:
            # add a padding around the char so it doesnt touch edges
            char = cv.copyMakeBorder(char, 2, 2, 5, 5, cv.BORDER_CONSTANT,0)
            char = cv.resize(char, (28,28))

        # rotate image so its same rotation as training set
        char = cv.flip(cv.rotate(char, cv.ROTATE_90_CLOCKWISE), 1)

        # normalize the image
        char = (char - np.mean(char)) / (np.std(char))

        # reshape char
        char = char.reshape(1,28,28,1)
        return char


if __name__ == "__main__":
    file = '../inputs/demo.jpg'

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

    line = lines[0]

    # segment by word
    word_seg = WordSegmentation(line)
    word_seg.prep()
    words = word_seg.segment()
    word = words[0]

    char_seg = CharSegmentation(word)
    char_seg.prep()
    chars = char_seg.segment()

    cnn = CNN(load=True)
    cnn.load_data()

    for char in chars:
         char = char_seg.clean_char(char)
         cv.imshow(label_to_letter[cnn.predict(char)+1], char.reshape(28,28))
         cv.waitKey()



