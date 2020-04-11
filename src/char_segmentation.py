import cv2 as cv
import numpy as np
from line_segmentation import LineSegmentation
from word_segmentation import WordSegmentation
from image_preprocess import PreProcess


class CharSegmentation:
    def __init__(self, word):
        self.word = word
        self.chars = []
        self.bw = None

    def prep(self):
        # grey scale and blurr the image
        gray = cv.cvtColor(self.word, cv.COLOR_RGB2GRAY)
        kernel = np.ones((2, 2), np.float32) / 4
        gray = cv.filter2D(gray, -1, kernel)

        # debug
        # cv.imshow("gray", gray)
        # cv.waitKey()

        # binarize the image
        self.bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 14)
        self.bw = 255 - self.bw

        # debug
        # cv.imshow("bw",self.bw)
        # cv.waitKey()

    def segment(self):
        # dilate the image horizontally to the contours are connected
        kernel = np.ones((15, 2), np.uint8)
        dilate = cv.dilate(self.bw, kernel, iterations=1)

        # debug
        # cv.imshow("dilate", dilate)
        # cv.waitKey()

        # find components
        components, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # sort contours
        components = sorted(components, key=lambda c: cv.boundingRect(c)[0])

        for c in components:
            # skip small boxes
            if cv.contourArea(c) < 0:
                continue
            # Get bounding box
            x, y, w, h = cv.boundingRect(c)
            # Getting line image
            self.chars.append(self.word[y:y + h, x:x + w])
            #cv.rectangle(self.word, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # debug
        # cv.imshow("boxed", self.word)
        # cv.waitKey()
        return self.chars

    def clean_char(self, char):
        char = cv.resize(char, (28, 28))
        char = cv.cvtColor(char, cv.COLOR_RGB2GRAY)
        kernel = np.ones((2, 2), np.float32) / 4
        char = cv.filter2D(char, -1, kernel)
        char = cv.adaptiveThreshold(char, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 14)
        char = 255 - char
        char = cv.flip(cv.rotate(char, cv.ROTATE_90_CLOCKWISE), 1)
        return char

if __name__ == "__main__":
    file = '../inputs/input.jpg'

    # pre process the image
    preproc = PreProcess(file)
    preproc.resize(540, 960)
    preproc.rotate()
    # preproc.show()
    img = preproc.get_image()

    # show image
    # cv.imshow("image",img)
    # cv.waitKey()

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

    char = chars[0]
    char = char_seg.clean_char(char)

    from model.cnn import CNN
    cnn = CNN(load=True)
    cnn.load_data()
    cnn.test()
    #letter = cnn.predict(char)
    #print(letter)

