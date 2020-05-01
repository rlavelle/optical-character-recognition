import cv2 as cv
import numpy as np
from line_segmentation import LineSegmentation
from word_segmentation import WordSegmentation
from image_preprocess import PreProcess
import matplotlib.pyplot as plt
from cnn import CNN
from data import label_to_letter
import math

debug = False
write = False

class CharSegmentation:
    def __init__(self, word):
        self.word = word
        self.chars = []
        self.hist_chars = []
        self.bw = None

    def prep(self):
        # grey scale and blur the image
        gray = cv.cvtColor(self.word, cv.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.float32) / 4
        gray = cv.filter2D(gray, -1, kernel)

        if debug:
            if write: cv.imwrite('../outputs/gray.jpg',gray)
            cv.imshow("gray", gray)
            cv.waitKey()

        # binarize the image
        self.bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 14)
        self.bw = 255 - self.bw

        if debug:
            if write: cv.imwrite('../outputs/bw.jpg',self.bw)
            cv.imshow("bw",self.bw)
            cv.waitKey()

    def segment(self):
        # dilate each component of the image vertically so that each character
        # becomes a single connected component for bounding boxes
        kernel = np.ones((2, 2), np.uint8)
        dilate = cv.dilate(self.bw, kernel, iterations=1)

        if debug:
            if write: cv.imwrite('../outputs/dilate.jpg', dilate)
            cv.imshow("dilate", dilate)
            cv.waitKey()

        # fill in gaps of letters
        kernel = np.ones((5,5))
        closing = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)

        if debug:
            if write: cv.imwrite('../outputs/closing.jpg',closing)
            cv.imshow("closing", closing)
            cv.waitKey()

        # thinning of letters down to about 1 pixel
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv.erode(closing, kernel, iterations=1)

        if debug:
            if write: cv.imwrite('../outputs/erosion.jpg',erosion)
            cv.imshow("erosion", erosion)
            cv.waitKey()

        # lists to store x values and corresponding widths
        xs = []
        ws = []

        # create a frequency histogram by column
        hist = np.count_nonzero(erosion, axis=0).tolist()

        if debug:
            plt.plot(range(0, np.array(hist).shape[0]),np.array(hist))
            plt.show()

        # find the first 0/1 freq -> x_start
        # find the first > 1 freq -> x_end
        # start = (x_start+x_end)/2
        # find the next 0/1 freq -> x1_start
        # find the next > 1 freq -> x1_end
        # width = (x1_start+x1_end)/2-start
        # use (x1_start+x1_end)/2 as start of next char
        i = 0
        while i < len(hist):
            # at end of a char
            if hist[i] <= 1:

                # loop until we find the start of the next char
                j = i
                while j < len(hist) and hist[j] <= 1:
                    j += 1

                # j is now at the start of the next char
                # i is now at the end of the prev char
                x = math.floor((i+j)/2)

                if len(xs) != 0:
                    width = x - xs[-1]
                    ws.append(width)
                xs.append(x)
                i = j+1
            i += 1

        # get rid of the extra last x value its just the end of the word
        xs.pop()

        # loop through calculated x's and widths of chars and box them
        for i in range(len(xs)):
            self.hist_chars.append(self.word[0:dilate.shape[0], xs[i]:xs[i]+ws[i]])

            if debug:
                if write: cv.imwrite(f'../outputs/hist_char{i}.jpg', self.hist_chars[i])
                cv.imshow("hist chars", self.hist_chars[i])
                cv.waitKey()

        i = 0
        # re bound the char boxes to get correct y values
        for char in self.hist_chars:
            # grey scale and blur the char
            gray = cv.cvtColor(char, cv.COLOR_RGB2GRAY)
            kernel = np.ones((3, 3), np.float32) / 4
            gray = cv.filter2D(gray, -1, kernel)

            # re binarze image
            bw_char = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            bw_char = 255 - bw_char

            if debug:
                if write: cv.imwrite(f'../outputs/bw_single{i}.jpg', bw_char)
                cv.imshow("bw single char", bw_char)
                cv.waitKey()

            # dialte char
            kernel = np.ones((5, 5), np.uint8)
            dilate = cv.dilate(bw_char, kernel, iterations=1)

            if debug:
                if write: cv.imwrite(f'../outputs/dilate_single{i}.jpg', dilate)
                cv.imshow("dilate single char", dilate)
                cv.waitKey()

            # find components
            components, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for c in components:
                # skip small boxes
                if cv.contourArea(c) < 150:
                    continue
                # Get bounding box
                x, y, w, h = cv.boundingRect(c)
                # Getting char image
                self.chars.append(char[y:y + h, x:x + w])
                cv.rectangle(char, (x, y), (x + w, y + h), (0, 255, 0), 2)
            i += 1

        if debug:
            cv.imwrite('../outputs/boxed.jpg',self.word)
            cv.imshow("boxed", self.word)
            cv.waitKey()

        return self.chars

    def segment_old(self):
        # dilate each component of the image vertically so that each character
        # becomes a single connected component for bounding boxes
        kernel = np.ones((15, 2), np.uint8)
        dilate = cv.dilate(self.bw, kernel, iterations=1)

        if debug:
            if write: cv.imwrite('../outputs/dilate.jpg',dilate)
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
            if write: cv.imwrite('../outputs/boxed.jpg',self.word)
            cv.imshow("boxed", self.word)
            cv.waitKey()

        return self.chars

    def clean_char(self, char):
        # gray scale
        char = cv.cvtColor(char, cv.COLOR_RGB2GRAY)

        # blurr image and threshold
        kernel = np.ones((2, 2), np.float32) / 4
        char = cv.filter2D(char, -1, kernel)

        if write:
            cv.imwrite('../outputs/clean_gray.jpg', char)

        _ , char = cv.threshold(char, 0, 255, cv.THRESH_OTSU)
        char = 255 - char

        if debug:
            if write: cv.imwrite('../outputs/clean_bw.jpg', char)
            cv.imshow("pre clean char", char)
            cv.waitKey()

        # thin characters again, gets better accuracy
        kernel = np.ones((1, 2), np.uint8)
        char = cv.erode(char, kernel, iterations=1)

        if debug:
            if write: cv.imwrite('../outputs/clean_thin.jpg', char)
            cv.imshow("erosion", char)
            cv.waitKey()

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
    word = words[0]

    char_seg = CharSegmentation(word)
    char_seg.prep()
    chars = char_seg.segment()

    cnn = CNN(load=True)
    cnn.load_data()

    i = 0
    for char in chars:
         char = char_seg.clean_char(char)
         if write: cv.imwrite(f'../outputs/clean_char{i}.jpg', char.reshape(28,28))
         cv.imshow(label_to_letter[cnn.predict(char)+1], char.reshape(28,28))
         cv.waitKey()
         i+=1



