from image_preprocess import PreProcess
from line_segmentation import LineSegmentation
from word_segmentation import WordSegmentation
from char_segmentation import CharSegmentation
from cnn import CNN
from data import label_to_letter
import cv2 as cv
from autocorrect import Speller
import numpy as np
import sys

show = False


class OCR:
    def __init__(self, img=None, file=None):
        if img: self.p = PreProcess(img=img)
        if file: self.p = PreProcess(file=file)

        # resize and rotate the image
        self.p.resize(1000,1400)
        self.p.rotate()
        self.img = self.p.get_image()

        # create instance of neural network
        self.cnn = CNN(load=True)
        self.cnn.load_data()

    def text(self):
        output = ""

        # segment image by its line
        line_seg = LineSegmentation(self.img)
        line_seg.prep()
        lines = line_seg.segment()

        for line in lines:
            if show:
                cv.imshow("line", line)
                cv.waitKey()

            # segment each line by its word
            word_seg = WordSegmentation(line)
            word_seg.prep()
            words = word_seg.segment()

            for word in words:
                if show:
                    cv.imshow("word", word)
                    cv.waitKey()

                # segment each word by its character
                char_seg = CharSegmentation(word)
                char_seg.prep()
                chars = char_seg.segment_method1()

                word = ""
                for char in chars:
                    if show:
                        cv.imshow("char", char)
                        cv.waitKey()

                    # clean character and get its classification from the network
                    char = char_seg.clean_char(char)
                    letter = label_to_letter[self.cnn.predict(char) + 1]

                    # update the output text

                    word += letter
                output += word + " "
            output += "\n"

        output = output.strip().split(' ')
        spell = Speller()
        new_line = ""
        for l in output:
            new_line += spell(l) + " "

        return new_line


if __name__ == "__main__":
    if len(sys.argv) != 2:
        file = f'../inputs/intro.jpg'
    else:
        file = f'../inputs/{sys.argv[1]}'
    ocr = OCR(file=file)
    text = ocr.text()
    print(text)
    cv.imshow("output",cv.resize(ocr.img,(600,400)))
    cv.waitKey()






