from image_preprocess import PreProcess
from line_segmentation import LineSegmentation
from word_segmentation import WordSegmentation
from char_segmentation import CharSegmentation
import cv2 as cv

if __name__ == "__main__":
    file = '../inputs/input.jpg'

    # pre process the image
    preproc = PreProcess(file)
    preproc.resize(540,960)
    preproc.rotate()
    # preproc.show()
    img = preproc.get_image()

    # show image
    cv.imshow("image",img)
    cv.waitKey()

    # segment by line
    line_seg = LineSegmentation(img)
    line_seg.prep()
    lines = line_seg.segment()

    # print line segments
    for line in lines:
        cv.imshow("line", line)
        cv.waitKey()

        # segment by word
        word_seg = WordSegmentation(line)
        word_seg.prep()
        words = word_seg.segment()

        for word in words:
            cv.imshow("word", word)
            cv.waitKey()

            char_seg = CharSegmentation(word)
            char_seg.prep()
            chars = char_seg.segment()
            for char in chars:
                #cv.imshow("char", char)
                #cv.waitKey()
                pass





