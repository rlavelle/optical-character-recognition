import cv2 as cv
import numpy as np
from image_preprocess import PreProcess

debug = False


class LineSegmentation:
    def __init__(self, img):
        self.img = img
        self.lines = []
        self.bw = None

    def prep(self):
        # grey scale the image
        gray = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.float32) / 4
        gray = cv.filter2D(gray, -1, kernel)

        if debug:
            cv.imshow("gray", gray)
            cv.waitKey()

        # binarize the image
        self.bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 15)
        self.bw = 255 - self.bw

        if debug:
            cv.imshow("bw", self.bw)
            cv.waitKey()

    def segment(self):
        # dilate each component of the image horizontally so that each
        # line becomes a single connected component for bounding boxes
        kernel = np.ones((1, 100), np.uint8)
        dilate = cv.dilate(self.bw, kernel, iterations=1)

        if debug:
            cv.imshow("dilate", dilate)
            cv.waitKey()

        # find components
        components, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # sort components y axis top to bottom
        components = sorted(components, key=lambda c: cv.boundingRect(c)[1])

        for c in components:
            # skip small boxes
            if cv.contourArea(c) < 1500:
                continue
            # Get bounding box
            x, y, w, h = cv.boundingRect(c)
            # Getting line image
            self.lines.append(self.img[y:y + h, x:x + w])
            if debug: cv.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if debug:
            cv.imshow("boxed", self.img)
            cv.waitKey()

        return self.lines


if __name__ == "__main__":
    file = '../inputs/hello_world.jpg'

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

    cv.imshow("line",line)
    cv.waitKey()