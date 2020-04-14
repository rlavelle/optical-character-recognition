import cv2 as cv
import numpy as np


class LineSegmentation:
    def __init__(self, img):
        self.img = img
        self.lines = []
        self.bw = None

    def prep(self):
        # grey scale the image
        gray = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        kernel = np.ones((2, 2), np.float32) / 4
        gray = cv.filter2D(gray, -1, kernel)

        # debug
        # cv.imshow("gray", gray)
        # cv.waitKey()

        # binarize the image
        self.bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 25)
        self.bw = 255 - self.bw

        # debug
        # cv.imshow("bw", self.bw)
        # cv.waitKey()

    def segment(self):
        # dilate the image horizontally to the contours are connected
        kernel = np.ones((1, 100), np.uint8)
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
            if cv.contourArea(c) < 1500:
                continue
            # Get bounding box
            x, y, w, h = cv.boundingRect(c)
            # Getting line image
            self.lines.append(self.img[y:y + h, x:x + w])
            # cv.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # debug
        # cv.imshow("boxed", self.img)
        # cv.waitKey()

        return self.lines
