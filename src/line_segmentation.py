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

        # binarize the image
        self.bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 45)
        self.bw = 255 - self.bw

    def segment(self):
        # dilate the image horizontally to the contours are connected
        kernel = np.ones((3, 200), np.uint8)
        dilate = cv.dilate(self.bw, kernel, iterations=1)

        # find components
        components, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # sort contours
        components = sorted(components, key=lambda c: cv.boundingRect(c)[0])

        self.lines = []
        for c in components:
            # skip small boxes
            if cv.contourArea(c) < 1100:
                continue
            # Get bounding box
            x, y, w, h = cv.boundingRect(c)
            # Getting line image
            self.lines.append(self.img[y:y + h, x:x + w])
            # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return self.lines
