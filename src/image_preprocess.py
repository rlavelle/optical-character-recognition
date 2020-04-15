import cv2 as cv


class PreProcess:
    def __init__(self,file=None,img=None):
        if img:
            self.img = img
        if file:
            self.img = cv.imread(file)

    def resize(self,w,h):
        self.img = cv.resize(self.img, (w, h))

    def rotate(self):
        self.img = cv.rotate(self.img, cv.ROTATE_90_COUNTERCLOCKWISE)

    def show(self):
        cv.imshow("input", self.img)
        cv.waitKey()

    def get_image(self):
        return self.img