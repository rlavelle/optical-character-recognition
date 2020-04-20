from PIL import Image
import pytesseract


class OCRlib:
    def __init__(self,file):
        self.file = file

    def ocr_core(self,):
        text = pytesseract.image_to_string(Image.open(self.file))
        return text


if __name__ == '__main__':
    file = '../inputs/sample.jpg'
    ocrlib = OCRlib(file=file)
    text = ocrlib.ocr_core()
    print(text)
