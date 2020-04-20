from ocr import OCR
from ocr_library import OCRlib

if __name__ == '__main__':
    files = ['../inputs/sample.jpg','../inputs/intro.jpg',
             '../inputs/lower-alphabet.jpg','../inputs/hello_world.jpg',
             '../inputs/upper-alphabet.jpg','../inputs/paragraph.jpg',
             '../inputs/rowan_lavelle.jpg',]

    for file in files:
        print("***************************************")
        ocr = OCR(file=file)
        text = ocr.text()
        ocrlib = OCRlib(file=file)
        text1 = ocrlib.ocr_core()
        print(text)
        print(text1)
        print("***************************************")
