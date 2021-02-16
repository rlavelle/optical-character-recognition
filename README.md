# optical-character-recognition

Program to decode handwritten documents into a machine readable format.

### requirements

Make sure you are using python3.6 (highest version of python tensorflow works with). Create a virtual environment by running:

```
python3.6 -m venv env
```

Once the environment is created run:

```
source env/bin/activate
```

Then go ahead an install the requirements:

```
python3.6 -m pip install --upgrade pip
python3.6 -m pip install -r requirements.txt
```

### running the program

If you want to see a sample output run (make sure you are in the src directory):

```
python3.6 ocr.py
```

This may take a second due to tensorflow loading in the model, but you should see the text for a sample image displayed in the terminal, then opencv will display the image with the bounding boxes. press any key to get back to the terminal (while the opencv image is in focus) if the image displayed is not in focus, click on the running python process in your dock to bring it up.

If you would like to test your own image do the following:
1. take the image, and make sure it is rotated 90 degrees clockwise (in the future I plan to add detection for image orientation but for now this is hardcoded)
2. place your image in the inputs folder
3. run `python3.6 ocr.py image-name.extension

Where image-name is the name of your image, and extension is either .jpg or .png

### Optimizations

- In the future I would like to firstly add image orientation detection, so that the input images do not have to strictly be rotated 90 degrees for input
- I would also like to attempt to use a better character segmentation method. If possible I would like to find a data set of words segmented with bounding boxes, and train a CNN to do the segmentation for me
