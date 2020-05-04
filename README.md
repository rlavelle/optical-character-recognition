# optical-character-recognition

run ocr.py after installing packages for sample output

PACKAGES
--- ! use virtual env and run python3 -m pip install -r requirements.txt) ! --
opencv-python==4.2.0.34
numpy==1.17.2
matplotlib==3.0.1
autocorrect==1.1.0
idx2numpy==1.2.2
tensorflow==2.1.0

INFO:

File structure is contained the the folders data, inputs and src

DATA/
data contains the EMNIST data set in byte form

INPUTS/
inputs contains inputs images that you can test
-- if you want to test an input image yourself make sure it meets the following assumptions:
-- * image is taken rotated 90 degrees clockwise (preprocess rotates image counter clockwise at the start)
-- * your image is in this inputs forlder

src contains the code and model

! For all files with output press any key to skip to next picture

SRC/
* model is stored in the trained\_model folder

* data.py
-- * file for preprocessing data for CNN
-- * if you run this file you can see examples of the letter 'r' from the data set

* cnn.py
-- * class file for the convolutional neural network
-- * if you run this file it loads in the current model and prints the accuracy 

* image\_preprocess.py
-- * class to preprocess an image (rotation and resize)
-- * no main method for this file

* line\_segmentation.py
-- * class to preform line segmentation
-- * set debug to true to see process of segmentation
-- * if you run this file it runs through a sample input for line segmentation

* word\_segmentation.py
-- * class to preform word segmnetation 
-- * set debug to true to see process of segmentation
-- * if you run this file it runs through a sample input for word segmentation

* char\_segmentation
-- * class to preform character segmentation in 2 different ways
-- * the segment\_method2() function is the method based of cursive
-- * the segment\_method1() function is the method based off connected components 
-- * set debug to true to see process of segmentation
-- * if you run this file it shows the process for segmenting a sample word

* ocr.py
-- * class to preform the overall algorithm to generate a document
-- * set show to true to see over all process
-- * if you run this file it gives a sample output for the algorithm
-- * shows the segmented document, and prints out the text in the terminal
