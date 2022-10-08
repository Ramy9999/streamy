# Covid19 Chest Scan Images Detector [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ramy9999/streamy/main/app.py)

Link: https://share.streamlit.io/ramy9999/streamy/main/app.py

Clone or download zip to run on your local machine if you want.

## Overview of All types of Images
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/CT%20and%20Xray%20snip.PNG)


## Running Web Page of the Web APP
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/xray%20streamy%201%20right.PNG)
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/xray%20streamy%202.PNG)


## Healthy CT Chest Scan 
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/ct%20streamy%201.PNG)
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/ct%20streamy%20Normal%201.PNG)
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/ct%20streamy%20Normal%202.PNG)

## COVID19 CT Chest Scan
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/ct%20streamy%20Covid19%202.PNG)
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/ct%20streamy%20Covid19%203.PNG)

## Healthy Xray Chest Scan
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/xray%20test%20streamy%201.PNG)
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/xray%20test%20streamy%202%20normal.PNG)
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/xray%20test%20streamy%203%20normal.PNG)

## COVID19 Xray Chest Scan
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/xray%20test%20streamy%201%20Covid19.PNG)
![alt text](https://github.com/Ramy9999/streamy/blob/main/testy/xray%20test%20streamy%202%20Covid19.PNG)


# >> Details
## Step 1
* I have collected most of the publicly available and most common datasets of CT and Xray COVID19 and Normal Chest Scan Images.
* Did the google colab of the three models one for Xrays, another for CT and a third Multi Class one for Both.
* Preprocessed the data and used several libraries for data transformation like image augmentation and converting to numbers with numpy among others.
* Used both my own custom CNN for the first model built entirely, and also transfer learning with its own custom CNN as well.
* Trained and saved the very good 3 models of the previous.

## Step 2
* Tested the models customly on never seen chest scan images before.
* Got 96% Accuracy and slightly similar Validation and Test Accuracy, and other metrics were good as well.
* Then I built and deployed the streamlit app after uploading the project to GitHub via Git LFS and linked them for hosting the models on the web and for their efficient utilzation by anyone.


# >> Usage: 

Clone or download the zip of the project, and the requirements text file contains all the latest dependancies required to install, install that first then run like below:

To run this COVID19 Image Detection app =>
```
pip install -r requirements.txt
streamlit run app.py
```

(or)

Use and go to my working project online - go to this link: https://share.streamlit.io/ramy9999/streamy/main/app.py

* The web app opens up in a local host if executed localy offline. Then you can use it for COVID19 Chest Scan Image classification. That's it!

* Upload any type of image you want or even multiple images, provided that each image does not exceed 200 MB.

