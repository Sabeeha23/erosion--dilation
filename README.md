# Implementation-of-Erosion-and-Dilation
## Aim
To implement Erosion and Dilation using Python and OpenCV.
## Software Required
1. Anaconda - Python 3.7
2. OpenCV
## Algorithm:
### Step1:

Import required libraries (OpenCV, NumPy) and load the image in grayscale.

### Step2:

Define a structuring element (kernel) for morphological operations.

### Step3:

Apply erosion using cv2.erode() on the image with the defined kernel.

### Step4:

Apply dilation using cv2.dilate() on the image with the same kernel.

### Step5:

Display and compare the original, eroded, and dilated images.
 
## Program:

``` 
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load the image
img1=np.zeros((100,500),dtype='uint8')
font=cv2.FONT_HERSHEY_COMPLEX_SMALL

# Create the text using cv2.putText
cv2.putText(img1,'SABEEHA SHAIK' ,(10,70),font,2,(255),2,cv2.LINE_AA)


# Create the structuring element
kernel1=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

# Dilate the image
img_dilate=cv2.dilate(img1,kernel1)
img_erode=cv2.erode(img1,kernel1)

# Display the results
plt.figure(figsize=(11, 5))
plt.subplot(1,3,1)
plt.imshow(img1,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(img_dilate,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(img_erode,cmap='gray')

```
## Output:

![image](https://github.com/user-attachments/assets/6639784d-139f-4b08-b29c-ef5718de58f1)

## Result
Thus the generated text image is eroded and dilated using python and OpenCV.
