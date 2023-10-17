import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
def centralize(img):
    _, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


    coords = cv2.findNonZero(thresholded)
    x, y, width, height = cv2.boundingRect(coords)

    crop_x = x
    crop_y = y
    crop_width = width
    crop_height = height


    cropped_image = thresholded[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    return cropped_image


model = load_model('/Users/vansh/PycharmProjects/Text-Extractor/script/model/cnn-mnist-new.h5')
img = cv2.imread('/Users/vansh/PycharmProjects/Text-Extractor/image_processing/451b6743-54e4-49cd-9574-532ccd3d14e1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

predicted_lines=[]
phone_txt=""
lines = []
current_line = []
im=[]
padding=8
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    x=x
    y=y
    w=w
    h=h
    if w < 10 or h < 10:
        continue
    if len(current_line) > 0 and y > current_line[0][0][1] + h:
        current_line = sorted(current_line, key=lambda c: c[0][0])
        lines.append(current_line)
        current_line = []

    character = gray[y:y+h, x:x+w]
    current_line.append(((x, y, w, h), character))

current_line = sorted(current_line, key=lambda c: c[0][0])
lines.append(current_line)

for line in lines:
    for _, character in line:
        (thresh, im_bw) = cv2.threshold(character, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Sharpen the image
        padding = 15
        
        im_bw=centralize(im_bw)
        im_bw2 = cv2.resize(im_bw, (28, 28)) 
        inverted_image = cv2.bitwise_not(im_bw2)

        w_new = inverted_image.shape[1] + padding
        h_new = inverted_image.shape[0] + padding
        image_centralized = np.ones((h_new, w_new), dtype=np.uint8) * 0 
        m=0
        k=0
        for i in range(inverted_image.shape[0]):
            for j in range(inverted_image.shape[1]):
                image_centralized[padding//2+i][padding//2+j]=inverted_image[i][j]

        resized_image = cv2.resize(image_centralized, (28, 28), interpolation=cv2.INTER_LINEAR)
        x_pred = resized_image.reshape(28, 28, 1)  
        
        
        kernel = np.ones((3, 3), np.uint8)  

        dilated_image = cv2.dilate(x_pred, kernel, iterations=1)
        batch = np.array([x_pred])  
#         cv2.imshow('Line', dilated_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        output=model.predict(batch)
        output=np.argmax(output)
        phone_txt=phone_txt+str(output)
        print(output,end=" ")

    print()
    predicted_lines.append(phone_txt)
    phone_txt=""

for i in predicted_lines:
    print(i)