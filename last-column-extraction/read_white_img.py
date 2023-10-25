import cv2
img = cv2.imread("/Users/vansh/PycharmProjects/Text-Extractor/last-column-extraction/white_image.png")
width, height , _  = img.shape
print(width, height)
for i in range(width):
    for j in range(height):
        if img[:,:,0][i,j]==0:
            img[:,:,0][i,j]=0
            img[:,:,1][i, j] = 0
            img[:,:,2][i, j] = 0
#cv2.imshow("New Image",img)
for i in range(width):
    for j in range(100):
        if (img[:,:,0][i,j] and img[:,:,1][i,j] and img[:,:,2][i,j]) <80 :
            intensity=img[:,:,0][i,j]
            print(intensity)
            break
for i in range(width):
    for j in range(height):
        if img[:,:,0][i,j]==intensity:
            img[:,:,0][i,j]=0
            img[:,:,1][i, j] = 0
            img[:,:,2][i, j] = 0
cv2.imshow("New Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
