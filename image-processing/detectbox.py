import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

image = cv2.imread('./test1.png')

#increasing saturation
image = Image.fromarray(image)
enhancer=ImageEnhance.Color(image)
image=enhancer.enhance(1.6)
image = np.array(image)

#convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blurring the image to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#sharpening the image
sharp = cv2.addWeighted(gray, 1.495, blurred, -0.4981, 0)

#apply edge detection to highlight edges of boxes
edges = cv2.Canny(sharp, 300, 500)

#find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#initialize a counter for boxes
box_count = 0

#loop over the contours
for contour in contours:
    #approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    #af the contour has 4 vertices, it's likely a box
    if len(approx) == 4:
        box_count += 1

        #draw the contour on the original image
        cv2.drawContours(edges, [approx], -1, (0, 255, 0), 2)

#display the result
# plt.imshow(edges)
# plt.show()
#cv2.waitKey(0)
# cv2.destroyAllWindows()

print("original image")
RED = (255,0,0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.5
font_color = RED
font_thickness = 2
text = f"Number of boxes counted: {box_count}"
x,y = 0,40
img_text = cv2.putText(edges, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
new_dimension = (600, 600)
img_resized = cv2.resize(img_text, new_dimension)
cv2.imshow("image",img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.putText(image,f"Number of boxes counted: {box_count}", org=(0,0))
# cv2.imshow("image",image)

print(f"Number of boxes counted: {box_count}")