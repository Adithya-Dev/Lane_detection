import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return mask



img = cv2.imread("/home/adithya/personal_ws/Udemy_course_practics/finding-lanes/data/images/test_image.jpg")
lane_img = np.copy(img)
canny = canny(lane_img)
cv2.imshow("result", region_of_interest(canny))
cv2.waitKey(0)