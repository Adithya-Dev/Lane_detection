import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_canny(image):
    """
    Apply canny to the image or Video that is fed into the function

    :param image:
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def make_coord(image, line_parameters):
    """
    Calculate the coordinates using the parameters from the argument
    """
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def calculate_average_slope_intercept(image, lines):
    """
    To calculate the intercept of slope in the region of lines
    """
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coord(image, left_fit_average)
    right_line = make_coord(image, right_fit_average)
    coord = np.array([left_line, right_line])
    return coord

def dispay_line(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_img

def region_of_interest(image):
    """
    compute the region of interst in the lane
    :param image:
    :return:
    """
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


"""
img = cv2.imread("/home/adithya/personal_ws/Udemy_course_practics/Adithya_practice_lane_detection/finding-lanes/data/images/test_image.jpg")
lane_img = np.copy(img)
canny = compute_canny(lane_img)
cropped_img = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
average_lines = calculate_average_slope_intercept(lane_img, lines)
line_img = dispay_line(lane_img, average_lines)
combined_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
cv2.imshow("result", combined_img)
cv2.waitKey(0)
"""

cap = cv2.VideoCapture("/home/adithya/personal_ws/Udemy_course_practics/Adithya_practice_lane_detection/finding-lanes/data/images/test2.mp4")
while(cap.isOpened()):
    cond, frame = cap.read()
    canny_image = compute_canny(frame)
    cropped_img = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = calculate_average_slope_intercept(frame, lines)
    line_img = dispay_line(frame, average_lines)
    combined_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    cv2.imshow("result", combined_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
