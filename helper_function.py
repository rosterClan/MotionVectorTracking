import math
import cv2

def arrowdraw(img, x1, y1, x2, y2):
    radians = math.atan2(x1-x2, y2-y1)
    
    x12 = -3
    y12 = -3

    u12 = 3
    v12 = -3

    x11, y11, u11, v11 = 0, 0, 0, 0

    x11_ = x11*math.cos(radians) - y11*math.sin(radians) + x2
    y11_ = x11*math.sin(radians) + y11*math.cos(radians) + y2

    x12_ = x12 * math.cos(radians) - y12 * math.sin(radians) + x2
    y12_ = x12 * math.sin(radians) + y12 * math.cos(radians) + y2
    
    u11_ = u11 * math.cos(radians) - v11 * math.sin(radians) + x2
    v11_ = u11 * math.sin(radians) + v11 * math.cos(radians) + y2

    u12_ = u12 * math.cos(radians) - v12 * math.sin(radians) + x2
    v12_ = u12 * math.sin(radians) + v12 * math.cos(radians) + y2

    img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    img = cv2.line(img, (int(x11_), int(y11_)), (int(x12_), int(y12_)), (255, 0, 255), 3)
    img = cv2.line(img, (int(u11_), int(v11_)), (int(u12_), int(v12_)), (255, 0, 255), 3)
    
    return img

def arrowLine(img, x1, y1, x2, y2):
    radians = math.atan2(x1-x2, y2-y1)
    
    x12 = -3
    y12 = -3

    u12 = 3
    v12 = -3

    x11, y11, u11, v11 = 0, 0, 0, 0

    x11_ = x11*math.cos(radians) - y11*math.sin(radians) + x2
    y11_ = x11*math.sin(radians) + y11*math.cos(radians) + y2

    x12_ = x12 * math.cos(radians) - y12 * math.sin(radians) + x2
    y12_ = x12 * math.sin(radians) + y12 * math.cos(radians) + y2
    
    u11_ = u11 * math.cos(radians) - v11 * math.sin(radians) + x2
    v11_ = u11 * math.sin(radians) + v11 * math.cos(radians) + y2

    u12_ = u12 * math.cos(radians) - v12 * math.sin(radians) + x2
    v12_ = u12 * math.sin(radians) + v12 * math.cos(radians) + y2

    img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    return img