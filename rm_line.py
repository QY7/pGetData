import cv2

image = cv2.imread('4.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#读取灰度
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Remove horizontal
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (filter_size,filter_size))
result = cv2.erode(thresh,horizontal_kernel)
# detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
# cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(image, [c], -1, (255,255,255), 2)



# Repair image
# repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
# result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

# cv2.imshow('thresh', thresh)
cv2.imshow('detected_lines', result)
# cv2.imshow('image', image)
# cv2.imshow('result', result)
cv2.waitKey()