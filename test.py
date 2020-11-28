import cv2
 
img_a=cv2.imread("img/7.jpg")
 
img_b=cv2.imread("img/8.jpg")
 
xishu=0.9
img_a=cv2.resize(img_a,(img_b.shape[1],img_b.shape[0]))
 
res=cv2.addWeighted(img_a,1-xishu,img_b,xishu,0)
cv2.imshow('asdf', res)
 
cv2.waitKey()