import numpy as np
from PIL import Image 
from matplotlib import pyplot as plt

import cv2
import math

face_cascade = cv2.CascadeClassifier('C:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
img = cv2.imread('C:\\Users\\prasangsha.ganguly\\Pictures\\Camera Roll\\lal3.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

newim = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

h = newim.shape[0]
w = newim.shape[1]

def threshold():
  for y in range(0, h):
       for x in range(0, w):
            # threshold the pixel
            if (newim[y, x, 0] >=70 and newim[y, x, 1] >= 133  and newim[y, x, 1] <= 173 and newim[y, x, 2] >= 80 and newim[y, x, 2] <= 127):
            	if((newim[y,x,1] >= (0.3448 * newim[y,x,2]) + 76.2069) and (newim[y,x,1] <= (-1.15 * newim[y,x,2]) + 301.75)):
                   newim[y,x,0] = 255
                   newim[y,x,1] = 128
                   newim[y,x,2] = 128
            else:
              newim[y,x,0] = 0
              newim[y,x,1] = 128
              newim[y,x,2] = 128

threshold()

for (x,y,w,h) in faces:
	for i in range(y, y+h):
	    for j in range(x, x+w):
	        newim[i, j, 0] = 0
	        newim[i,j,1] = 128
	        newim[i,j,2] = 128




imgray = cv2.cvtColor(newim, cv2.COLOR_YCrCb2BGR)

imgr = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(imgr,(5,5),0)          	

ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

h1 = thresh1.shape[0]
w1 = thresh1.shape[1]


im3, contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

no_of_contours1 = len(contours1)

j=1
area1 = [None]*no_of_contours1
for k in range(1, no_of_contours1):
 area1[j] = cv2.contourArea(contours1[k])
 j=j+1
 

max_ar = max(area1)
for p in range(1, no_of_contours1):
  if (area1[p] == max_ar):
   break
   
handcnt = contours1[p]

   	
for y in range(0, h1):
       for x in range(0, w1):
          pixel = thresh1[y, x]	
          dist = cv2.pointPolygonTest(handcnt, (x,y) , False)  
          if(dist == 0 or dist == 1):
            thresh1[y, x] = 255
          else:
          	thresh1[y, x] = 0


cv2.drawContours(blur, contours1, p, (255,255,255), 3)

hul = cv2.convexHull(contours1[p])

hul = cv2.convexHull(contours1[p], returnPoints = False)
defects1 = cv2.convexityDefects(contours1[p], hul)
cnt1 = contours1[p]
no_of_defects = len(defects1)

extLeft = tuple(cnt1[cnt1[:, :, 0].argmin()][0])
extRight = tuple(cnt1[cnt1[:, :, 0].argmax()][0])
extTop = tuple(cnt1[cnt1[:, :, 1].argmin()][0])
extBot = tuple(cnt1[cnt1[:, :, 1].argmax()][0])


ctr = 1

angle_thresh = input("Input the angle threshold")
for i in range(defects1.shape[0]):
    s,e,f,do = defects1[i,0]
    start = tuple(cnt1[s][0])
    end = tuple(cnt1[e][0])
    far = tuple(cnt1[f][0])
    
    c = tuple(np.subtract( start , far) )
    d = tuple(np.subtract(end , far) )
    startdef = c[1]
    fardef = d[1]

    x1, y1 = start
    x2, y2 = end
    x3, y3 = far

    netx1 = x1 - x3
    nety1 = y1 - y3
    netx2 = x2 - x3
    nety2 = y2 - y3

    inner_product = netx1*netx2 + nety1*nety2
    len1 = math.hypot(netx1, nety1)
    len2 = math.hypot(netx2, nety2)

    denglu = math.acos(inner_product/ (len1 * len2))

    ang = denglu * 180 / math.pi

    #ang = ang * -1

    if(startdef<0 and fardef<0):
      if(ang < angle_thresh):
        ctr = ctr+1
       

print("number of fingers %d", ctr)    
    
