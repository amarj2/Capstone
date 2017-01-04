import cv2
import numpy as np

img = cv2.imread('testimage.jpg',0)
h = np.size(img, 0)
w = np.size(img, 1)
wleft = w/10
hleft = h/10
hright = h - hleft
wright = w-wleft
feature = img[hleft:hright,wleft:wright]
img3 = img - img
img3[hleft:hright,wleft:wright] = feature
img2 = feature

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
cv2.imshow('Original',img)
cv2.imshow('Cropped',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(40,40))
cl1 = clahe.apply(img2)
cv2.equalizeHist(img2)

cv2.namedWindow('Global Eq', cv2.WINDOW_NORMAL)
cv2.namedWindow('Local Eq', cv2.WINDOW_NORMAL)
cv2.imshow('Global Eq',img2)
cv2.imshow('Local Eq',cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.medianBlur(cl1,5)
img = cv2.bilateralFilter(img,5,75,75)

cv2.namedWindow('Filtered', cv2.WINDOW_NORMAL)
cv2.imshow('Filtered',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

h = np.size(img, 0)
w = np.size(img, 1)

avg = 0;
for i in range(0,h):
    for j in range (0,w):
        avg += img[i][j]

avg = avg/float(h*w)
high = 0.35*avg
low = 0.1*avg

edges = cv2.Canny(img,low,high)

cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
cv2.imshow('Final',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''img = cv2.imread('testimage.jpg',0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()'''


'''def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

img = cv2.imread('vein_test.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,0)
contours,hier,a = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

LENGTH = len(contours)
status = np.zeros((LENGTH,1))

for i,cnt1 in enumerate(contours):
    x = i    
    if i != LENGTH-1:
        for j,cnt2 in enumerate(contours[i+1:]):
            x = x+1
            dist = find_if_close(cnt1,cnt2)
            if dist == True:
                val = min(status[i],status[x])
                status[x] = status[i] = val
            else:
                if status[x]==status[i]:
                    status[x] = i+1

unified = []
maximum = int(status.max())+1
for i in xrange(maximum):
    pos = np.where(status==i)[0]
    if pos.size != 0:
        cont = np.vstack(contours[i] for i in pos)
        hull = cv2.convexHull(cont)
        unified.append(hull)

cv2.drawContours(img,unified,-1,(0,255,0),2)
cv2.drawContours(thresh,unified,-1,255,-1)

cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
cv2.imshow('Binary',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


'''kernel = cv2.getGaussianKernel(3,0)
img = cv2.imread('vein_test.jpg')
img = cv2.bilateralFilter(img,5,75,75)
#img = cv2.sepFilter2D(img, -1, kernel, kernel)
#img = cv2.medianBlur(img,5)
#img = cv2.blur(img,(5,5))
#img = cv2.GaussianBlur(img,(3,3),0)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img4 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
          #cv2.THRESH_BINARY_INV,11,2)
img3 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
           cv2.THRESH_BINARY_INV,11,2)
#img3 = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
#img3 = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel)
#contours, hierarchy,a = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#ctr = np.array(contours).reshape((-1,1,2)).astype(np.int32)
#cv2.drawContours(img3, [ctr], 0, (0, 255, 0), -1)

edges = cv2.Canny(img2,100,200)
edges = cv2.bilateralFilter(edges,5,75,75)
cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
cv2.imshow('Edges',edges)
cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
cv2.imshow('Binary',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)
cv2.imshow('Hough Lines',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''plt.subplot(2,2,4),plt.imshow(edges,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,1),plt.imshow(img2,cmap = 'gray')
plt.title('Grey'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(img4,cmap = 'gray')
plt.title('Binary'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(img3,cmap = 'gray')
plt.title('Contour'), plt.xticks([]), plt.yticks([])

plt.show()'''

'''import cv2
import numpy as np

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
#lines = np.array(lines)
for rho,theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines.jpg',lines)
cv2.imshow('Hough Lines',lines)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

