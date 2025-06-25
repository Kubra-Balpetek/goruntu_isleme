#featureExtraction  

# cv2.sift
# görüntüdeki anahtar NOKTALARI ve bunların betimleyicilerini bulur.küçültüldüğünde,büyütüldüğünde,döndürüldüğünde değişmez

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('3gi_algoritma/resim/atam.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray, None)

img_with_kp = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(8,8))
plt.imshow(cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB))
plt.title("SIFT Anahtar Noktalari")
plt.axis('off')
plt.show()

# cv2.ORB
# sift gibi ama daha hizli gercek görüntülere uygun

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("3gi_algoritma/resim/atam.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(gray, None)

img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f"ORB Anahtar Noktalari (Toplam: {len(keypoints)})")
plt.axis('off')
plt.show()


# HOG
# Görüntüdeki kenar yapıları ve şekil bilgilerini temsil etmek 
# onceden egitilmis model sayesinde insan tespiti yapabiliriz

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("3gi_algoritma/resim/image.png")

hog = cv2.HOGDescriptor()

# önceden eğitilmiş insan tespiti modeli
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
# resim, pencerenin adım boyutu, kenar boşluğu, ölçekleme oranı

for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("HOG + SVM ile İnsan Tespiti")
plt.axis('off')
plt.show()
