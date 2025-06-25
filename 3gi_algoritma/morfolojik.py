import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("3gi_algoritma/resim/image1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((3, 3), np.uint8)

# Morfolojik işlemler
erosion = cv2.erode(gray, kernel, iterations=1)
dilation = cv2.dilate(gray, kernel, iterations=1)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

titles = ['Orijinal (Gri)', 'Erosion', 'Dilation', 'Opening', 'Closing']
images = [gray, erosion, dilation, opening, closing]

"""
Dilation bölgenin sinirlarinin genişletilmesinde kullanilmaktadir.
erosion görüntü üzerinde bir aşindirma işlemi uygular ve gürültülü olarak adlandirilan bozuk olan görüntü temizlenir. 
acma icin önce erosion uygulanir ve ardindan dilation
kapama görüntüye dilation operatörü uygulanir ve ardindan Erosion
"""

plt.figure(figsize=(20,8))

plt.subplot(1, 5, 1)
plt.imshow(gray, cmap='gray')
plt.title(titles[0])
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(images[1], cmap='gray')
plt.title(titles[1])
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(images[2], cmap='gray')
plt.title(titles[2])
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(images[3], cmap='gray')
plt.title(titles[3])
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(images[4], cmap='gray')
plt.title(titles[4])
plt.axis('off')

plt.show()
