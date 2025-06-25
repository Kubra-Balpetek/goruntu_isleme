import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("3gi_algoritma/resim/sekil1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
canny = cv2.Canny(gray, 50, 150)

"""
sobel x ve y eksenleri için değişimleri bulur ve sonra birleştirir
laplacian kenar bulurken parlaklik değişiminin maksimum olduğu bölgeleri alir sobelin aksine yönden bağimsiz
canny daha etkili sobeldeki gibi gradyan kullaniyor 
kernel boyutu arttikca detay kaybi artar kenarlar daha kalin olur
"""

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(np.abs(sobel), cmap='gray')
plt.title("Sobel (5x5)")

plt.subplot(1,3,2)
plt.imshow(np.abs(laplacian), cmap='gray')
plt.title("Laplacian (5x5)")

plt.subplot(1,3,3)
plt.imshow(canny, cmap='gray')
plt.title("Canny (50,150)")

plt.show()


sobel_k7 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=7)
sobel_k3 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.imshow(np.abs(sobel_k7), cmap='gray')
plt.title("Sobel (7*7)")

plt.subplot(1,2,2)
plt.imshow(np.abs(sobel_k3), cmap='gray')
plt.title("Sobel (3*3)")

plt.show()


