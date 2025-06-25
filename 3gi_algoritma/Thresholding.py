import cv2
import matplotlib.pyplot as plt

img = cv2.imread("3gi_algoritma/resim/atam.jpg", cv2.IMREAD_GRAYSCALE)


ret1, th_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

ret2, th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

th_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

"""
Global Threshold Sabit bir eşik belirlenir piksel bu değerin altindaysa 0, üstündeyse 255 olur.
Otsu Threshold Eşik değeri otomatik hesaplanir piksel değerlerinin dağilimina göre optimal eşik bulunur.
Adaptive Threshold Görüntünün farkli bölgeleri için farkli eşik Özellikle işik değişimi olduğu zaman.
"""

plt.figure(figsize=(15,5))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Orijinal")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(th_global, cmap='gray')
plt.title("Global Threshold (127)")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(th_otsu, cmap='gray')
plt.title(f"Otsu Threshold (t={int(ret2):d})")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(th_adaptive, cmap='gray')
plt.title("Adaptive Threshold")
plt.axis('off')

plt.show()
