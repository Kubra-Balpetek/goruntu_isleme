import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
görüntüdeki benzer renkleri gruplamak (clustering) için kullanilir.
Görüntüyü K adet renge indirerek bölütleme (segmentasyon) yapmak.
k degeri büyük olursa daha fazla renkle gruplanacaği için detay fazla olur 
k kücük olduğunda önemli detaylari kacirabiliriz
"""

image = cv2.imread('3gi_algoritma/resim/kugu.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

# KMeans kriterleri
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# farkli k değerlerinde görmek icin
K_values = [2, 4, 10]
segmented_images = []

for K in K_values:
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(image.shape)
    segmented_images.append(segmented)


plt.figure(figsize=(15, 8))
plt.subplot(1, 4, 1)
plt.title("Orijinal Görüntü")
plt.imshow(image)
plt.axis('off')

for i, K in enumerate(K_values):
    plt.subplot(1, 4, i + 2)
    plt.title(f"K = {K}")
    plt.imshow(segmented_images[i])
    plt.axis('off')

plt.show()
