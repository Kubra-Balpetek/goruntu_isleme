import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

image = cv2.imread('3gi_algoritma/resim/kugu.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

h, w, c = image.shape

n_components = 3
reconstructed_channels = []

for i in range(c): 
    channel = image[:, :, i]
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(channel)
    restored = pca.inverse_transform(transformed)
    reconstructed_channels.append(restored)

restored_image = np.stack(reconstructed_channels, axis=2)
restored_image = np.clip(restored_image, 0, 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Orijinal Görüntü")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"PCA restore (3 Bileşen)")
plt.imshow(restored_image)
plt.axis('off')

plt.tight_layout()
plt.show()

"""
Görüntünün boyutlarini alyoruz yükseklik, genişlik ve kanallar icin
Her bir renk kanali (R, G, B  bgr formatina geliyor onu rgb ye çevirdik) için ayri PCA uyguluyorum
Kanaldaki veriyi alip PCA ile bu veriyi sadece 2 ana bileşene indirdim.Yani bu kanal orijinal boyutundan çok daha az veriyle temsil ediliyor.
Ardindan, bu 3 bileşenlik veriyi kullanarak orijinal görüntüyü geri tahmin ediyoruz.
yüksek boyutlu verileri kucultur
"""

