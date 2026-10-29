# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sobel_opencv import sobel_opencv
from sobel_native import sobel_native

image_bgr = cv2.imread("image.png")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

print(f"Изображение загружено: {image_gray.shape[1]}x{image_gray.shape[0]} пикселей")

RUNS = 5

times_cv = []
times_nat = []

for i in range(RUNS):
    result_cv,  t_cv  = sobel_opencv(image_gray)
    result_nat, t_nat = sobel_native(image_gray)
    times_cv.append(t_cv)
    times_nat.append(t_nat)

avg_cv  = np.mean(times_cv)  * 1000
avg_nat = np.mean(times_nat) * 1000

print(f"\nOpenCV  (cv2.Sobel): {avg_cv:.1f} мс")
print(f"Нативная (NumPy):    {avg_nat:.1f} мс")
print(f"OpenCV быстрее в {avg_nat / avg_cv:.1f} раз")

diff = np.abs(result_cv.astype(int) - result_nat.astype(int))
print(f"\nРасхождение между результатами:")
print(f"  Максимальное: {diff.max()} (из 255)")
print(f"  Среднее:      {diff.mean():.4f}")

cv2.imwrite("result_opencv.png",  result_cv)
cv2.imwrite("result_native.png",  result_nat)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
axes[0].set_title("Исходное изображение")
axes[0].axis("off")

axes[1].imshow(result_cv, cmap="gray")
axes[1].set_title(f"Собель OpenCV  ({avg_cv:.1f} мс)")
axes[1].axis("off")

axes[2].imshow(result_nat, cmap="gray")
axes[2].set_title(f"Собель нативный  ({avg_nat:.1f} мс)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("result_comparison.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["OpenCV\n(cv2.Sobel)", "Нативная\n(NumPy)"],
       [avg_cv, avg_nat],
       color=["#2196F3", "#FF5722"],
       edgecolor="black", width=0.4)
ax.set_ylabel("Среднее время (мс)")
ax.set_title(f"Сравнение производительности  ({RUNS} запусков)")
for i, val in enumerate([avg_cv, avg_nat]):
    ax.text(i, val + 0.5, f"{val:.1f} мс", ha="center", fontweight="bold")
ax.set_ylim(0, max(avg_cv, avg_nat) * 1.25)
plt.tight_layout()
plt.savefig("result_benchmark.png", dpi=120)
plt.close()

print("\nСохранено: result_opencv.png, result_native.png, result_comparison.png, result_benchmark.png")
