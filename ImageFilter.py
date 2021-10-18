import numpy as np
from PIL import Image


def Filter(A, B):
    X = np.multiply(A, B)
    Y = np.sum(X)
    return Y


ori_image = Image.open('Ro9.jpg')
im = ori_image.convert('L')

imW, imH = im.size
new = np.zeros((imH - 3, imW - 3))

Sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Edge1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
Edge2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
Edge3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
Sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

for w in range(1, imW-2):
    for h in range(1, imH-2):
        trans = np.array([[im.getpixel((w-1, h-1)), im.getpixel((w, h-1)), im.getpixel((w+1, h-1))],
                      [im.getpixel((w-1, h)), im.getpixel((w, h)), im.getpixel((w+1, h))],
                      [im.getpixel((w-1, h+1)), im.getpixel((w, h+1)), im.getpixel((w+1, h+1))]])
        new[h-1, w-1] = Filter(trans, Sharpen)

pixels_out = []
for row in new:
    for tup in row:
        pixels_out.append(tup)

img_filter = Image.new(mode='L', size=(imW - 3, imH - 3))
img_filter.putdata(pixels_out)
ori_image.show()
im.show()
img_filter.show()

"""
for i in range(1, imW):
    for j in range(1, imH):
        pixVal = im.getpixel((i, j))
        if pixVal != (255, 255, 255):
            nonWhitePixels.append([i, j])
"""
