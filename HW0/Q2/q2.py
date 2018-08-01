from __future__ import print_function
from PIL import Image

img1 = Image.open("data/lena.png")
img2 = Image.open("data/lena_modified.png")

x, y = img2.size

for i in range(0, x):
    for j in range(0, y):
        if img1.getpixel((i,j)) == img2.getpixel((i,j)):
            img2.putpixel((i,j), 255)


img2.save("ans_two.png")




