import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# emotions: 1. angry  2. disgusted    3. fearful  4. happy    5. neutral  6. sad  7. suprised

# images folder path
dir_path = os.getcwd() + '\\images'

image = {}
for path in os.listdir(dir_path):
    dir = os.path.join(dir_path, path)
    image[path] = 0
    setImg = []
    for img in os.listdir(dir):
        getImage = cv2.imread(os.path.join(dir, img), 1)
        if getImage is not None:
            setImg.append(cv2.resize(getImage, (672, 420)))
        if getImage is None:
            print(path + ' ' + img)
    print(path + ': ' + str(len(setImg)))

img1_weight = 0
reverse = False # this is used to reverse the weight
imgKeys = list(image.keys())

i = 0
img1_weight = 0.5
reverse = False
first_image = None
second_image = None

while True:
    # continue to add or decrease weight
    if reverse:
        img1_weight -= 0.1
    else:
        img1_weight += 0.1

    # if img1_weight goes up, then img2_weight goes down accordingly and vice versa.
    img2_weight = 1 - img1_weight

    dst = cv2.addWeighted(first_image, img1_weight, second_image, img2_weight, 0, dtype=cv2.CV_32F)
    dst = cv2.addWeighted(image[imgKeys[0]][i], img1_weight, image[imgKeys[0]][i+1], img2_weight, 0, dtype=cv2.CV_32F)

    if i+2 > len(image[imgKeys[0]])-1:
        i = 0
    else:
        i += 2

    # we will have a 0.15 transition between frames for a smooth transition
    time.sleep(0.10)

    cv2.imshow('dst', image[imgKeys[0]][i])

    key = cv2.waitKey(3000)
    i += 1

    # if threshold is reached set reverse to True
    if img1_weight > 0.9:
        # let's have 1 second wait before reversing
        time.sleep(1)
        reverse = True

    # if inverse threshold is reached set reverse to False
    elif img1_weight < 0.1:
        time.sleep(1)
        reverse = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()