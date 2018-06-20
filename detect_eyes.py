import cv2
import numpy as np
import display
import argparse
import os
import torch
import torch.nn as nn

import scipy
from scipy.ndimage import gaussian_filter

class BlurConv(nn.Module):
    def __init__(self, nChannels, radius, sigma):
        super(BlurConv, self).__init__()
        self.radius = radius
        self.sigma = sigma
        self.conv = nn.Conv2d(nChannels, nChannels, radius, padding=(radius-1)//2, bias=False)
        # self.conv.bias = False
        a = np.zeros((radius, radius))
        gf = gaussian_filter(a, sigma)
        self.conv.weight = gf

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', type=str, default="")
argparser.add_argument('-o', '--output', type=str, default=".")
argparser.add_argument('-n', '--count', type=int, default=10)
argparser.add_argument('-d', '--display', action='store_true')
args = argparser.parse_args()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# eye_cascade = cv2.CascadeClassifier('../opencv-3.4.1/data/haarcascades_cuda/haarcascade_eye.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

fp = args.input + '/'
ff = os.listdir(fp)

# ff = []
# with open('namelist2') as fp:
#    for cnt, line in enumerate(fp):
#     #    print("Line {}: {}".format(cnt, line))
#     ff.append(line)

ff = sorted(ff)

if args.count != -1:
    ff = ff[:args.count]

print (len(ff))
# exit(0)

result = []
numlist = []

for f in ff:
    # read both the images of the face and the glasses
    image = cv2.imread(fp + f)
    if image is None:
        print("can not read Image", f , "(", fp, ")")
        exit(1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    centers = []
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cl = len(numlist)
    # if f == 'frame_253.png' or f == 'frame_254.png':
    #     print(faces)
    #     exit(0)
    sfaces = sorted(faces, key=lambda face: face[3], reverse=True)
    
    # iterating over the face detected
    for (x, y, w, h) in sfaces:
        # print ("found faces in ", (x, y, w, h))
        # create two Regions of Interest.
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        cv2.rectangle(image,(x, y),(x + w, y + h),(0,255,0),3)
        cc = [(255,0,0), (0,0,255), (0,255,0)]
        i = 0

        eps = [] #np.zeros((1,4))    

        seyes = sorted(eyes, key=lambda eye: eye[3], reverse=True)
        # Store the coordinates of eyes in the image to the 'center' array
        for (ex, ey, ew, eh) in seyes:
            centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))
            # print(" - And eyes: ", (ex, ey, ew, eh))
            # cv2.rectangle(image,(x + ex, y + ey),(x + ex + ew, y + ey + eh),cc[i % 3],3)

            eps.append(np.asarray([x + ex, y + ey, ew, eh]))

            numlist.append(x + ex)
            numlist.append(y + ey)
            numlist.append(ew)
            numlist.append(eh)

            i += 1

        if len(eps) > 2:
            eps = eps[:- (i - 2)]
            numlist = numlist[:-(i - 2) * 4]
        
        while len(eps) < 2:
            eps.append(np.asarray([0,0,0,0], dtype=np.int32))
            numlist.append(0)
            numlist.append(0)
            numlist.append(0)
            numlist.append(0)
            
        break
    
    
    
    print(f, " - ", len(eps), " len: ", len(eyes))
    result.append(np.asarray(eps))

    for ei in range(len(eps)):
        x, y, w, h = eps[ei]
        if w > 0 and h > 0:
            print("eye @ ", eps[ei])
            eye_img = image[y:y + h, x:x + w]
            cv2.imwrite(f"{args.output}/eye_{ei}_{f}", eye_img)

    # cv2.imwrite(args.output + "/final.png", image)
    # print(type(image), image.shape)
    if args.display:
        destRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        destRGB = cv2.flip(destRGB, 0)
        display.image(destRGB, win='eye', title='eyes', width=800, height=450)

# cv2.imshow('Lets wear Glasses', image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# net = BlurConv(3, 7, 1.5)
# print(net)

print(len(result))

result = np.asarray(result)

print(result.shape)

numlist = np.asarray(numlist)

print("nl ", numlist.shape)
# result = result.flat()
# print(result)

numlist.tofile("eyes.csv", sep=',')
result.tofile("eyes2.csv", sep=',')