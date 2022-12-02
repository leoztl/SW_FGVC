import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.io import imread
from skimage import color
import os
import glob

class kpMatcher():
  def __init__(self,img1,img2):
    self.img1 = img1
    self.img2 = img2
  
  def match(self):
    target_gray = np.uint8(color.rgb2gray(self.img1)*255)
    test_gray = np.uint8(color.rgb2gray(self.img2)*255)

    descriptor = cv2.SIFT_create()
    self.kp_1, d_1 = descriptor.detectAndCompute(target_gray, None)

    self.kp_2, d_2 = descriptor.detectAndCompute(test_gray, None)

    bf = cv2.BFMatcher()
    self.matches = bf.knnMatch(d_1,d_2,k=2)

    self.good_match = []
    for m1,m2 in self.matches:
      if m1.distance > 0.6*m2.distance and m1.distance < 0.7*m2.distance:
        self.good_match.append([m1])

    ''' print("total matches number: ",len(self.matches))
    print("good matches number: ",len(self.good_match)) '''
    return len(self.good_match)

  def drawResult(self):
    img3 = cv2.drawMatchesKnn(self.img1,self.kp_1,self.img2,self.kp_2,self.good_match,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(22,14))
    plt.axis('off')
    plt.title('Good matched key points', fontsize=20)
    plt.imshow(img3)
    plt.show()

def similarityCheck(path):
  filename_list = sorted(glob.glob(os.path.join(path, '*.png')) + \
                    glob.glob(os.path.join(path, '*.jpg')))
  img_ls = [plt.imread(filename) for filename in filename_list]
  matched = []
  for i, img in enumerate(img_ls):
    optimalImg = [0,0,0]
    optimalVal = [0,0,0]
    for j, img2 in enumerate(img_ls):
      if i==j:
        continue
      matcher = kpMatcher(img, img2)
      matchNum = matcher.match()
      for idx2, val in enumerate(optimalVal):
        if matchNum > val:
          optimalImg[idx2] = j
          optimalVal[idx2] = matchNum
          break
    matched.append(optimalImg)
  return matched