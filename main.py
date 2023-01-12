import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cascade import CascadeClassifier
from analysis import test_cascade, test_viola, train_cascade, train_viola
import sys
from integralImage import integralImage

def detect(clf,img):
    window_size = 24
    for i in range(0, img.shape[0] - window_size, 2):
        for j in range(0, img.shape[1] - window_size, 2):
            window = img[i:i+window_size, j:j+window_size]
            #convert to grayscale
            window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            if clf.classify(window) == 1:
                cv2.rectangle(img, (j, i), (j+window_size, i+window_size), (0, 255, 0), 2)

    cv2.imwrite("result.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # train_viola("20")
    # train_cascade([1,2,5,10,50],"cascade")

    if(len(sys.argv) < 3):
        print("Invalid arguments")
        return

    fg=0
    if sys.argv[1] == "--path":
        path = sys.argv[2]
        fg=1
    
    if sys.argv[3] == "--cascade":
        cascade_path = sys.argv[4]
    
    if sys.argv[1] == "--test":
        test_type = sys.argv[2]
        fg=2
    
    if sys.argv[3] == "--path":
        path = sys.argv[4]
    
    if sys.argv[3] == "--num":
        num = sys.argv[4]
    
    if fg == 1:
        
        img = cv2.imread(path)
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        
        img = cv2.resize(img, (384, 288))
        cv2.imshow("Resized",img)
        cv2.waitKey(0)
        
        clf = CascadeClassifier.load(cascade_path)
        detect(clf,img)
    
    if fg == 2:
        if test_type == "cascade":
            test_cascade(path)
        if test_type == "viola":
            test_viola(path)


if __name__ == '__main__':
    main()

 