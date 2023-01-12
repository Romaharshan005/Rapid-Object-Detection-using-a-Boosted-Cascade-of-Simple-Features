import numpy as np
import pickle
from viola_jones import ViolaJones
from cascade import CascadeClassifier
import time
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics

def train_viola(t):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    clf = ViolaJones(T=t)
    clf.train(training, 2429, 4548)
    evaluate(clf, training)
    clf.save(str(t))

def test_viola(filename):
    with open("test.pkl", 'rb') as f:
        test = pickle.load(f)
    
    clf = ViolaJones.load(filename)
    evaluate(clf, test)

def train_cascade(layers, filename="Cascade"):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    
    clf = CascadeClassifier(layers)
    clf.train(training)
    evaluate(clf, training)
    clf.save(filename)

def test_cascade(filename="cascade"):
    with open("test.pkl", "rb") as f:
        test = pickle.load(f)
    
    clf = CascadeClassifier.load(filename)
    evaluate(clf, test)

def evaluate(clf, data):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0
    y_arr = []
    y_pred = []
    for x, y in data:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        y_pred.append(prediction)
        y_arr.append(y)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        
        correct += 1 if prediction == y else 0
    confusion_matrix = metrics.confusion_matrix(y_arr, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=["Not Face", "Face"])
    cm_display.plot()
    plt.savefig("confusion_matrix.png")
    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))

# if __name__ == "__main__":
#     # test_cascade("cascade")

#     # test_viola("200")

#     img = cv2.imread("solvay.jpg")
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #resize image
#     img = cv2.resize(img, (384, 288))

#     clf = CascadeClassifier.load("cascade1")
#     print(len(clf.clfs))
    
#     #print number of features in each layer
#     for i in range(len(clf.clfs)):
#         print(clf.clfs[i].T)


#     #iterate through image and classify faces
#     window_size = 24
#     for i in range(0, img.shape[0] - window_size, 2):
#         for j in range(0, img.shape[1] - window_size, 2):
#             window = img[i:i+window_size, j:j+window_size]
#             #convert to grayscale
#             window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
#             if clf.classify(window) == 1:
#                 cv2.rectangle(img, (j, i), (j+window_size, i+window_size), (0, 255, 0), 2)
    
#     cv2.imshow("result", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    # #get all the classifiers in the cascade
    # clfs = clf.clfs
    # #iterate through the classifiers and store face locations in a list for each classifier
    # face_locations = []
    # for i in range(len(clfs)):
    #     face_locations.append([])

    # for i in range(0, img.shape[0] - window_size, 2):
    #     for j in range(0, img.shape[1] - window_size, 2):
    #         window = img[i:i+window_size, j:j+window_size]
    #         #convert to grayscale
    #         window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    #         for k in range(len(clfs)):
    #             if clfs[k].classify(window) == 1:
    #                 face_locations[k].append((i, j, i+window_size, j+window_size))
    #                 break       
    
    # #show the faces detected by each classifier
    # for i in range(len(face_locations)):
    #     for (x1, y1, x2, y2) in face_locations[i]:
    #         cv2.rectangle(img, (y1, x1), (y2, x2), (0, 255, 0), 2)
    #     cv2.imwrite("result" + str(i) + ".jpg", img)


    # img = cv2.imread("cric.jpg")
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # Loading the required haar-cascade xml classifier file
    # haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # # Applying the face detection method on the grayscale image
    # faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    
    # # Iterating through rectangles of detected faces
    # for (x, y, w, h) in faces_rect:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # cv2.imshow('Detected faces', img)
    # cv2.waitKey(0)