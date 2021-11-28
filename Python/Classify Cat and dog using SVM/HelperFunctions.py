#Imported Modules
import os
from cv2 import imread
from skimage.transform import resize
from skimage.feature import hog
import pandas as pd
import time
from sklearn import svm

# This Function used only To Create the File [Never Call it again]
def ReadImages(path):
    # Defined Variables
    dataset = pd.DataFrame()
    animal_type = []
    
    # Extraction Start From Here
    for filename in os.listdir(path):
        # Read image from file, resize it and get its features
        img = imread(os.path.join(path, filename))
        resized_img = resize(img, (128, 64))
        image_feature, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                                       cells_per_block=(2, 2), visualize=True, multichannel=True)
        
        # let features as columns then append it to dataset
        ser = pd.Series(image_feature).transpose()
        dataset = dataset.append(ser, ignore_index = True)          
        
        # Determine the output[animal type] such that 1 for cat and 0 for dog
        animal_type.append(1) if filename.find('cat') == 0 else animal_type.append(0)
    
    # Finally append the Colunm of animal type to dataset then save it into file
    dataset['Animal'] = animal_type
    dataset.to_csv('catsVsDogs.csv', index=False)

def TrainingProcess(X_train, Y_train):
    C = 0.1
    start = time.time()
    svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
    print("Training Time: ", time.time() - start)
    return svc
  
def TestingProcess(svc, X_train, X_test, Y_train, Y_test):
    accuracy = svc.score(X_train, Y_train)
    print('For Training Process SVM accuracy: ' + str(accuracy))
    accuracy = svc.score(X_test, Y_test)
    print('For Testing Process SVM accuracy: ' + str(accuracy))
