import cv2
import os
import shutil
import pywt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import joblib
from sklearn.svm import SVC
from sklearn import svm
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

img = cv2.imread('./test_images/sharapova1.jpg')
# img.shape
# plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray.shape
# plt.imshow(gray, cmap='gray')

face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# (x, y, w, h) = faces[0]
# face_img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.destroyAllWindows()
# for (x, y, w, h) in faces:
#     face_img = cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = face_img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey,  ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)

def get_cropped_img_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

cropped_image = get_cropped_img_if_2_eyes('./test_images/sharapova1.jpg')
path_to_data = './dataset/'
path_to_cr_data = './dataset/cropped/'

img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    print(celebrity_name)

    celebrity_file_names_dict[celebrity_name] = []

    for entry in os.scandir(img_dir):
        roi_color = get_cropped_img_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.mkdir(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print('Generating images in folder', cropped_folder)

            cropped_file_name = celebrity_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name

            cv2.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1

def w2d(img, mode='haar', level=1):
    imArray = img
    # data conversion
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficient
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imgArray_H = np.uint8(imArray_H)

    return imArray_H

im_har = w2d(cropped_image, 'db1', 5)

class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1

x = []
y = []

for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combine_img = np.vstack((scalled_raw_img.reshape(32*32*3,1), scalled_img_har.reshape(32*32, 1)))
        x.append(combine_img)
        y.append(celebrity_name)
x = np.array(x).reshape(len(x), 4096).astype(float)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C= 10))])
# pipe.fit(x_train, y_train)
# print(pipe.score(x_test, y_test))
# print(classification_report(y_test, pipe.predict(x_test)))
model = SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
# print(classification_report(y_test, model.predict(x_test)))
# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto', probability=True),
#         'params': {
#             'svc_C': [1, 10, 100, 1000],
#             'svc_kernel': ['rbf', 'linear']
#         }    
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params': {
#             'randomforestclassifier_n_estimators': [1, 5, 10]
#         }
#     },
#     'logistic_regression': {
#         'model': LogisticRegression(solver='liblinear', multi_class='auto'),
#         'params': {
#             'logisticregresion_C': [1, 5, 10]
#         }
#     }
# }

# scores = []
# best_estimators = {}

# for algo, mp in model_params.items():
#     pipe = make_pipeline(StandardScaler(), mp['model'])
#     clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
#     clf.fit(x_train, y_train)
#     scores.append({
#         'model': algo,
#         'best_score': clf.best_score_,
#         'best_params': clf.best_params_
#     })
#     best_estimators[algo] = clf.best_estimator_

# df = pd.Dataframe(scores, columns=['model', 'best_score', 'best_params'])
# best_clf = best_estimators('svm')
cm = confusion_matrix(y_test, model.predict(x_test))
# print(cm)
joblib.dump(model, 'save_model.pkl')
plt.figure(figsize= (10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# plt.imshow(im_har, cmap='gray')
plt.show()

import json
with open('class_dictionary.json', 'w') as f:
    f.write(json.dumps(class_dict))