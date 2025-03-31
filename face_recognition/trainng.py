from imutils import paths
import api as face_recognition
import pickle
import cv2
import os
from sklearn import neighbors

images_folder = 'train_images'

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(images_folder))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes, model="large")
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}
with open('./trained_data.pickle', "wb") as f:
    f.write(pickle.dumps(data))

n_neighbors = 2
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
knn_clf.fit(knownEncodings, knownNames)

with open('./knn_model', 'wb') as f:
    pickle.dump(knn_clf, f)
