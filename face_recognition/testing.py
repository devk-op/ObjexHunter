import api as face_recognition
import pickle
import cv2
from pkg_resources import resource_filename
import numpy as np

with open(resource_filename(__name__, 'trained_data.pickle'), 'rb') as f:
    data = pickle.load(f)

with open(resource_filename(__name__, 'knn_model'), 'rb') as f:
    knn_clf = pickle.load(f)


def face_recognition_distance(img, tolerance=0.6):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes, model="large")
    names = []
    for encoding in encodings:
        face_distances = face_recognition.face_distance(data["encodings"], encoding)
        min_ind = np.argmin(face_distances)
        if face_distances[min_ind] < tolerance:
            names.append(data["names"][min_ind])
        else:
            names.append("Unknown")
    return boxes, names


def face_recognition_knn(img, tolerance=0.6):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if len(boxes) < 1:
        return [], []
    encodings = face_recognition.face_encodings(rgb, boxes, model="large")
    closest_distances = knn_clf.kneighbors(encodings)
    names = knn_clf.predict(encodings)
    names[np.where(closest_distances[0][:, 0] >= tolerance)[0]] = "Unknown"
    return boxes, names


if __name__ == '__main__':
    from imutils import paths
    import os

    for image in paths.list_images('test_images'):
        img = cv2.imread(image)
        boxes, names = face_recognition_knn(img)
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        origin_name = os.path.basename(image)
        origin_name = origin_name[:origin_name.rfind('.')]
        cv2.imshow(origin_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(origin_name)
