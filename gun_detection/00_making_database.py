import os
import cv2

path = 'guns-object-detection'
class_name = 'Gun'
Images_path = os.path.join(path, 'Images')
Labels_path = os.path.join(path, 'Labels')

list_text = open(os.path.join(path, 'list.txt'), 'w')
Images = os.listdir(Images_path)
for Image in Images:
    text = Image[:Image.rfind('.')] + '.txt'
    text_full_path = os.path.join(Labels_path, text)
    image_full_path = os.path.join(Images_path, Image)
    if not os.path.exists(text_full_path):
        continue
    if cv2.imread(image_full_path) is None:
        continue
    with open(text_full_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            x1, y1, x2, y2 = lines[0].replace('\n', '').split()
            list_text.write('{},{},{},{},{},{}\n'.format(image_full_path, x1, y1, x2, y2, class_name))
list_text.close()
