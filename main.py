import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return -1, -1
    (x, y, w, h) = faces[0]
    return image[y:y + w, x:x + h], faces[0]


def prepare_training_data(training_data_path):
    detected_faces = []
    face_labels = []
    # access list of folders  of training images
    training_images_dir = os.listdir(training_data_path)

    for dir_name in training_images_dir:
        # labeling the names of each folder
        label = int(dir_name)

        # set the path of the each folder path
        training_images_path = training_data_path + "/" + dir_name

        training_images_name = os.listdir(training_images_path)
        for image_name in training_images_name:
            # get the path of each image
            image_path = training_images_path + "/" + image_name
            # read each train image
            image = cv2.imread(image_path)
            # detect the face of each image

            face, rect = detect_face(image)



            # resize the each detected face because eigenface algorithm only accept the same size of all images
            resized_face = cv2.resize(face, (120, 120), interpolation=cv2.INTER_AREA)
            # store or append the detected faces and labels into the two lists
            detected_faces.append(resized_face)
            face_labels.append(label)

    return detected_faces, face_labels



# the eigenface module

# we must train it however we have to turn the labels into an array using numpy

# drawing a rectangle around the  predicted face
def draww_rectangle(test_img, rect):
    # coordinates of rect
    x, y, w, h = rect
    # draw the rect
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# this fn will help us write labels of predicted face
def write_text(test_img, label_text, x, y):
    # write the label
    cv2.putText(test_img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)




def predict(test_image):
    detected_face, rect = detect_face(test_image)
    resized_test_image = cv2.resize(detected_face, (120, 120), interpolation=cv2.INTER_AREA)
    label = eigenfaces_recognizer.predict(resized_test_image)
    label_text = tags[label[0]]
    draww_rectangle(test_image, rect)
    write_text(test_image, label_text, rect[0], rect[1] - 5)
    return test_image, label_text


training_data_path = 'images'

haarcascade_frontalface = '\haarscascade.xml'

tags = os.listdir(training_data_path)

detected_faces, face_labels = prepare_training_data(training_data_path)

print("total faces = ", len(detected_faces))
print("total labels = ", len(face_labels))

eigenfaces_recognizer = cv2.face.EigenFaceRecognizer.create()
eigenfaces_recognizer.train(detected_faces, np.array(face_labels))
n = input("Please type how many pictures")
i = 0
test_images = []
while(i<int(n)):
    test_image = input("Type the picture name/file path")
    test_image = test_image+".jpg"
    try:
        file =open(test_image,"r")
        file.close()
    except FileNotFoundError:
        print("File does not exist, type another")
        continue
    i+=1
    test_images.append(test_image)
i=0
while(i<int(n)):
    test_image = cv2.imread(test_images[i])
    pred_image, pred_label = predict(test_image)

    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax1.set_title( 'predicted class: ' + pred_label)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB))
    plt.show()
    i+=1
