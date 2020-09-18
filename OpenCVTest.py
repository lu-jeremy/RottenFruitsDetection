import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

model = load_model('small_last4.h5')


def classify(result):
    class_identifier = None
    classification = np.argmax(result)

    if classification == 0:
        class_identifier = "freshapples"
    elif classification == 1:
        class_identifier = "freshbanana"
    elif classification == 2:
        class_identifier = "freshorange"
    elif classification == 3:
        class_identifier = "rottenapple"
    elif classification == 4:
        class_identifier = "rottenbanana"
    elif classification == 5:
        class_identifier = "rottenorange"

    return class_identifier


# opencv
cap = cv2.VideoCapture("http://192.168.1.148:8080/video")

# fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

    # kernel = np.ones((5, 5), np.uint8)
    #
    # fgmask = fgbg.apply(frame)
    #
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    # cnt = max(contours, key=cv2.contourArea)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 191, 0), 4)


    # epsilon = 0.0005 * cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    #
    # hull = cv2.convexHull(cnt)
    #
    # cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
    #
    # cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    # crop_img = frame[y: y + h, x: x + w]
    #
    # resized = cv2.resize(crop_img, (64, 64))
    # resized2 = resized[:, :, 0]
    # reshaped = np.expand_dims(resized2, axis=0)
    #
    # result = model.predict(reshaped)
    #
    # print(result)
    #
    # classification = classify(result)

    # cv2.putText(frame, "Prediction : {}".format(classification),
    #             (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()