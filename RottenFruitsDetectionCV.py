import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
from keras.models import load_model
import numpy as np

# model = load_model("ImageNetModel.h5")
model = load_model("AppleDetectModel")


def classify_rotten(image):
    result = model.predict(image)
    classification = np.argmax(result)

    if classification == 0:
        return "fresh"
    else:
        return "rotten"


cap = cv2.VideoCapture("http://192.168.1.148:8080/video")

fruit_labels = ['apple', 'orange', 'banana']

while True:
    ret, frame = cap.read()

    detected_box = []
    detected_conf = []
    detected_labels = []

    bbox, total_labels, conf = cv.detect_common_objects(frame)

    for l in fruit_labels:
        if l in total_labels:
            i = total_labels.index(l)
            detected_box.append(bbox[i])
            detected_conf.append(conf[i])
            detected_labels.append(l)

    output_image = draw_bbox(frame, detected_box, detected_labels, detected_conf)

    # frame_mod = cv2.resize(frame, (64, 64))
    # frame_mod = np.expand_dims(frame_mod, axis=0)

    # final_verdict = classify_rotten(frame_mod)

    # print(final_verdict)

    for i, fb in enumerate(detected_box):
        crop_img = frame[detected_box[i][0]: detected_box[i][0] + detected_box[i][2],
                              detected_box[i][1]: detected_box[i][1] + detected_box[i][3]]

        resized = cv2.resize(crop_img, (64, 64))
        reshaped = np.expand_dims(resized, axis=0)

        final_verdict = classify_rotten(reshaped)

        detected_conf[i] = int(detected_conf[i] * 100)

        cv2.putText(output_image, "Conf:{0}% Class:{1}".format(detected_conf[i], final_verdict),
                    (fb[0] - 20, fb[1] + 64), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 2)

    cv2.imshow("frame", output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
