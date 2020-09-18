import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

cap = cv2.VideoCapture("http://192.168.1.148:8080/video")

while True:
    ret, frame = cap.read()

    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, label, conf)

    cv2.imshow("frame", output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()