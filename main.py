import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cam = "Uav_Video.mp4"

cap = cv2.VideoCapture(cam)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("Uav_Output.avi", fourcc, 60.0, (640, 360))

def cam_setting(x):
    width = 640
    height = 360
    size = (width, height)
    return cv2.resize(x, size)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cam_setting(frame)
    results = model(
        frame,
        conf=0.7,
    )
    res_plotted = results[0].plot(
        conf=False,
        line_width=2,
        font_size=10,
        font="Arial.ttf",
        pil=True,
        img=None,
        im_gpu=True,
        kpt_radius=3,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
    )
    res_plotted = np.array(res_plotted)
    for r in results:
        c = r.boxes.xyxy.tolist()
        print(c)
        if c == []:
            pass
        else:
            cv2.line(
                res_plotted,
                (320, 180),
                (
                    int((int(c[0][2]) - int(c[0][0])) / 2 + int(c[0][0])),
                    int((int(c[0][3]) - int(c[0][1])) / 2 + int(c[0][1])),
                ),
                (0, 0, 255),
                1,
            )

    cv2.line(res_plotted, (320, 170), (320, 190), (0, 0, 255), 1)
    cv2.line(res_plotted, (310, 180), (330, 180), (0, 0, 255), 1)
    cv2.line(res_plotted, (360, 220), (360, 200), (0, 0, 255), 1)
    cv2.line(res_plotted, (360, 220), (340, 220), (0, 0, 255), 1)
    cv2.line(res_plotted, (280, 220), (280, 200), (0, 0, 255), 1)
    cv2.line(res_plotted, (280, 220), (300, 220), (0, 0, 255), 1)
    cv2.line(res_plotted, (280, 140), (280, 160), (0, 0, 255), 1)
    cv2.line(res_plotted, (280, 140), (300, 140), (0, 0, 255), 1)
    cv2.line(res_plotted, (360, 140), (360, 160), (0, 0, 255), 1)
    cv2.line(res_plotted, (360, 140), (340, 140), (0, 0, 255), 1)

    cv2.imshow("res_plotted", res_plotted)
    out.write(res_plotted)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()