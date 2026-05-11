import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 8000)

sx_smooth = 0.0
sy_smooth = 0.0


smooth = 0.9


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    ret, frame = cap.read()
    h, w, c = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:

            x1 = int(hand[4].x * w)
            y1 = int(hand[4].y * h)


            x2 = int(hand[8].x * w)
            y2 = int(hand[8].y * h)

            cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            distance = math.hypot(x2 - x1, y2 - y1)
            norm_dist = max(0, min(1.0, distance / 500))

            client.send_message("/pinch", norm_dist)


            sx = -((((x1 + x2) / 2) / w) * 10 - 5)
            sy = -((((y1 + y2) / 2) / h) * 10 - 5)

            
            sx_smooth = sx_smooth * smooth + sx * (1 - smooth)
            sy_smooth = sy_smooth * smooth + sy * (1 - smooth)

            
            if norm_dist < 0.08:
                client.send_message("/obj/tx", sx_smooth)
                client.send_message("/obj/ty", sy_smooth)
                client.send_message("/switch", 1)
            else:
                client.send_message("/switch", 0)

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()