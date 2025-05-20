from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
import pygame
import time
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Cảnh báo toàn cục
global latest_warning
latest_warning = ""
lock = threading.Lock()

# Khởi tạo âm thanh
pygame.init()
chopmat_sound = pygame.mixer.Sound("D:/AI/py/Sound/beep.wav")
ngap_sound = pygame.mixer.Sound("D:/AI/py/yawn_alert.wav")
phone_baodong = pygame.mixer.Sound("D:/AI/py/Sound/not_phone.wav")
seatbelt_baodong = pygame.mixer.Sound("D:/AI/py/Sound/seatbelt_alert.wav")

# Dlib & YOLO models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/AI/py/shape_predictor_68_face_landmarks.dat")
phone_mau = YOLO("yolov8n.pt")
seatbelt_mau = YOLO("D:/AI/py/weights/last.pt")

# Facial landmarks
left_eye_indexes = [36, 37, 38, 39, 40, 41]
right_eye_indexes = [42, 43, 44, 45, 46, 47]
mouth_indexes = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

# Thresholds
EAR_THRESHOLD = 0.30
EAR_MIN_DURATION = 2
YAWN_THRESHOLD = 25
YAWN_CONSEC_FRAMES = 15

# Mô hình đầu để xác định hướng đầu
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype="double")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def detect_yawn(shape_points):
    top_lip = [shape_points[i][1] for i in [50, 51, 52, 61, 62]]
    bottom_lip = [shape_points[i][1] for i in [56, 57, 58, 65, 66]]
    return np.mean(bottom_lip) - np.mean(top_lip)

def get_head_pose(shape_points, frame_size):
    image_points = np.array([
        shape_points[30], shape_points[8], shape_points[36],
        shape_points[45], shape_points[48], shape_points[54],
    ], dtype="double")

    focal_length = frame_size[1]
    center = (focal_length / 2, frame_size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    return [float(x) for x in eulerAngles]

def generate_frames():
    global latest_warning
    cap = cv2.VideoCapture(0)
    eye_closed_time = None
    counter_yawn = 0
    alarm_yawn_on = False
    seatbelt_alert_playing = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.dtype != 'uint8':
            gray = gray.astype('uint8')

        try:
            faces = detector(gray)
        except Exception as e:
            print(f"[Lỗi detector]: {e}")
            faces = []


        warning_text = ""

        for face in faces:
            shape = predictor(gray, face)
            points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]


            # Vẽ landmark mắt và miệng
            for idx in left_eye_indexes + right_eye_indexes + mouth_indexes:
                x, y = points[idx]
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Tính EAR
            left_eye = [points[i] for i in left_eye_indexes]
            right_eye = [points[i] for i in right_eye_indexes]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            if ear < EAR_THRESHOLD:
                if eye_closed_time is None:
                    eye_closed_time = time.time()
                elif time.time() - eye_closed_time > EAR_MIN_DURATION:
                    chopmat_sound.play()
                    warning_text = "NHẮM MẮT QUÁ LÂU!"
            else:
                eye_closed_time = None

            # Ngáp
            yawn_ratio = detect_yawn(points)
            if yawn_ratio > YAWN_THRESHOLD:
                counter_yawn += 1
                if counter_yawn >= YAWN_CONSEC_FRAMES and not alarm_yawn_on:
                    alarm_yawn_on = True
                    ngap_sound.play()
                    warning_text = "NGÁP NGỦ!"
            else:
                counter_yawn = 0
                alarm_yawn_on = False

            # Hướng đầu
            pitch, yaw, roll = get_head_pose(points, frame.shape)
            if abs(yaw) > 35 or pitch > 25:
                warning_text = "ĐẦU ĐẢO DỮ DỘI!"

        # YOLO - điện thoại
        results = phone_mau(frame)
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = phone_mau.names[int(cls)]
                if "phone" in label.lower() and conf > 0.5:
                    phone_baodong.play()
                    warning_text = "DÙNG ĐIỆN THOẠI!"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # YOLO - dây an toàn
        seatbelt_detected = False
        seatbelt_results = seatbelt_mau.predict(source=frame, stream=False)
        for result in seatbelt_results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = result.names[int(cls)]
                if "seat_belt" in label.lower() and conf > 0.5:
                    seatbelt_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not seatbelt_detected:
            if not seatbelt_alert_playing:
                seatbelt_baodong.play()
                seatbelt_alert_playing = True
            warning_text = "KHÔNG ĐEO DÂY AN TOÀN!"
        else:
            seatbelt_alert_playing = False

        # Hiển thị cảnh báo lên khung hình
        if warning_text:
            cv2.putText(frame, warning_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Ghi lại cảnh báo
        with lock:
            latest_warning = warning_text

        frame = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_warning')
def get_warning():
    with lock:
        return latest_warning

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False, host="0.0.0.0", port=8000)
