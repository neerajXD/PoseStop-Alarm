import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from flask import Flask, render_template, Response, request
import pygame
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Initialize Mediapipe and OpenCV instances
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # Alarm sound file

# Initialize Kalman filter
kalman_filters = {}

def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3
    return kalman

def kalman_filter_update(kalman, x, y):
    measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kalman.correct(measurement)
    prediction = kalman.predict()
    return [prediction[0, 0], prediction[1, 0]]

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B (the vertex)
    c = np.array(c)  # Point C

    ba = a - b  # Vector from B to A
    bc = c - b  # Vector from B to C

    # Calculate the angle using arctan2 to get the angle between vectors
    radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle = np.abs(radians * 180.0 / np.pi)  # Convert radians to degrees

    if angle > 180.0:  # Angle correction
        angle = 360 - angle
    
    return angle

# Timer logic for playing alarm at a specified time
alarm_time = None
alarm_playing = False
right_counter = 0
left_counter = 0
right_stage = "down"
left_stage = "down"

def check_alarm_time():
    global alarm_time, alarm_playing, right_counter, left_counter
    while True:
        if alarm_time and datetime.now().strftime('%H:%M') == alarm_time:
            alarm_playing = True
            pygame.mixer.music.play(-1)  # Start alarm when the time matches
            right_counter = 0  # Reset counters when alarm starts
            left_counter = 0
            print("Alarm triggered!")
            break
        time.sleep(60)  # Check every minute

# Start alarm time checking in a separate thread
alarm_thread = threading.Thread(target=check_alarm_time, daemon=True)
alarm_thread.start()

# Generate frames from the webcam feed for pose detection
def gen_frames():
    global right_counter, left_counter, right_stage, left_stage, alarm_playing
    
    cap = cv2.VideoCapture(1)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    for idx, landmark in enumerate(landmarks):
                        if idx not in kalman_filters:
                            kalman_filters[idx] = initialize_kalman_filter()

                    def get_filtered_landmark(landmark_index):
                        landmark = landmarks[landmark_index]
                        filtered = kalman_filter_update(kalman_filters[landmark_index], landmark.x, landmark.y)
                        return filtered

                    # Get coordinates for important joints
                    right_hip = get_filtered_landmark(mp_pose.PoseLandmark.RIGHT_HIP.value)
                    right_shoulder = get_filtered_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                    right_elbow = get_filtered_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                    left_hip = get_filtered_landmark(mp_pose.PoseLandmark.LEFT_HIP.value)
                    left_shoulder = get_filtered_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                    left_elbow = get_filtered_landmark(mp_pose.PoseLandmark.LEFT_ELBOW.value)

                    # Calculate angles for right and left arms
                    right_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                    left_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

                    # Update the frame with calculated angles and counters
                    cv2.putText(image, f'R: {int(right_angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image, f'L: {int(left_angle)}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image, f'Right Count: {right_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(image, f'Left Count: {left_counter}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Counting logic
                    if right_angle > 90 and right_stage == "down":
                        right_counter += 1
                        right_stage = "up"
                    elif right_angle < 45:
                        right_stage = "down"

                    if left_angle > 90 and left_stage == "down":
                        left_counter += 1
                        left_stage = "up"
                    elif left_angle < 45:
                        left_stage = "down"

                    # Check if we have reached 10 "ups" for both arms
                    if right_counter >= 10 and left_counter >= 10:
                        if alarm_playing:
                            pygame.mixer.music.stop()  # Stop the alarm sound
                            alarm_playing = False
                        print("10 ups detected! Alarm stopped.")

                except Exception as e:
                    print(e)

            # Draw the landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to stream the video feed
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to set alarm time
@app.route('/set_time', methods=['POST'])
def set_time():
    global alarm_time, right_counter, left_counter, alarm_playing
    alarm_time = request.form['time']
    right_counter = 0  # Reset counters when new time is set
    left_counter = 0
    alarm_playing = False  # Stop the alarm if a new alarm is set
    print(f"Alarm time set to {alarm_time}")
    return render_template('index.html', message=f"Alarm time set to {alarm_time}.")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
