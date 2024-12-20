import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading
import json
import paho.mqtt.client as mqtt
import time

# MQTT Broker Configuration
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "video/file/path"
mqtt_video_path = None  # Global variable to store video path from MQTT


def load_json(file_name):
    with open(file_name, "r") as file:
        return json.load(file)


def save_json(file_name, data):
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)


def reset_json_mail():
    entries = load_json("license_plate_details2.json")
    for entry in entries:
        entry["email_status"] = "False"
    save_json("license_plate_details2.json", entries)
    print("Email status reset completed.")


def run_reset_mail_periodically(interval):
    while True:
        reset_json_mail()
        time.sleep(interval)


def start_reset_mail_thread():
    thread = threading.Thread(
        target=run_reset_mail_periodically, args=(5,), daemon=True
    )
    thread.start()
    print("Reset mail thread started.")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")


def on_message(client, userdata, msg):
    global mqtt_video_path, detection_active
    mqtt_video_path = msg.payload.decode()  # Decode the file path
    detection_active = False  # Signal to stop the current detection loop
    print(f"Received video path from MQTT: {mqtt_video_path}")


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def extract_email_and_send_email():
    sender_email = "khansafgan1743@gmail.com"
    sender_password = "vyve qrzv wypd crwc"
    subject = "Litter Detection"

    with open("currentLicense.txt", "r") as file:
        license_plate_text = file.read()

    print("Extracted License Number: ", license_plate_text)

    entries = load_json("license_plate_details2.json")
    for entry in entries:
        print("Entry License Number: ", entry["license_plate"].lower())
        if entry["license_plate"].lower() in license_plate_text.lower():
            if entry["email_status"] == "False":
                receiver_email = entry["email_id"]
                number_of_times = entry["number_of_times_detected"]
                number_of_times += 1
                entry["number_of_times_detected"] = number_of_times
                message_body = f"Litter detected. Vehicle Number: {license_plate_text}. {number_of_times} number of times You are fined with Rs. {number_of_times*1000}."
                send_gmail_threaded(
                    sender_email, sender_password, receiver_email, subject, message_body
                )
                entry["email_status"] = "True"

    with open("license_plate_details2.json", "w") as json_file:
        json.dump(entries, json_file, indent=4)


def send_gmail(sender_email, sender_password, receiver_email, subject, message_body):
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message_body, "plain"))
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


def send_gmail_threaded(
    sender_email, sender_password, receiver_email, subject, message_body
):
    threading.Thread(
        target=send_gmail,
        args=(sender_email, sender_password, receiver_email, subject, message_body),
    ).start()


def initialize_models():
    return (
        YOLO("models/vehicleDetectionModel.pt"),
        YOLO("models/license_plate_detector.pt"),
        YOLO("models/litterDetectionModel.pt"),
    )


def detectLitter(model, frame):
    litterDetected = False
    results = model.predict(source=frame, conf=0.25, device="cpu")

    boxes = results[0].boxes.xywh.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]

    for i, (box, confidence) in enumerate(zip(boxes, confidences)):
        x, y, w, h = map(int, box)
        label = labels[i]
        label_text = f"{label} {confidence:.2f}"

        cv2.rectangle(
            frame,
            (x - w // 2, y - h // 2),
            (x + w // 2, y + h // 2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            label_text,
            (x - w // 2, y - h // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        litterDetected = True

    return frame, litterDetected


def detect_and_track_vehicles(frame, coco_model, mot_tracker, vehicles):
    detections = coco_model(frame)[0]
    vehicle_detections = [
        [x1, y1, x2, y2, score]
        for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist()
        if int(class_id) in vehicles
    ]
    return mot_tracker.update(np.asarray(vehicle_detections))


def process_license_plate(frame, license_plate, track_ids):
    result = {"car_id": None, "license_plate_text": None}
    x1, y1, x2, y2, _, _ = license_plate
    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

    if car_id != -1:
        crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, crop_thresh = cv2.threshold(crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        license_text, _ = read_license_plate(crop_thresh)

        result.update({"car_id": car_id, "license_plate_text": license_text})
        print("License Plate number is:", license_text)

        cv2.rectangle(
            frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (255, 0, 0), 5
        )
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
        if license_text:
            cv2.putText(
                frame,
                license_text,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            with open("currentLicense.txt", "w") as file:
                file.write(license_text)

    return result, frame


def detect_car_and_license_plate(
    frame, coco_model, license_plate_detector, mot_tracker, vehicles
):
    track_ids = detect_and_track_vehicles(frame, coco_model, mot_tracker, vehicles)
    license_plate_detections = license_plate_detector(frame)[0]

    for license_plate in license_plate_detections.boxes.data.tolist():
        result, frame = process_license_plate(frame, license_plate, track_ids)
        if result["car_id"] and result["license_plate_text"]:
            return frame, True
    return frame, False


def main():
    global mqtt_video_path, detection_active, cap
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)

    # Start MQTT client in a background thread
    client.loop_start()

    coco_model, license_plate_detector, litter_detection_model = initialize_models()
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]

    while True:
        if mqtt_video_path is None:
            print("Waiting for video path from MQTT...")
            time.sleep(1)
            continue

        print(f"Starting detection for: {mqtt_video_path}")
        cap = cv2.VideoCapture(mqtt_video_path)
        detection_active = True  # Indicate the detection loop is active

        litter_frame_counter = 0
        litterCount = 0
        while detection_active:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            result_frame, detection_found = detect_car_and_license_plate(
                frame, coco_model, license_plate_detector, mot_tracker, vehicles
            )
            result_frame, litter_detected = detectLitter(
                litter_detection_model, result_frame
            )

            with open("currentLicense.txt", "r") as file:
                license_plate_text = file.read()

            if litter_detected and detection_found:
                litterCount += 1
                if litterCount > -1:
                    extract_email_and_send_email()
                    litter_frame_counter = (
                        5  # Set the counter to 5 frames if litter is detected
                    )
            elif not litter_detected:
                litterCount = 0

            elif litter_frame_counter > 0:
                litter_frame_counter -= 1
            else:
                cv2.putText(
                    result_frame,
                    "Normal",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            if litter_frame_counter > 0:
                cv2.putText(
                    result_frame,
                    "Litter Detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Vehicle and License Plate Detection", result_frame)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                cv2.destroyWindow("Vehicle and License Plate Detection")
                print("User terminated the current detection.")
                detection_active = False
                break

        cap.release()
        cv2.destroyAllWindows()
        mqtt_video_path = None  # Reset path to wait for a new one

    client.loop_stop()


if __name__ == "__main__":
    start_reset_mail_thread()
    cap = None
    detection_active = False
    main()
