import socket
import struct
import traceback

import numpy as np
import cv2
from PIL import Image
import io
import torch
from torchvision.transforms import v2
import json
from ultralytics import YOLO
from torchvision import tv_tensors

from modelCatStates import catPositionClassifier
from modelBodyLanguage import bodyLanguageClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolo11n.pt")

position_model = catPositionClassifier(num_classes=4).to(device)
position_model.eval()

position_model.load_state_dict(torch.load("../../../PycharmProjects/NAO_client_server/server_PC/cat_position_classifier.pth", map_location=device))

emotion_model = bodyLanguageClassifier(num_classes=6).to(device)
emotion_model.load_state_dict(torch.load("../../../PycharmProjects/NAO_client_server/server_PC/cat_emotion_classifier.pth", map_location=device))
emotion_model.eval()
position_class_mapping = {0: 'accovacciato', 1: 'alzato', 2: 'seduto', 3: 'steso'}
emotion_class_mapping = {0: 'aggressivo', 1: 'dorme', 2: 'felice', 3: 'giocoso', 4: 'mangiare', 5: 'spaventato'}

CONFIDENCE_THRESHOLD = 0.30
transform = v2.Compose([v2.Resize((640, 640)), v2.ToTensor()])

def process_image(image_data):
    try:

        image = Image.open(io.BytesIO(image_data))
        image_tensor = transform(image).unsqueeze(0)
        cat_detected = False
        coordinates = None
        results = yolo_model(image_tensor)

        if results[0].boxes is not None:

            for result in results:
                for box_idx, cls in enumerate(result.boxes.cls.cpu().numpy()):
                    class_name = result.names[int(cls)]
                    conf = float(result.boxes.conf[box_idx].cpu().numpy())
                    if "cat" in class_name and conf > 0.3:
                        cat_detected = True
                        if hasattr(result.boxes, 'xyxy') and len(result.boxes.xyxy) > box_idx:
                            coordinates = result.boxes.xyxy[box_idx].cpu().numpy().tolist()
                        break

        if cat_detected:
            position_model.eval()

            with torch.no_grad():
                position_tensor = image_tensor.to(device)
                position_outputs = position_model(position_tensor)
                position_probabilities = torch.nn.functional.softmax(position_outputs[0], dim=0)
            position_confidence = float(np.max(position_probabilities.cpu().numpy()))
            position_class = np.argmax(position_probabilities.cpu().numpy())
            position_class_name = position_class_mapping[position_class]
            position_reliable = position_confidence >= CONFIDENCE_THRESHOLD

            emotion_model.eval()

            with torch.no_grad():
                emotion_tensor = image_tensor.to(device)
                emotion_outputs = emotion_model(emotion_tensor)
                emotion_probabilities = torch.nn.functional.softmax(emotion_outputs[0], dim=0)
            emotion_confidence = float(np.max(emotion_probabilities.cpu().numpy()))
            emotion_class = np.argmax(emotion_probabilities.cpu().numpy())
            emotion_class_name = emotion_class_mapping[emotion_class]
            emotion_reliable = emotion_confidence >= CONFIDENCE_THRESHOLD

            response = "Cat detected"

            if coordinates:
                response += " at coordinates [{:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(
                    coordinates[0], coordinates[1], coordinates[2], coordinates[3])
            response += "\n"

            if position_reliable:
                response += "Position: {} ({:.2f})\n".format(position_class_name, position_confidence)

            if emotion_reliable:
                response += "Emotion: {} ({:.2f})\n".format(emotion_class_name, emotion_confidence)
        else:
            response = "No cat detected"
        return response, coordinates
    except Exception:
        return "Errore", None

@torch.inference_mode()
def process_frame(frame):
    """
    Elabora un frame OpenCV, rileva il gatto con YOLO, classifica posizione ed emozione.
    Restituisce un JSON con i risultati.
    """
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #pil_image = Image.fromarray(frame_rgb)
    frame_rgb = tv_tensors.Image(frame).permute(2,0,1)/255.0
    image_tensor = transform(frame_rgb).unsqueeze(0).to(device).float()

    data = {
        "emotion": None,
        "position": None,
        "coordinates": None
    }

    results = yolo_model(image_tensor)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box_idx, cls in enumerate(results[0].boxes.cls.cpu().numpy()):
            class_name = results[0].names[int(cls)]
            conf = float(results[0].boxes.conf[box_idx].cpu().numpy())

            if "cat" == class_name.lower() and conf > 0.3:
                coordinates = results[0].boxes.xyxy[box_idx].cpu().numpy().tolist()

                # Classificazione posizione
                position_outputs = position_model(image_tensor)
                position_probabilities = torch.nn.functional.softmax(position_outputs[0], dim=0)
                position_confidence = float(np.max(position_probabilities.cpu().numpy()))
                position_class = np.argmax(position_probabilities.cpu().numpy())
                position_class_name = position_class_mapping[int(position_class)]

                # Classificazione emozione

                emotion_outputs = emotion_model(image_tensor)
                emotion_probabilities = torch.nn.functional.softmax(emotion_outputs[0], dim=0)
                emotion_confidence = float(np.max(emotion_probabilities.cpu().numpy()))
                emotion_class = np.argmax(emotion_probabilities.cpu().numpy())
                emotion_class_name = emotion_class_mapping[int(emotion_class)]

                data = {
                    "emotion": emotion_class_name if emotion_confidence >= CONFIDENCE_THRESHOLD else None,
                    "position": position_class_name if position_confidence >= CONFIDENCE_THRESHOLD else None,
                    "coordinates": coordinates
                }
                break
    return json.dumps(data)

def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind(('127.0.0.1', 8080))
        server_socket.listen(5)
        print("Server pronto su 127.0.0.1:8080")

        while True:
            client_socket, _ = server_socket.accept()
            try:
                client_socket.settimeout(30)
                data_size_bytes = client_socket.recv(4)

                if not data_size_bytes or len(data_size_bytes) != 4:
                    continue
                data_size = struct.unpack("I", data_size_bytes)[0]
                image_data = b""

                while len(image_data) < data_size:
                    chunk = client_socket.recv(min(8192, data_size - len(image_data)))
                    if not chunk:
                        break
                    image_data += chunk

                if len(image_data) > 0:
    
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        response_json = process_frame(frame)
                    else:
                        response_json = json.dumps({"message": "Errore nella decodifica dell'immagine", "coordinates": None})
                    result_bytes = response_json.encode('utf-8')
                    client_socket.sendall(struct.pack("I", len(result_bytes)))
                    client_socket.sendall(result_bytes) 
            except Exception:
                traceback.print_exc()

            finally:
                client_socket.close()

    except KeyboardInterrupt:
        print("Server fermato")

    finally:
        server_socket.close()

if __name__ == "__main__":
    run_server()