import socket
import struct
import numpy as np
import cv2
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from modelCatStates import catPositionClassifier
from modelBodyLanguage import bodyLanguageClassifier

# Configurazioni come nel tuo codice originale
NAO_IP = "nao01.local"
NAO_PORT = 9559

# Carica i modelli (mantenuto dal tuo codice)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolo11n.pt")

# Aggiungi qui le definizioni dei modelli position_model e emotion_model come nel tuo codice originale
position_model = catPositionClassifier(num_classes=4).to(device)
position_model.load_state_dict(torch.load("../../../PycharmProjects/NAO_client_server/server_PC/cat_position_classifier.pth", map_location=device))
emotion_model = bodyLanguageClassifier(num_classes=6).to(device)
emotion_model.load_state_dict(torch.load("../../../PycharmProjects/NAO_client_server/server_PC/cat_emotion_classifier.pth", map_location=device))

position_class_mapping = {0: 'accovacciato', 1: 'alzato', 2: 'seduto', 3: 'steso'}
emotion_class_mapping = {0: 'aggressivo', 1: 'dorme', 2: 'felice', 3: 'giocoso', 4: 'mangiare', 5: 'spaventato'}


# Funzione per processare l'immagine ricevuta
def process_image(image_data):
    try:
        # Converti i dati dell'immagine in un formato utilizzabile
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        # Preprocessa l'immagine per i modelli
        transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)  # Aggiungi una dimensione batch

        cat_detected = False

        # Previsioni con YOLO
        results = yolo_model(image_tensor)

        if results[0].boxes is not None:
            for result in results:
                for cls in result.boxes.cls.cpu().numpy():
                    if "cat" in result.names[int(cls)]:
                        cat_detected = True

        if cat_detected:
            # Prediction con position_model
            position_model.eval()
            with torch.no_grad():
                position_tensor = image_tensor.to(device)
                position_outputs = position_model(position_tensor)
                position_probabilities = torch.nn.functional.softmax(position_outputs[0], dim=0)
            position_class = np.argmax(position_probabilities.cpu().numpy())
            position_class_name = position_class_mapping[position_class]

            # Prediction con emotion_model
            emotion_model.eval()
            with torch.no_grad():
                emotion_tensor = image_tensor.to(device)
                emotion_outputs = emotion_model(emotion_tensor)
                emotion_probabilities = torch.nn.functional.softmax(emotion_outputs[0], dim=0)
            emotion_class = np.argmax(emotion_probabilities.cpu().numpy())
            emotion_class_name = emotion_class_mapping[emotion_class]

            # Prepara la risposta
            response = f"Position: {position_class_name} ({float(np.max(position_probabilities.cpu().numpy())):.2f})\n" \
                       f"Emotion: {emotion_class_name} ({float(np.max(emotion_probabilities.cpu().numpy())):.2f})"
        else:
            response = "No cat detected"

        return response
    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine: {e}")
        return f"Errore: {str(e)}"


# Inizializza il server socket
def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 6000))  # IP e porta del server
    server_socket.listen(1)  # Ascolta per una connessione
    print("Server in ascolto su 0.0.0.0:6000")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Connessione stabilita con {client_address}")

            try:
                # Ricevi la dimensione dei dati dell'immagine usando struct.unpack
                data_size = struct.unpack("I", client_socket.recv(4))[0]
                print(f"Dimensione dati in arrivo: {data_size} bytes")

                # Ricevi i dati dell'immagine
                image_data = b""
                while len(image_data) < data_size:
                    packet = client_socket.recv(min(65536, data_size - len(image_data)))
                    if not packet:
                        break
                    image_data += packet

                print(f"Ricevuti {len(image_data)} bytes di dati")

                # Processa l'immagine
                result = process_image(image_data)

                # Invia il risultato al client
                result_bytes = result.encode('utf-8')
                client_socket.sendall(struct.pack("I", len(result_bytes)))
                client_socket.sendall(result_bytes)
                print(f"Inviata risposta: {result}")

            except Exception as e:
                print(f"Errore durante la comunicazione con il client: {e}")
            finally:
                client_socket.close()

    except KeyboardInterrupt:
        print("\nServer fermato manualmente")
    finally:
        server_socket.close()
        print("Server chiuso")


if __name__ == "__main__":
    run_server()