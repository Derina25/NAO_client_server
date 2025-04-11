# -*- coding: utf-8 -*-
import qi  # Framework per interagire con il robot NAO
import socket
import struct
import numpy as np
import cv2
from PIL import Image
import io
import vision_definitions  # Definizioni specifiche per le videocamere di NAO
import time

# Configura l'indirizzo IP e la porta del robot NAO
NAO_IP = "nao01.local"  # Sostituisci con l'IP reale del robot NAO
NAO_PORT = 9559  # Porta predefinita per NAO

# Configura il server a cui inviare i frame di NAO
SERVER_IP = "0.0.0.0"  # Sostituisci con l'IP del server
SERVER_PORT = 6000  # Porta predefinita per il server
SOCKET_TIMEOUT = 30  # Timeout in secondi per le operazioni socket


def connect_to_nao():
    """
    Si connette al robot NAO e al servizio video 'ALVideoDevice'.
    Restituisce il video_client e il video_service.
    """
    try:
        # Avvia l'applicazione qi
        app = qi.Application(["NAO_Client"])
        app.start()

        # Connettiti alla sessione di NAO
        session = app.session
        session.connect('tcp://' + NAO_IP + ':' + str(NAO_PORT))

        # Ottieni il servizio video del robot
        video_service = session.service('ALVideoDevice')
        if not video_service:
            print("Errore: Servizio ALVideoDevice non disponibile.")
            return None, None

        # Configurazione videocamera
        resolution = vision_definitions.kQQVGA  # Risoluzione 160x120
        color_space = vision_definitions.kRGBColorSpace  # Colori RGB
        fps = 10  # Frame per secondo

        # Iscrizione al servizio videocamera
        video_client = video_service.subscribeCamera("python_GVM", 0, resolution, color_space, fps)
        print("Connessione al robot NAO completata con successo.")
        return video_client, video_service

    except RuntimeError as e:
        print("Errore nella connessione al robot NAO: {}".format(e))
        return None, None


def get_nao_frame(video_client, video_service):
    """
    Acquisisce un frame dalla videocamera del robot NAO.
    Restituisce il frame come immagine OpenCV.
    """
    try:
        # Ottiene l'immagine dalla videocamera
        nao_image = video_service.getImageRemote(video_client)
        if nao_image:
            # Dimensioni dell'immagine
            width, height = nao_image[0], nao_image[1]

            # Dati grezzi dell'immagine
            image_data = nao_image[6]

            # Conversione in array numpy
            nparray = np.frombuffer(image_data, np.uint8).reshape((height, width, 3))

            # In Python 2.7 non è necessario chiamare imdecode perché abbiamo già i dati raw
            # Semplicemente organizziamo i dati in un formato BGR per OpenCV
            frame = cv2.cvtColor(nparray, cv2.COLOR_RGB2BGR)

            print("Frame acquisito dal robot NAO.")
            return frame
        else:
            print("Errore: Nessun frame ricevuto dal robot.")
            return None
    except Exception as e:
        print("Errore durante l'acquisizione del frame da NAO: {}".format(e))
        return None


def send_frame_to_server(frame):
    """
    Invia il frame acquisito al server tramite socket.
    Restituisce la risposta del server.
    """
    try:
        # Converti il frame da OpenCV (BGR) a PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Usa un buffer per salvare il frame come immagine JPEG
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        data = buffer.getvalue()
        print("Immagine compressa: {} bytes".format(len(data)))

        # Connettiti al server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(SOCKET_TIMEOUT)  # Imposta un timeout per evitare blocchi
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print("Connesso al server {}:{}".format(SERVER_IP, SERVER_PORT))

        # Invia la lunghezza dell'immagine
        print("Invio dimensione dell'immagine: {} bytes".format(len(data)))
        client_socket.sendall(struct.pack("I", len(data)))

        # Invia i dati dell'immagine
        print("Invio dati dell'immagine...")
        client_socket.sendall(data)
        print("Dati inviati. In attesa di risposta...")

        # Ricevi la lunghezza della risposta
        resp_size_data = client_socket.recv(4)
        if not resp_size_data or len(resp_size_data) != 4:
            print("Errore: Risposta dal server incompleta o mancante")
            client_socket.close()
            return None

        response_size = struct.unpack("I", resp_size_data)[0]
        print("Dimensione risposta prevista: {} bytes".format(response_size))

        # Ricevi la risposta completa
        response_data = b""
        bytes_received = 0

        while bytes_received < response_size:
            chunk = client_socket.recv(min(4096, response_size - bytes_received))
            if not chunk:
                break
            response_data += chunk
            bytes_received += len(chunk)

        client_socket.close()

        # In Python 2.7, le stringhe sono byte di default, quindi non serve decodificare
        # Ma per chiarezza, convertiamo esplicitamente per garantire coerenza
        response_str = response_data.decode('utf-8') if hasattr(response_data, 'decode') else response_data

        print("Risposta ricevuta dal server: {}".format(response_str))
        return response_str

    except socket.timeout:
        print("Timeout durante la comunicazione con il server")
        return None
    except Exception as e:
        print("Errore nell'invio al server: {}".format(e))
        return None


def main():
    """
    Funzione principale del client.
    Acquisisce i frame dalla videocamera di NAO e li invia al server.
    """
    video_client, video_service = connect_to_nao()
    if not video_client or not video_service:
        print("Errore: Impossibile connettersi al robot NAO. Esco...")
        return

    try:
        while True:
            # Acquisisci il frame dal robot NAO
            frame = get_nao_frame(video_client, video_service)
            if frame is None:
                print("Errore nella cattura del frame. Riprovando...")
                time.sleep(1)  # Breve pausa prima di riprovare
                continue

            # Mostra il frame (opzionale)
            cv2.imshow("Frame da NAO", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Premi 'q' per uscire dal loop
                break

            # Invia il frame al server
            print("Invio il frame al server...")
            server_response = send_frame_to_server(frame)
            if server_response:
                print("Risultato dal server: {}".format(server_response))
            else:
                print("Errore: Nessuna risposta dal server.")

            # Breve pausa tra un invio e l'altro
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterruzione manuale ricevuta. Esco...")

    finally:
        # Rilascia risorse
        if video_service and video_client:
            video_service.unsubscribe(video_client)
        cv2.destroyAllWindows()
        print("Connessione al robot e alle risorse terminata.")


if __name__ == "__main__":
    main()