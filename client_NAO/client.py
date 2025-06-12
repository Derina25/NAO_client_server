# -*- coding: utf-8 -*-
import socket
import struct
import traceback
import uuid

from uuid import uuid4
import numpy as np
import cv2
from PIL import Image
import io
import vision_definitions
import time
from naoqi import ALProxy

NAO_IP = "169.254.248.149"
NAO_PORT = 9559
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8080
SOCKET_TIMEOUT = 30

CAMERA_NAME = uuid.uuid4().hex

def connect_to_nao():
    """
    Si connette al robot NAO e si iscrive al servizio video per acquisire immagini dalla telecamera.

    Restituisce:
        video_client: identificatore della sottoscrizione al servizio video.
        video_service: oggetto proxy per il servizio video di NAO.
    """

    try:
        video_service = ALProxy("ALVideoDevice", NAO_IP, 9559)
        resolution = vision_definitions.kVGA
        color_space = vision_definitions.kRGBColorSpace
        fps = 10
        video_client = video_service.subscribeCamera(CAMERA_NAME, 0, resolution, color_space, fps)
        return video_client, video_service
    except Exception as e:
        print("Errore connessione NAO:", e)
        return None, None

def get_nao_frame(video_client, video_service):
    """
    Acquisisce un frame dalla telecamera del NAO.

    Parametri:
        video_client: identificatore della sottoscrizione al servizio video.
        video_service: oggetto proxy per il servizio video di NAO.

    Restituisce:
        frame (ndarray): immagine acquisita dalla telecamera, oppure None in caso di errore.
    """
    try:
        nao_image = video_service.getImageRemote(video_client)
        if nao_image:
            width, height = nao_image[0], nao_image[1]
            image_data = nao_image[6]
            nparray = np.frombuffer(image_data, np.uint8).reshape((height, width, 3))
            return nparray
        else:
            return None
    except Exception:
        traceback.print_exc()

def send_frame_to_server(frame):
    """
    Invia un frame al server per l'elaborazione e riceve la risposta.

    Parametri:
        frame (ndarray): immagine da inviare al server.

    Restituisce:
        response (str): risposta del server in formato JSON, oppure None in caso di errore.
    """
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        data = buffer.getvalue()
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(SOCKET_TIMEOUT)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        client_socket.sendall(struct.pack("I", len(data)))
        client_socket.sendall(data)
        resp_size_data = client_socket.recv(4)
        if not resp_size_data or len(resp_size_data) != 4:
            client_socket.close()
            return None
        response_size = struct.unpack("I", resp_size_data)[0]
        response_data = b""
        while len(response_data) < response_size:
            chunk = client_socket.recv(min(4096, response_size - len(response_data)))
            if not chunk:
                break
            response_data += chunk
        client_socket.close()
        return response_data.decode('utf-8')
    except Exception:
        return None

def nao_move_head(coordinates, video_service, img_width=None, img_height=None):
    try:
        motion = ALProxy("ALMotion", NAO_IP, NAO_PORT)

        if img_width is None or img_height is None:
            resolution = video_service.getParameter("current_resolution")
            img_width = resolution[0]
            img_height = resolution[1]

        x1, y1, x2, y2 = coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        norm_x = center_x / img_width
        norm_y = center_y / img_height

        yaw = (norm_x - 0.5) * 2
        pitch = (0.5 - norm_y) * 2

        max_yaw, max_pitch = 1.0, 0.5
        yaw = max(-max_yaw, min(yaw, max_yaw))
        pitch = max(-max_pitch, min(pitch, max_pitch))

        motion.setAngles(["HeadYaw", "HeadPitch"], [yaw, pitch], 0.2)
    except Exception as e:
        print(e)

def main():
    """
    Funzione principale: gestisce il ciclo di acquisizione immagini, invio al server, ricezione risposta e movimento della testa del NAO.
    """
    import json

    video_client, video_service = connect_to_nao()
    try:
        if not video_client or not video_service:
            print("Errore: Impossibile connettersi al robot NAO.")
            if not video_client:
                print("Errore su video client")
            if not video_service:
                print("Errore su video service")
            return
        try:
            resolution = video_service.getParameter("current_resolution")
            img_width, img_height = resolution[0], resolution[1]
        except Exception:
            img_width, img_height = 160, 120

        while True:

            frame = get_nao_frame(video_client, video_service)

            if frame is None:
                time.sleep(0.2)
                continue

            server_response = send_frame_to_server(frame)

            if server_response:
                try:
                    response_dict = json.loads(server_response)
                    message = response_dict.get("message", "")
                    coordinates = response_dict.get("coordinates", None)
                    if "Cat detected" in message and coordinates and len(coordinates) == 4:
                        nao_move_head(coordinates, video_service, img_width, img_height)
                except Exception:
                    pass
            cv2.imshow("NAO - Tracciamento Gatto", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
    except Exception:
        traceback.print_exc()
        if video_service and video_client:
            video_service.unsubscribe(video_client)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()