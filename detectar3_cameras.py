import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import multiprocessing

def run_detection(camera_url):
    # Carrega o modelo YOLO
    model = YOLO("yolov8n.pt")

    # Configuração da câmera IP
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print(f"Não foi possível abrir a câmera {camera_url}")
        exit()

    track_history = defaultdict(lambda: [])
    seguir = True
    deixar_rastro = True

    while True:
        # Lê um frame da câmera IP
        success, img = cap.read()

        if success:
            # Realiza a detecção de objetos na imagem
            if seguir:
                results = model.track(img, persist=True)
            else:
                results = model(img)

            # Processa os resultados da detecção
            for result in results:
                # Visualiza os resultados na imagem
                img = result.plot()

                if seguir and deixar_rastro:
                    try:
                        # Obtém as caixas delimitadoras e IDs de rastreamento
                        boxes = result.boxes.xywh.cpu()
                        track_ids = result.boxes.id.int().cpu().tolist()

                        # Desenha as linhas de rastreamento para cada objeto detectado
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box
                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # Ponto central (x, y)
                            if len(track) > 30:  # Mantém um histórico de 30 pontos (90 frames)
                                track.pop(0)

                            # Desenha as linhas de rastreamento no frame
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                    except:
                        pass

            # Exibe o frame com as detecções na tela
            cv2.imshow("Tela", img)

        # Aguarda uma tecla ser pressionada
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()
    print(f"Desligando {camera_url}")

if __name__ == "__main__":
    # Configuração das URLs das câmeras
    camera_urls = [
        "rtsp://admin:andrey123@192.168.100.48:554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://admin:andrey123@192.168.100.47:554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://admin:andrey123@192.168.100.49:554/cam/realmonitor?channel=1&subtype=0"
    ]

    # Inicia os processos para cada câmera
    processes = []
    for url in camera_urls:
        process = multiprocessing.Process(target=run_detection, args=(url,))
        processes.append(process)
        process.start()

    # Aguarda até que todos os processos terminem
    for process in processes:
        process.join()
