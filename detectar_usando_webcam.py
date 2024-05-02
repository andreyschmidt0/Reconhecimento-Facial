from ultralytics import YOLO
import cv2
from collections import defaultdict
from winotify import Notification
import time
import numpy as np

# Inicializa a captura de vídeo da câmera
cap = cv2.VideoCapture(0)

# Carrega o modelo YOLO
model = YOLO("runs/detect/train9/weights/best.pt")

# Variável para rastrear o histórico de rastreamento
track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

# Variáveis para controlar a exibição da notificação
last_notification_time = 0
notification_interval = 30

# Cria uma instância de notificação
notificacao = Notification(app_id="Código Python", title="Notificação da Automação", msg="Andrey foi detectado!", icon=r"C://Users/Andrey/Downloads/unnamed.png", duration="long")

while True:
    success, img = cap.read()

    if success:
        if seguir:
            results = model.track(img, persist=True)
        else:
            results = model(img)

        # Processa a lista de resultados
        for result in results:
            # Obtém o texto do relatório
            report_text = result.names[0]  # Supondo que "Andrey" é o primeiro nome no relatório

            # Verifica se "Andrey" foi detectado e se já passou tempo suficiente desde a última notificação
            current_time = time.time()
            if "Andrey" in report_text and current_time - last_notification_time >= notification_interval:
                notificacao.show()
                last_notification_time = current_time

            # Visualiza os resultados no quadro
            img = result.plot()

            if seguir and deixar_rastro:
                try:
                    # Obtém as caixas e IDs de rastreamento
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Desenha as trilhas de rastreamento
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # Ponto central (x, y)
                        if len(track) > 30:  # Mantém um histórico de 30 pontos (90 frames)
                            track.pop(0)

                        # Desenha as linhas de rastreamento no quadro
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                except:
                    pass

        # Exibe o quadro com as detecções na tela
        cv2.imshow("Tela", img)

    # Aguarda uma tecla ser pressionada
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
print("Desligando")
