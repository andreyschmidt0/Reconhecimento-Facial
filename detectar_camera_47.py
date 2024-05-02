import cv2
from ultralytics import YOLO
from winotify import Notification

# Carrega o modelo YOLO
model = YOLO("yolov8n.pt")

# Configuração da câmera IP
camera_url = "rtsp://admin:andrey123@192.168.100.47:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Não foi possível abrir a câmera")
    exit()

# Captura de frame inicial
ret, frame_old = cap.read()

# Limiar de movimento
threshold = 10000

# Flag para indicar se há movimento
moving = False

#Com a biblioteca winotify é possivel exibir uma notificação ao executar uma tarefa.. (Notifica que a aplicação foi inicializada)
notificacao = Notification(app_id="Código Python", title="Notificação da Automação", msg="Captura Inicializada", icon=r"C://Users/Andrey/Downloads/unnamed.png", duration="long")
notificacao.show()

while True:
    # Lê um frame da câmera IP
    ret, frame = cap.read()

    if not ret:
        break

    # Calcula a diferença absoluta entre os frames
    diff = cv2.absdiff(frame, frame_old)
    motion = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))

    # Verifica se o movimento excede o limiar
    if motion > threshold:
        moving = True

    if moving:
        # Realiza a detecção de objetos na imagem
        results = model(frame)

        # Processa os resultados da detecção
        for result in results:
            # Visualiza os resultados na imagem
            img = result.plot()

            # Exibe o frame com as detecções na tela
            cv2.imshow("Tela", img)

    # Atualiza o frame anterior
    frame_old = frame.copy()

    # Aguarda uma tecla ser pressionada
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
print("Desligando")

notificacao = Notification(app_id="Código Python", title="Notificação da Automação", msg="Captura Desligada", icon=r"C://Users/Andrey/Downloads/unnamed.png", duration="long")
notificacao.show()
