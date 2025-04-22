import math
import cv2
import time
import csv
from ultralytics import YOLO

# Cálculo de distâncias com trigonometria
def calculo_distancias(A, h, alfa):
    alfa_radianos = math.radians(alfa)
    Distancia = (A - (h / 2)) / math.cos(alfa_radianos)
    return Distancia

# Função para detetar pessoas usando o modelo
# Obter centro e tamanho da pessoa na imagem 
def detecao_pessoas(frame, yolo_model): 
    results = yolo_model(frame)

    person_box = None  # Apenas uma caixa
    dados = None

    for r in results:
        boxes = r.boxes.xyxy.cpu()
        classes = r.boxes.cls.cpu()
        confidences = r.boxes.conf.cpu()

        for box, cls, conf in zip(boxes, classes, confidences):
            if r.names[int(cls)] == "person" and conf > 0.6:
                x1, y1, x2, y2 = map(int, box[:4])
                person_box = [x1, y1, x2, y2]
                break  # Encontrou a primeira pessoa
        
        annotated_frame = r.plot()
        cv2.imshow("Result", annotated_frame)

        if person_box:
            x1, y1, x2, y2 = person_box
            centrox = (x1 + x2) / 2
            centroy = (y1 + y2) / 2
            altura = y2 - y1
            dados = [centrox, centroy, altura]
            break  # Já encontrou uma pessoa, não precisa continuar

    return dados

def ficheiro_csv(alt, centrox, centroy, Distancia):
    with open("distancia.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([alt, centrox, centroy, Distancia])

if __name__ == "__main__":
    h = 1.70  # altura da pessoa média
    alfa = 25  # ângulo da câmara
    ult_dis = 0
    intervalo = 10  # segundos entre deteções
    yolo_model = YOLO('best.pt')

    # Abrir a câmara
    cap = cv2.VideoCapture(0)
    next_check = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        # Mostrar tempo até próxima deteção
        remaining = int(next_check - now)
        if remaining > 0:
            cv2.putText(frame, f"Proxima detecao em {remaining}s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if now >= next_check:
            try:
                with open("Altitudes.txt", 'r', encoding="utf-8") as f:
                    A_str = f.readline().strip()  # lê só a primeira linha válida
                    if A_str:
                        A = float(A_str)
                        dados = detecao_pessoas(frame, yolo_model)
                        if dados is not None:
                            Dist = calculo_distancias(A, h, alfa)
                            if abs(Dist - ult_dis) > 0.1:
                                ficheiro_csv(A, dados[0], dados[1], Dist)
                                ult_dis = Dist
            except Exception as e:
                print(f"Erro a ler altitude: {e}")

            next_check = now + intervalo

        cv2.imshow("Deteção", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
