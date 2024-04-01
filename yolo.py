import cv2
import torch

# Carregamento do modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Caminho da imagem
image_path = '/home/alcirheber/Imagens/chutando.jpg'

# Carregar a imagem usando OpenCV
image_cv2 = cv2.imread(image_path)

# Obter as dimensões da imagem
altura, largura, _ = image_cv2.shape

# Realizar inferência na imagem
results = model(image_cv2, size=640)

# Acessar as coordenadas das bounding boxes, confiança e classe para cada detecção
for detection in results.xyxy[0]:
    x1, y1, x2, y2, confidence, class_id = detection.tolist()
    
    # Normalizar as coordenadas
    x1_normalizado = x1 / largura
    y1_normalizado = y1 / altura
    w_normalizado = (x2 - x1) / largura
    h_normalizado = (y2 - y1) / altura
    
    # Imprimir as coordenadas normalizadas das bounding boxes no terminal
    print(f'Bounding Box Normalizada: (x1: {x1_normalizado:.2f}, y1: {y1_normalizado:.2f}, largura: {w_normalizado:.2f}, altura: {h_normalizado:.2f}), Confiança: {confidence:.2f}, ID da Classe: {class_id}')

    # Desenhar a bounding box na imagem original
    cv2.rectangle(image_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Escrever o texto da classe e confiança
    label = f'Classe: {int(class_id)}, Confiança: {confidence:.2f}'
    cv2.putText(image_cv2, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Salvar a imagem com as bounding boxes desenhadas
output_image_path = '/home/alcirheber/ImagensBB/imagem_com_bounding_boxes_chutando.jpg'
cv2.imwrite(output_image_path, image_cv2)

print(f'Imagem salva em: {output_image_path}')
