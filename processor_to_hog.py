import dlib
import cv2
import os
import time
import numpy as np


def train_detector():
    # O dicionario ira guardar as imagens e a posição dos boxes.
    # Salvar o caminho da imagem
    # percorre todos os arquivos no directório e fatia a string pegando tudo antes do ponto
    data = {}
    path = 'train_images_h/'
    indexes = [int(img_name.split('.')[0]) for img_name in os.listdir(path)]
    np.random.shuffle(indexes)

    # Abrir e ler o arquivo
    # Convertendo os boxes para o dicionário
    f = open('boxes_h.txt', "r")
    boxes = f.read()
    box_dict = eval('{' + boxes + '}')
    f.close()

    # Percorrer todas as imagens
    # Ler a imagem do directorio de treino
    # Ler os boxes associados as imagens de acordo com o index
    # Converter os boxes compatível com o dlib
    # Salvar a imagem e as boxes
    for index in indexes:
        img = cv2.imread(os.path.join(path, str(index) + '.png'))
        bounding_box = box_dict[index]
        l, t, r, b = bounding_box
        dlib_box = [dlib.rectangle(left=l, top=t, right=r, bottom=b)]
        data[index] = (img, dlib_box)

    print('Numero de boxes e imagens associadas:', len(data))

    # O percentual de treino de 80%
    # Pegando a quantidade dos 80%
    percent = 0.8
    split = int(len(data) * percent)

    # Separar as imagens e caixas
    images = [tuple_value[0] for tuple_value in data.values()]
    bounding_boxes = [tuple_value[1] for tuple_value in data.values()]

    # Initialize object detector Options
    # Ignorar diferença de espelhamento
    # Valor 5 variável para cada caso
    options = dlib.simple_object_detector_training_options()
    options.add_left_right_image_flips = False
    options.C = 5

    # Executando o treinamento somente com os 80% da base capturada
    detector = dlib.train_simple_object_detector(images[:split], bounding_boxes[:split], options)

    name_detector = 'Helmet_Detector.svm'
    detector.save(name_detector)

    print("Metricas de treino: {}".format(
        dlib.test_simple_object_detector(images[:split], bounding_boxes[:split], detector)))
    print("Metricas de teste: {}".format(
        dlib.test_simple_object_detector(images[split:], bounding_boxes[split:], detector)))

#train_detector()

name_detector = 'Helmet_Detector.svm'
detector = dlib.simple_object_detector(name_detector)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
image = cv2.imread('emilly6.jpg')

detections = detector(image)

# Rodando o loop para cada detecção e desenhando na tela
for helmet in (detections):
    l, t, r, b = helmet.left(), helmet.top(), helmet.right(), helmet.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)
    cv2.putText(image, 'Helmet', (l, b + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow('frame', cv2.resize(image, (640, 360)))
cv2.waitKey(0)
cv2.destroyAllWindows()