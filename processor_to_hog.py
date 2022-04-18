import dlib
import cv2
import os
import time
import numpy as np

# O dicionario ira guardar as imagens e a posição dos boxes.
data = {}
path = 'train_images_h/'

# Pegar todos os index das imagens
# percorre todos os arquivos no directório e fatia a string pegando tudo antes do ponto
indexes = [int(img_name.split('.')[0]) for img_name in os.listdir(path)]

# Colocar os indices das imagens aleatorios para diferenciar sempre que rodar o treinamento
np.random.shuffle(indexes)

# Abrir e ler o arquivo
f = open('boxes_h.txt', "r")
boxes = f.read()

# Convertendo os boxes para o dicionário
box_dict =  eval( '{' + boxes + '}')

# fechar o arquivo
f.close()

# Percorrer todas as imagens
for index in indexes:
    # Ler a imagem do directorio de treino
    img = cv2.imread(os.path.join(path, str(index) + '.png'))
    # Ler os boxes associados as imagens de acordo com o index
    bounding_box = box_dict[index]
    # Converter os boxes compatível com o dlib
    l, t, r, b = bounding_box
    dlib_box = [dlib.rectangle(left=l, top=t, right=r, bottom=b)]
    # Salvar a imagem e as boxes
    data[index] = (img, dlib_box)

print('Numero de boxes e imagens associadas:', len(data))

# O percentual de treino e de teste
percent = 0.8

# Pegando a quantidade dos 80% definido anteriormente
split = int(len(data) * percent)

# Separar as imagens e caixas
images = [tuple_value[0] for tuple_value in data.values()]
bounding_boxes = [tuple_value[1] for tuple_value in data.values()]

# Initialize object detector Options
options = dlib.simple_object_detector_training_options()

# Desabilitando esta opção ele ignora o espelhamento do objeto
# por exemplo caso o objeto possa ser represantado da esquerda para direito e vise e versa
options.add_left_right_image_flips = False

# para o nosso exemplo o valor 6 foi suficiente para detecção
# Aqui podemos alterar os valores para se ajustar melhor ao objeto detectado
options.C = 5

# contar o tempo antes do treino
st = time.time()

detector = dlib.train_simple_object_detector(images[:split], bounding_boxes[:split], options)

# Print the Total time taken to train the detector
print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

file_name = 'Helmet_Detector.svm'
detector.save(file_name)

print("Metricas de treino: {}".format(dlib.test_simple_object_detector(images[:split], bounding_boxes[:split], detector)))
print("Metricas de teste: {}".format(dlib.test_simple_object_detector(images[split:], bounding_boxes[split:], detector)))

detector = dlib.train_simple_object_detector(images, bounding_boxes, options)
detector.save(file_name)

file_name = 'Helmet_Detector.svm'
detector = dlib.simple_object_detector(file_name)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

image = cv2.imread('emilly6.jpg')

detections = detector(image)

# Loop for each detection.
for helmet in (detections):
    l, t, r, b = helmet.left(), helmet.top(), helmet.right(), helmet.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)
    cv2.putText(image, 'Helmet', (l, b + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow('frame', cv2.resize(image, (640, 360)))
cv2.waitKey(0)
cv2.destroyAllWindows()