import cv2
import os
import shutil

# Variável usada para limpar ou não os arquivos existentes de teste
clean = False

# Colocar o cv2.Windows_Normal permite ajustar o quadro da imagem
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 1920, 1080)
cv2.moveWindow("frame", 0, 0)

# Inicializando a webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Irá salvar as imagens a cada 4 frames
# Para não ter as mesmas imagens sempre
skip_frames = 3
frame_gap = 0

# Esta pasta salvará as imagens de treinamento
# E o box file salva a posição das caixas para cada imagen
directory = 'train_images_h'
box_file = 'boxes_h.txt'

# Caso clean seja True
# Deletar as imagens do directorio
# Apagar tudo do arquivo das caixas
# Inicialiar o contador como zero
if clean:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    open(box_file, 'w').close()
    counter = 0

# Juntar o arquivo atual lendo o arquivo anteriormente criado
# colocar o contador de acordo com o arquivo boxes para não ter problemas
# aqui ele pega a penúltima posição quando encontrar : e depois pega a ultima posição quando encontra a ,
elif os.path.exists(box_file):
    with open(box_file, 'r') as text_file:
        box_content = text_file.read()
    counter = int(box_content.split(':')[-2].split(',')[-1])

# Abrir o arquivo ou criar caso não exista
fr = open(box_file, 'a')

# Caso não tenha o directorio criar
if not os.path.exists(directory):
    os.mkdir(directory)

# Iniciar a captura das imagens e das caixas
ret, frame = cap.read()
# Tempo de execução das capturas
time = 600

while time >= 0:
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    frame_gap += 1

    # Salvar a imagem a cada 4 frames
    if frame_gap == skip_frames:
        img_name = str(counter) + '.png'
        img_full_name = directory + '/' + str(counter) + '.png'
        cv2.imwrite(img_full_name, orig)
        fr.write('{}:({},{},{},{}),'.format(counter, 200, 100, 400, 300))
        counter += 1
        frame_gap = 0
    cv2.rectangle(frame, (200, 100), (400, 300), (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    time -= 1
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
fr.close()