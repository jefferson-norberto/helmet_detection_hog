import os
import numpy as np
from PIL import Image
import dlib
import cv2

paths = [os.path.join('images/', f) for f in os.listdir('images')]
detector_face_hog = dlib.get_frontal_face_detector()

print('Iniciando os cortes das fotos')
i = 0
p = 0
for path in paths:
    print('Path:',p)
    image = cv2.imread(path)
    height, width = image.shape[:2]
    image = cv2.resize(image, (854, 480))

    # Esse número final é necessário pois informa a escala para o detector
    detections = detector_face_hog(image, 2)

    j += 1
    for face in detections:
        # ele usa esses parametros no lugar do x, y, h, w
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        print('l:', l, 't:', t)
        print('r:', r, 'b:', b)
        cv2.rectangle(image, (l-30, t-80), (r+30, b+50), (0, 255, 255), 2)
        image2 = Image.fromarray(image)
        croppedIm = image2.crop((l-40, t-80, r, b))
        image2 = np.array(croppedIm, 'uint8')
        cv2.imwrite('edited/'+str(i)+'.png', image2)
        i = i + 1
