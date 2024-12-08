import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn đến thư mục chứa ảnh khuôn mặt
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Hàm lấy ảnh và nhãn dữ liệu
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        try: 
            PIL_img = Image.open(imagePath).convert('L') # Chuyển sang ảnh xám
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        except ValueError:
            print(f"Bỏ qua file {imagePath} do định dạng ID không hợp lệ.")
            continue

    return faceSamples,ids

print ("\n [INFO] Đang train. Sẽ mất vài giây. Vui lòng đợi ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Lưu model vào trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# In ra số lượng khuôn mặt đã train và kết thúc chương trình
print("\n [INFO] {0} Dữ Liệu đã được train. Thoát chương trình".format(len(np.unique(ids))))