# Gerekli kütüphaneleri içeri aktarıyoruz
import cv2
import os
import pickle
# import RPi.GPIO as GPIO


_id = 0

# Kilitle
def lock(pin):
    # GPIO.output(pin, GPIO.LOW)
    print('Kilitlendi!')


# Kilit Aç
def unlock(pin):
    # GPIO.output(pin, GPIO.HIGH)
    print('Kilit Açıldı!')


# Kaydını yaptığımız kullanıcıların adlarını 'names' listesine ekliyoruz
names = ['None']

# GPIO26 pinini seçtiğimizi belirtiyoruz
role_pini = [26]
# Gelebilecek gereksiz uyarıları devre dışı bırakıyoruz
# GPIO.setwarnings(False)
# GPIO numaralarına göre seçim yaptık
# GPIO.setmode(GPIO.BCM)
# Seçtiğimiz röle pinini çıkış olarak ayarlıyoruz.
# GPIO.setup(role_pini, GPIO.OUT)
lock(role_pini)

# Oluşturduğumuz etiket dosyasını açıyoruz ve yüklüyoruz
with open('labels', 'rb') as f:
    dicti = pickle.load(f)
    f.close()

# Kamera kullanacağımızı belirtiyoruz
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Kameranın boyut ve çözünürlük ayarlarını yapıyoruz.
camera.set(3, 640)
camera.set(4, 480)
minW = 0.1 * camera.get(3)
minH = 0.1 * camera.get(4)

path = os.path.dirname(os.path.abspath(_file_))
# Yüz tespiti için kullanacağımız Classifier'ın yolunu koda belirtiyoruz.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# OpenCV paketinde bulunan LBPH (Local Binary Pattern Histogram) yüz tanıyıcı kullanıyoruz.
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Oluşturulan eğitici dosyayı açıyoruz
recognizer.read("trainer.yml")

# Yazı tipi belirleme
font = cv2.FONT_HERSHEY_SIMPLEX

# Sonsuz Döngü
while True:
    # Kameradan gelen görüntüler okunuyor.
    ret, im = camera.read()

    # Gelen görüntüler renkliden griye çevriliyor.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Görüntülerde ki yüzler tespit ediliyor.
    faces = faceCascade.detectMultiScale(gray, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        # Yüzlerin etrafına dikdörtgen çiziliyor
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Güven katsayısının 100'den küçük olması gerekiyor. 0 olması en iyi eşleşmeyi temsil eder
        if confidence < 50:
            unlock(role_pini)
            # Kullanıcı ismi tespit edilip yerine koyulur
            _id = names[_id]
            # Güven hesaplanır
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            lock(role_pini)
            # Tespit edilemeyen yüzler için 'Unknown'(Bilinmeyen) yazılır
            _id = "unknown"
            # Güven hesaplanır
            confidence = "  {0}%".format(round(100 - confidence))

        # Yüzün tespit edildiği yere isim ve güven yüzdesi yazılır
        cv2.putText(im, str(_id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(im, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Kamera açık tutulur
    cv2.imshow('camera', im)

    # Programın sonlanması için 'ESC' tuşuna basılması beklenir
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


# GPIO.cleanup()
print("\n [BILGI] Programdan çıkış yapılıyor. Gerekli temizlikler yapılıyor..")
camera.release()
cv2.destroyAllWindows()
