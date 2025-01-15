import cv2
import os
import sys

# Dosyanın bulunduğu konumu buluyoruz.
path = os.path.dirname(os.path.abspath(_file_))
# Yüz tespiti için kullanacağımız Classifier'ın yolunu koda belirtiyoruz.
detector = cv2.CascadeClassifier(path+r'\Classifiers\face.xml')

# Kod için video değil kamera kullanacağımızı '0' yazarak belirtiyoruz.
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Kameranın boyut ve çözünürlük ayarlarını yapıyoruz.
camera.set(3, 640)
camera.set(4, 480)
minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)
# Ön yüz tespit için kullanacağımız dokümanı belirtiyoruz.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Kod çalıştığında kaydedilecek yeni kullanıcı için isim soruyoruz.
name = input("Lütfen kullanıcı adınızı giriniz: ")
# Aldığımız kullanıcı adı için images klasörünün içinde kullanıcının adıyla bir klasör oluşturuyoruz.
dirName = "./images/" + name
print(dirName)
# Eğer kullanıcı adı önceden kullanılmamışsa klasörü oluşturuyoruz.
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Klasör oluşturuldu")
# Kullanıcı adı mevcutsa bunu belirtip programdan çıkış yapıyoruz.
else:
    print("İsim önceden kullanılmış")
    sys.exit()
# Bir adet sayıcı tanımlıyoruz.
count = 1
# Sonsuz döngü başlatıyoruz.
while True:
    # Tanımladığımız sayıcının 100 olup olmadığını kontrol ediyoruz. 100 ise döngüden çıkıyoruz.
    if count >= 100:
        break
    # Kameradan gelen görüntüler okunuyor.
    ret, im = camera.read()
    # Gelen görüntüler renkliden griye çevriliyor.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Görüntülerde ki yüzler tespit ediliyor.
    faces = faceCascade.detectMultiScale(gray)
    # Gelen görüntülerde ki bulunan yüzlerin x, y koordinatı ve width(genişlik), height(uzunluk) bilgilerini alıyoruz.
    for (x, y, w, h) in faces:
        roiGray = gray[y:y+h, x:x+w]
        # Bu görüntüleri oluşturduğumuz kullanıcıya ait klasörün içine sırayla kaydediyoruz.
        fileName = dirName + "/" + name + str(count) + ".jpg"
        cv2.imwrite(fileName, roiGray)
        cv2.imshow("face", roiGray)
        # Bulunan yüzlerin etrafına dikdörtgen ekliyoruz.
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Sayıcıyı her döngünün sonunda 1 arttırıyoruz.
        count += 1
    # Görüntüyü ekrana vermek için olan kod satırı.
    cv2.imshow('im', im)
    # Görüntünün ekranda kalmasını sağlayan kod satırı.
    key = cv2.waitKey(10)
    # Eğer ki "ESC"(Escape) tuşuna basılırsa program sonlanıyor.
    if key == 27:
        break
camera.release()
cv2.destroyAllWindows()