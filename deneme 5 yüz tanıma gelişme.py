import cv2
import face_recognition
import os

# Kamerayı başlat
video_capture = cv2.VideoCapture(0)

# Yüzler için bir ID sözlüğü ve yüz kodlamaları
face_ids = {}
face_encodings_list = []  # Yüz kodlamalarını saklayacak liste
next_id = 0

# Resimlerin kaydedileceği klasör
save_folder = "saved_faces"
os.makedirs(save_folder, exist_ok=True)

frame_skip = 3  # Her 3. kareyi işle
frame_count = 0

while True:
    # Kameradan bir kare al
    ret, frame = video_capture.read()

    if not ret:
        print("Kameradan kare alınamadı!")
        break

    # Her 3. kareyi işlemek için kontrol
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Yüz tanıma işlemi için görüntüyü küçült (işlem hızını artırmak için)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # OpenCV BGR kullanır, bu yüzden görüntüyü RGB'ye dönüştürelim
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Yüzlerin yerini bulalım
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Yüzlerin etrafına dikdörtgen çizelim
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Yüzlerin yerini orijinal boyutlarına geri çevir
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Yüzün tanımlayıcısını al
        face_encoding = face_recognition.face_encodings(rgb_small_frame, [face_location])[0]

        # Yüz tanıma için mevcut yüzlerle karşılaştır
        matches = face_recognition.compare_faces(face_encodings_list, face_encoding)

        # Eğer tanınan yüz yoksa yeni bir ID ata
        if not any(matches):
            face_encodings_list.append(face_encoding)  # Yüz kodlamasını listeye ekle
            face_ids[face_encoding.tobytes()] = next_id  # Yüzü tanımlayıcı olarak ekle
            face_id = next_id  # Yeni ID'yi al
            next_id += 1  # Sonraki ID için artır
            
            # Yüzü kaydet
            face_image = frame[top:bottom, left:right]  # Yüz kısmını al
            save_path = os.path.join(save_folder, f'face_{face_id}.jpg')
            cv2.imwrite(save_path, face_image)  # Yüzü kaydet
            print(f'Saved: {save_path}')  # Kayıt işlemini bildir
        else:
            # Tanımlanan ID'yi bul
            first_match_index = matches.index(True)
            face_id = face_ids[face_encodings_list[first_match_index].tobytes()]

        # Dikdörtgen çiz ve ID'yi yazdır
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {face_id}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Sonuçları göster
    cv2.imshow('Video', frame)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Her şeyi temizle
video_capture.release()
cv2.destroyAllWindows()
