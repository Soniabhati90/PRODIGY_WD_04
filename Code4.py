import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt


def load_images_from_folder(folder, image_size):
    images = []
    labels = []
    for gesture in os.listdir(folder):
        gesture_folder = os.path.join(folder, gesture)
        for filename in os.listdir(gesture_folder):
            img = cv2.imread(os.path.join(gesture_folder, filename))
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))
                images.append(img)
                labels.append(gesture)
    return images, labels


data_folder = 'path/to/gesture_dataset'


image_size = 64


images, labels = load_images_from_folder(data_folder, image_size)


images = np.array(images)
labels = np.array(labels)


images = images / 255.0


label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.array([label_mapping[label] for label in labels])
labels = to_categorical(labels, num_classes=len(label_mapping))


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_mapping), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=64)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


model.save('gesture_recognition_model.h5')

model = tf.keras.models.load_model('gesture_recognition_model.h5')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (image_size, image_size))
    reshaped = resized.reshape(1, image_size, image_size, 3) / 255.0

    
    prediction = model.predict(reshaped)
    gesture_idx = np.argmax(prediction)
    gesture = [key for key, value in label_mapping.items() if value == gesture_idx][0]

    
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()