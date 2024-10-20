import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('final_fruit_classifier.keras')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_fruit_condition(image_path):
    image = preprocess_image(image_path)   
    prediction = model.predict(image)
    return 'Fresh' if prediction[0][0] < 0.5 else 'Rotten'

test_image_path ="C:\\Users\\agnim\\OneDrive\\Desktop\\download (9).jpeg"
result = predict_fruit_condition(test_image_path)
print(f'The fruit is: {result}')
