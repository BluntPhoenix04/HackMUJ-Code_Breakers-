# predictor.py

from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

def predict_image(model_path, img_path):
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    
    # Make prediction
    prediction = model.predict(img_array)
    if prediction < 0.5:
        return "Healthy"
    else:
        return "Diseased"

if __name__ == "__main__":
    print(predict_image('plant_disease_model.h5', 'test_plant_image.jpg'))

