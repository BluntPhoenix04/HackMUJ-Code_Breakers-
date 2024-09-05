import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(dataset_dir, img_size=(150, 150)):
    train_datagen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_dir, target_size=img_size, batch_size=32, class_mode='binary', subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        dataset_dir, target_size=img_size, batch_size=32, class_mode='binary', subset='validation'
    )
    
    return train_generator, validation_generator

if __name__ == "__main__":
    train_gen, val_gen = preprocess_data('dataset')
