# trainer.py

from model_builder import build_model
from preprocessor import preprocess_data

def train_model(train_generator, validation_generator, epochs=10):
    model = build_model()
    history = model.fit(
        train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator, validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs
    )
    return model

if __name__ == "__main__":
    train_gen, val_gen = preprocess_data('dataset')
    trained_model = train_model(train_gen, val_gen)
