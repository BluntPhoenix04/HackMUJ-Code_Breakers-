# evaluator.py

from trainer import train_model
from preprocessor import preprocess_data

def evaluate_and_save(model, validation_generator, model_path='plant_disease_model.h5'):
    validation_loss, validation_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
    print(f'Validation Accuracy: {validation_acc * 100:.2f}%')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_gen, val_gen = preprocess_data('dataset')
    model = train_model(train_gen, val_gen)
    evaluate_and_save(model, val_gen)
