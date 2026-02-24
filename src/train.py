import os
import tensorflow as tf
from src.data_loader import FaceDataLoader
from src.model import build_hybrid_model
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    print("Initializing Training Pipeline...")
    
    # -------------------------------------------------------------
    # 1. Hyperparameters & Configuration
    # -------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 1200 # As per project requirement
    VOCAB_SIZE = 5000
    MAX_CAPTION_LEN = 20
    
    # Checkpoint Directory
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    
    # -------------------------------------------------------------
    # 2. Mock Data Generator Setup (for structural run)
    # Ideally, replace with actual Custom Dataset of 1200+ images
    # -------------------------------------------------------------
    print("Loading Custom Dataset (mock structure)...")
    # Simulating 500 records for the pipeline testing
    dummy_images = [f"dataset/dummy_{i}.jpg" for i in range(500)]
    dummy_labels = np.random.randint(0, NUM_CLASSES, 500)
    # Target sequence with '<start>' (idx 1), '<end>' (idx 2)
    dummy_captions = [[1] + np.random.randint(3, VOCAB_SIZE, MAX_CAPTION_LEN-2).tolist() + [2] for _ in range(500)]
    
    # 80-20 Split
    split = int(0.8 * len(dummy_images))
    train_gen = FaceDataLoader(dummy_images[:split], dummy_labels[:split], dummy_captions[:split], 
                               vocab=None, batch_size=BATCH_SIZE, target_size=IMG_SIZE, augment=True)
    val_gen = FaceDataLoader(dummy_images[split:], dummy_labels[split:], dummy_captions[split:], 
                             vocab=None, batch_size=BATCH_SIZE, target_size=IMG_SIZE, augment=False)
    
    # -------------------------------------------------------------
    # 3. Model Architecture Instantiation
    # -------------------------------------------------------------
    model = build_hybrid_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), 
                               num_classes=NUM_CLASSES, 
                               vocab_size=VOCAB_SIZE, 
                               max_caption_len=MAX_CAPTION_LEN)
    
    # -------------------------------------------------------------
    # 4. Callbacks (Checkpoints, Early Stopping)
    # Targeting >97% recognition accuracy and >92% semantic accuracy
    # -------------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/best_model.h5",
            monitor="val_recognition_head_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        )
    ]
    
    # -------------------------------------------------------------
    # 5. Execute Training
    # -------------------------------------------------------------
    print("Starting Model Training...")
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        print("Training Complete. Model saved to 'checkpoints/best_model.h5'")
    except Exception as e:
        print(f"Training Exception (likely due to missing dummy images if running locally without data): {e}")
        print("Model pipeline compiled successfully. Ensure dataset is placed in 'dataset/' to train.")

if __name__ == "__main__":
    train()
