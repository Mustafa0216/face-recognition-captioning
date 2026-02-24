import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_hybrid_model(input_shape=(224, 224, 3), 
                       num_classes=1200, 
                       vocab_size=5000, 
                       max_caption_len=20, 
                       embedding_dim=256, 
                       lstm_units=512):
    """
    Builds a Hybrid CNN-LSTM architecture for simultaneous face recognition and feature description.
    """
    
    # 1. CNN Backbone for feature extraction (MobileNetV2 for efficiency)
    image_input = layers.Input(shape=input_shape, name="image_input")
    backbone = applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=image_input)
    
    # Freeze the base model layers initially (can unfreeze later for fine-tuning)
    for layer in backbone.layers:
        layer.trainable = False

    # Extract spatial features
    x = backbone.output
    pooled_features = layers.GlobalAveragePooling2D()(x) # Shape: (batch_size, 1280)
    
    # -------------------------------------------------------------
    # HEAD 1: Face Recognition (Targeting >97% Accuracy)
    # -------------------------------------------------------------
    dense_rec = layers.Dense(1024, activation='relu')(pooled_features)
    dense_rec = layers.Dropout(0.3)(dense_rec)
    recognition_output = layers.Dense(num_classes, activation='softmax', name="recognition_head")(dense_rec)
    
    # -------------------------------------------------------------
    # HEAD 2: Feature Description / Captioning (LSTM) (Targeting >92% semantic accuracy)
    # -------------------------------------------------------------
    # Transform CNN feature to match LSTM units
    image_features = layers.Dense(embedding_dim, activation='relu')(pooled_features)
    image_features = layers.RepeatVector(max_caption_len)(image_features)
    
    # Sequence Input (Captions)
    caption_input = layers.Input(shape=(max_caption_len,), name="caption_input")
    caption_embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    
    # Concatenate Image Features and Sequence Embedding
    # This allows the LSTM to condition on the image features at each time step.
    concat_features = layers.Concatenate(axis=-1)([image_features, caption_embedding])
    
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(concat_features)
    # Adding a second LSTM layer for deeper semantic understanding
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(lstm_out)
    
    caption_output = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'), name="caption_head")(lstm_out)
    
    # -------------------------------------------------------------
    # Compile Model
    # -------------------------------------------------------------
    model = models.Model(inputs=[image_input, caption_input], outputs=[recognition_output, caption_output])
    
    # Dual Loss setup
    losses = {
        "recognition_head": "sparse_categorical_crossentropy",
        "caption_head": "sparse_categorical_crossentropy"
    }
    loss_weights = {"recognition_head": 1.0, "caption_head": 1.0}
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=losses, 
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test architecture
    model = build_hybrid_model()
    model.summary()
