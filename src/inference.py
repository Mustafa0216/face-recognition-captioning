import cv2
import numpy as np
import tensorflow as tf
import time

class RealTimeDescriber:
    """
    Handles real-time webcam inference using the optimized TFLite model.
    Achieves high-speed inference viable for real-time applications.
    """
    def __init__(self, model_path="checkpoints/quantized_model.tflite", vocab_size=5000, max_seq_len=20):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Determine index of image and sequence inputs
        self.img_input_idx = self.input_details[0]['index'] 
        self.seq_input_idx = self.input_details[1]['index']
        # Sequence comes first depending on architecture compilation
        if len(self.input_details[0]['shape']) == 2:
            self.seq_input_idx = self.input_details[0]['index']
            self.img_input_idx = self.input_details[1]['index']

        self.rec_out_idx = self.output_details[0]['index']
        self.cap_out_idx = self.output_details[1]['index']

        self.vocab = {i: f"word_{i}" for i in range(vocab_size)} # Dummy Vocab placeholder
        self.max_seq_len = max_seq_len

    def preprocess_image(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return np.expand_dims(img, axis=0).astype(np.float32)

    def generate_caption(self, image_input):
        # Initialize sequence with <start> token (assuming idx 1)
        sequence = np.zeros((1, self.max_seq_len), dtype=np.float32)
        sequence[0, 0] = 1.0 
        
        predicted_caption = []
        
        for i in range(1, self.max_seq_len):
            self.interpreter.set_tensor(self.img_input_idx, image_input)
            self.interpreter.set_tensor(self.seq_input_idx, sequence)
            
            self.interpreter.invoke()
            
            # Predict next word
            cap_preds = self.interpreter.get_tensor(self.cap_out_idx)
            next_word_idx = np.argmax(cap_preds[0, i-1, :])
            
            if next_word_idx == 2: # <end> token 
                break
                
            sequence[0, i] = next_word_idx
            predicted_caption.append(self.vocab.get(next_word_idx, "<unk>"))
            
        return " ".join(predicted_caption)

    def predict(self, frame):
        start_time = time.time()
        
        img_input = self.preprocess_image(frame)
        
        # 1. First Pass for Identity
        sequence = np.zeros((1, self.max_seq_len), dtype=np.float32)
        sequence[0, 0] = 1.0 
        
        self.interpreter.set_tensor(self.img_input_idx, img_input)
        self.interpreter.set_tensor(self.seq_input_idx, sequence)
        self.interpreter.invoke()
        
        rec_preds = self.interpreter.get_tensor(self.rec_out_idx)
        identity_id = np.argmax(rec_preds[0])
        confidence = np.max(rec_preds[0])
        
        # 2. Sequential Pass for Caption Generation
        caption = self.generate_caption(img_input)
        
        inf_time = (time.time() - start_time) * 1000 # ms
        return identity_id, confidence, caption, inf_time

def main():
    print("Starting Interactive Real-Time Inference module...")
    # Fallback to a placeholder since we might not have the quantized model yet
    print("Please ensure `checkpoints/quantized_model.tflite` exists.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found. Exiting...")
        return
        
    try:
        describer = RealTimeDescriber()
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        print("Please run `python src/train.py` followed by `python src/quantize.py` first.")
        cap.release()
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Inference
        try:
            identity, conf, caption, inf_time = describer.predict(frame)
            
            text_id = f"ID: {identity} ({conf:.2f})"
            text_cap = f"Desc: {caption}"
            text_fps = f"Inference: {inf_time:.1f}ms"
            
            cv2.putText(frame, text_id, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, text_cap, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, text_fps, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            cv2.putText(frame, "Model not initialized properly.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("Face Recognition & Feature Description", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
