import os
os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
import tensorflow as tf
import keras
from transformers import BertTokenizer, TFBertForSequenceClassification
import pickle

class MedicalMultimodalSystem:
    def __init__(self, cv_model_rel_path, nlp_model_rel_dir):

        print("[INFO] loading...")
        
        self.cv_model_path = cv_model_rel_path
        self.nlp_model_dir = nlp_model_rel_dir
        self.encoder_path = os.path.join(self.nlp_model_dir, "label_encoder.pkl")
        if os.path.isfile(self.cv_model_path):
            self.cv_model = keras.models.load_model(self.cv_model_path)
        else:
            raise FileNotFoundError(f"cv_models dosen't exist: {self.cv_model_path}")
            
        if os.path.isdir(self.nlp_model_dir):
            self.tokenizer = BertTokenizer.from_pretrained(self.nlp_model_dir)
            self.nlp_model = TFBertForSequenceClassification.from_pretrained(self.nlp_model_dir)
        else:
            raise FileNotFoundError(f"NLP_models dosen't exist : {self.nlp_model_dir}")
            
        if os.path.isfile(self.encoder_path):
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            raise FileNotFoundError(f"label_encoder dosen't exist: {self.encoder_path}")
            
        print("[INFO] loading successfully ")

    def _preprocess_image(self, image_path):
        img = keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.efficientnet.preprocess_input(img_array)
        return img_array

    def _preprocess_text(self, text):

        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='tf'
        )

        return encoded
    
    def predict(self,image_path,text,w_cv=0.5,w_nlp=0.5):
        if w_cv < 0 or w_nlp < 0:
            raise ValueError("Weights must be non-negative.")

        total = w_cv + w_nlp
        if total == 0:
            raise ValueError("Sum of weights cannot be zero.")

        w_cv /= total
        w_nlp /= total

        img_ready = self._preprocess_image(image_path)
        cv_probs = self.cv_model.predict(img_ready, verbose=0)[0]
        
        text_ready = self._preprocess_text(text)
        nlp_outputs = self.nlp_model(text_ready, training=False)
        nlp_probs = tf.nn.softmax(nlp_outputs.logits, axis=1).numpy()[0]
        
        final_probs = np.average(
        [cv_probs, nlp_probs],
        axis=0,
        weights=[w_cv, w_nlp])
        
        final_idx = np.argmax(final_probs)
        final_label = self.label_encoder.inverse_transform([final_idx])[0]
        
        classes = self.label_encoder.classes_
        breakdown = dict(
    zip(
        classes,
        map(float, final_probs)
    ))

        
        return final_label, breakdown
