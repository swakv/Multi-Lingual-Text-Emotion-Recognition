import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer

jarvis = tf.keras.models.load_model("best_xlm_backup2.h5")
tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-roberta-base")


def prediction(model,tokenizer,input_string):
    
    padded_tokens = tokenizer(
    text=[input_string],
    add_special_tokens=True,
    max_length=max_len,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)
    
    prediction = model.predict(padded_tokens['input_ids'])
    print(prediction)
    conversion_dict = {0:'joy',1:'fear',2:'anger',3:'sadness'}
    
    return conversion_dict[np.argmax(prediction)]

print(prediction(jarvis,tokenizer,"I am happy"))

