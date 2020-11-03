from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.utils import to_categorical
import pandas as pd
import tensorflow as tf

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()
    

train = pd.read_csv('/kaggle/input/dataset/train.csv')
valid = pd.read_csv('/kaggle/input/dataset/val.csv')
test = pd.read_csv('/kaggle/input/dataset/test.csv')

model_name = 'bert-base-multilingual-cased'
max_length = 200

with strategy.scope():
    
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
    transformer_model = TFBertModel.from_pretrained(model_name, config = config)

    bert = transformer_model.layers[0]
    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    inputs = {'input_ids': input_ids}

    bert_model = bert(inputs)[1]
    # dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
    dropout = Dropout(rate=0.05, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)

    emotion = Dense(units=len(train.Emotion.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='emotion')(pooled_output)
    outputs = {'emotion': emotion}

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    optimizer = Adam(learning_rate=0.0001, decay=0.01)

    loss = {'emotion': SparseCategoricalCrossentropy(from_logits = False)}
    metric = {'emotion': SparseCategoricalAccuracy('accuracy')}

model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

y_emotion = to_categorical(train['Emotion'], 7)
print(y_emotion)

x = tokenizer(
    text=train['Comment'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)


history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'emotion': y_emotion},
    validation_data=(valid['Comment'], valid['Emotion']),
    batch_size=32,
    epochs=10)

model.save('bert_model.h5')

print("done")


test_y_emotion = to_categorical(test['Emotion'])
test_x = tokenizer(
    text=test['Comment'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)

model_eval = model.evaluate(
    x={'input_ids': test_x['input_ids']},
    y={'emotion': test_y_emotion}
)