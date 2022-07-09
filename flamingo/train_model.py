import tensorflow as tf
import Fused_Transformer
import pickle
import numpy as np


image_text_file = open('data_process/image_text_file.pkl', 'rb')
image_text_data = pickle.load(image_text_file)
image_text_file.close()

image_embedding = []
input_ids = []
token_type_ids = []
masked_lm_positions = []
masked_lm_ids = []

len_ = len(image_text_data)

for index in range(64):
    image_embedding.append(image_text_data[index]['image_embedding'])
    input_ids.append(image_text_data[index]['input_ids'])
    token_type_ids.append(image_text_data[index]['token_type_ids'])
    masked_lm_positions.append(image_text_data[index]['masked_lm_positions'])
    masked_lm_ids.append(image_text_data[index]['masked_lm_ids'])
all_image_embeddings = np.vstack(image_embedding)
all_input_ids = np.vstack(input_ids)
all_token_type_ids = np.vstack(token_type_ids)
all_masked_lm_positions = np.vstack(masked_lm_positions)
all_masked_lm_ids = np.vstack(masked_lm_ids)


# optimizer = optimization.AdamWeightDecayOptimizer(learning_rate=0.001)
optimizer = tf.train.AdamOptimizer()
model = Fused_Transformer.FusedTransformer()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./models/model_{epoch:02d}_{val_loss:.8f}.ckpt',
                                                 save_weights_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='tensorboard_logs/', histogram_freq=1)

model.compile(optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    history = model.fit(x=[all_input_ids, all_token_type_ids, all_image_embeddings, all_masked_lm_positions],
                        y=all_masked_lm_ids,
                        batch_size=32, epochs=5, validation_split=0.2, validation_freq=1, callbacks=[cp_callback, tensorboard])
