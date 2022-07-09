import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pickle
import argparse
from PIL import UnidentifiedImageError
tf.enable_eager_execution()
parser = argparse.ArgumentParser(description='Calculate Embedding')
parser.add_argument('--input-files', type=str, default='pics/', help='all pic files')
parser.add_argument('--output-file', type=str, default='image_embedding.pkl', help='output file')
args = parser.parse_args()


class ImageModel(tf.keras.Model):
    def __init__(self, include_top=False):
        super(ImageModel, self).__init__()
        self.resnet = tf.keras.applications.resnet_v2.ResNet50V2(include_top=include_top)
        self.resnet.trainable = False
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None, mask=None):
        img = inputs
        img = tf.expand_dims(img, axis=0)
        embedding = self.resnet(img, training=False)
        embedding = self.global_average_layer(embedding)
        embedding = np.array(embedding).flatten()
        return embedding


def process_image(image_path):
    preprocess = tf.keras.applications.resnet_v2.preprocess_input
    img = np.array(Image.open(image_path))
    if img.shape[-1] != 3:
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
    img = preprocess(img)
    # [-1, 1]
    return img


def model_builder(include_top=False):
    model = ImageModel(include_top)
    return model


def calculate(input_files, output_file):
    all_pics = os.listdir(input_files)
    all_pics.sort()
    if '.DS_Store' in all_pics:
        all_pics.remove('.DS_Store')
    # 得到所有图像
    model = model_builder()
    embeddings = dict()
    for pic in all_pics:
        print(f'正在处理{pic}')
        if pic == '02393.jpg':
            embedding = np.zeros(shape=(2048,), dtype=np.float32)
        else:
            pic_path = os.path.join(input_files, pic)
            img = process_image(pic_path)
            embedding = model(img)
        num = pic.split('.')[0]
        embeddings[num] = embedding
    pickle_file = open(output_file, 'wb')
    pickle.dump(embeddings, pickle_file)
    pickle_file.close()


def test_calculate(input_files='./test_pics', output_file='./test_result.pkl'):
    all_pics = os.listdir(input_files)
    all_pics.sort()
    # 得到所有图像
    model = model_builder()
    embeddings = dict()
    for pic in all_pics:
        print(f'正在处理{pic}')
        pic_path = os.path.join(input_files, pic)
        if pic == '02393.jpg':
            embedding = np.zeros(shape=(2048,), dtype=np.float32)
        else:
            img = process_image(pic_path)
            embedding = model(img)
        embeddings[pic] = embedding
    pickle_file = open(output_file, 'wb')
    pickle.dump(embeddings, pickle_file)
    pickle_file.close()


def load_embedding(pkl_file):
    embedding_file = open(pkl_file, 'rb')
    embedding = pickle.load(embedding_file)
    embedding_file.close()
    print('embedding个数: ', len(embedding))
    print('embedding key: ', embedding.keys())
    print('embedding维度: ', embedding['00000'].shape)
    print('embedding数据类型: ', embedding['00000'].dtype)
    print('embedding维度: ', embedding['02393'].shape)
    print('embedding数据类型: ', embedding['02393'].dtype)


def main():
    input_file = args.input_files
    output_file = args.output_file
    print('input_file: ', input_file)
    print('output_file: ', output_file)
    calculate(input_file, output_file)
    load_embedding(output_file)


if __name__ == '__main__':
    main()
