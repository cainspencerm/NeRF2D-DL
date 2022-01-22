import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras import Model, Sequential, Input

from PIL import Image
from cv2 import cv2

import argparse
import os


class Positional_Encoding(object):
    available_encodings = ['sin_cos', 'repeat_xy']
    def __init__(self, image, encoding):
        super().__init__()

        self.image = image
        self.encoding = encoding

    def get_dataset(self, shuffle=True):
        height, width, _ = self.image.shape

        # Format positions to range [-1, 1)
        x_linspace = (np.linspace(0, width - 1, width) / width) * 2 - 1
        y_linspace = (np.linspace(0, height - 1, height) / height) * 2 - 1

        L = 10

        x_encoding = []
        y_encoding = []
        x_encoding_hf = []
        y_encoding_hf = []
        for l in range(L):
            val = 2 ** l

            # Gamma encoding described in (4) of NeRF paper.
            if self.encoding == 'sin_cos':
                x = np.sin(val * np.pi * x_linspace)
                x_encoding.append(x)

                x = np.cos(val * np.pi * x_linspace)
                x_encoding_hf.append(x)

                y = np.sin(val * np.pi * y_linspace)
                y_encoding.append(y)

                y = np.cos(val * np.pi * y_linspace)
                y_encoding_hf.append(y)

            # Deep Learning Group proposed encoding.
            elif self.encoding == 'repeat_xy':
                x_encoding.append(x_linspace)

                x_encoding_hf.append(x_linspace)

                y_encoding.append(y_linspace)

                y_encoding_hf.append(y_linspace)

        inputs, outputs, indices = [], [], []
        for y in range(height):
            for x in range(width):
                r, g, b = self.image[y, x]
                r = r * 2 - 1
                g = g * 2 - 1
                b = b * 2 - 1

                # Concatenate positional encoding.
                p_enc = []
                for l in range(L):
                    p_enc.append(x_encoding[l][x])
                    p_enc.append(x_encoding_hf[l][x])

                    p_enc.append(y_encoding[l][y])
                    p_enc.append(y_encoding_hf[l][y])

                inputs.append(p_enc)
                outputs.append([r, g, b])
                indices.append([float(x), float(y)])

        inputs, outputs, indices = np.asarray(inputs), np.asarray(outputs), np.asarray(indices)

        if shuffle:
            shuffle_list = []
            for input, output, index in zip(inputs, outputs, indices):
                shuffle_list.append([input, output, index])
            shuffle_list = np.asarray(shuffle_list)
            np.random.shuffle(shuffle_list)

            inputs, outputs, indices = [], [], []
            for row in shuffle_list:
                input, output, index = row
                inputs.append(input)
                outputs.append(output)
                indices.append(index)

        return np.asarray(inputs), np.asarray(outputs), np.asarray(indices)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=128, help='Number of neurons per layer')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256 * 256, help='Number of pixels per batch')
    parser.add_argument('--encoding', type=str, default='sin_cos',
            help=f'Positional encoding, one of {Positional_Encoding.available_encodings}')
    parser.add_argument('--image', type=str, default='fractal', help='Image to learn')
    parser.add_argument('--dataset', type=str, default='dataset', help='The directory of images')
    parser.add_argument('--verbose', action='store_true', help='The amount of information printed')
    parser.add_argument('--export-memory', type=str, default=None,
            help='Name to image to save model\'s memory')
    parser.add_argument('--model-backup', type=str, default='saved_model',
            help='Directory to save the model. \'None\' disables backup.')
    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Format input image.
    image_path = os.path.join(args.dataset, args.image + '.jpg')
    image = Image.open(image_path)
    image = np.asarray(image)
    image = image / 255.0

    # Create dataset of image pixels.
    positional_encoding = Positional_Encoding(image, args.encoding)
    positions, targets, indices = positional_encoding.get_dataset(shuffle=True)
    input_shape = [None, positions.shape[1]]

    # Design the model.
    channels = image.shape[-1]
    model = Sequential()
    for _ in range(args.layers):
        model.add(Dense(args.neurons, activation='relu'))
    model.add(Dense(channels, activation='linear'))

    # Initialize the model.
    loss_fn = tf.keras.losses.MeanSquaredError()
    metric = tf.keras.metrics.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])
    model.build(input_shape=input_shape)
    model.summary()

    # Train the model.
    if args.verbose:
        verbose = 1
    else:
        verbose = 0
        print('Training...')
    model.fit(positions, targets, batch_size=args.batch_size, epochs=args.epochs, verbose=verbose)

    # Optionally save the model.
    if args.model_backup != 'None':
        learning_rate = '{:.0e}'.format(args.learning_rate)
        save_path = f'./{args.model_backup}/{args.neurons}x{args.layers}_{args.encoding}_{learning_rate}_{args.image}'
        model.save(save_path)
        print(f'Saved model to {save_path}')

    # Optionally save the output.
    if args.export_memory is not None:
        pred_image = np.zeros_like(image)
        output = model(positions, training=False)

        indices = indices.astype('int')
        indices = indices[:, 1] * width + indices[:, 0]

        np.put(pred_image[:, :, 0], indices, np.clip((output[:, 0]+1)/2.0, 0, 1))
        np.put(pred_image[:, :, 1], indices, np.clip((output[:, 1]+1)/2.0, 0, 1))
        np.put(pred_image[:, :, 2], indices, np.clip((output[:, 2]+1)/2.0, 0, 1))

        save_img = np.copy(pred_image[...,::-1]*255.0)
        cv2.imwrite(args.export_memory + '.jpg', save_img.astype('uint8'))
