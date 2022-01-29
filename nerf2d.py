import os
import logging

# Silence Tensworflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import numpy as np

from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.callbacks import LambdaCallback

from PIL import Image
from cv2 import cv2

import argparse


class Positional_Encoding(object):
    available_encodings = ['sin_cos', 'repeat_xy', 'test']
    def __init__(self, image, encoding, L = 10):
        if encoding not in Positional_Encoding.available_encodings:
            raise Exception(f'"{encoding}" encoding not supported.')
        
        super().__init__()

        self.image = image
        self.encoding = encoding
        self.L = L

    def get_dataset(self, shuffle=True):
        height, width, _ = self.image.shape

        # Format positions to range [-1, 1)
        x_linspace = (np.linspace(0, width - 1, width) / width) * 2 - 1
        y_linspace = (np.linspace(0, height - 1, height) / height) * 2 - 1

        x_encoding = []
        y_encoding = []
        x_encoding_hf = []
        y_encoding_hf = []
        for l in range(self.L):
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
                
            elif self.encoding == 'test':
                x = np.cos(val * np.pi * x_linspace)
                x_encoding.append(x)

                x = np.cos(val * np.pi * x_linspace)
                x_encoding_hf.append(x)

                y = np.cos(val * np.pi * y_linspace)
                y_encoding.append(y)

                y = np.cos(val * np.pi * y_linspace)
                y_encoding_hf.append(y)
                
        # Format positional encodings.
        inputs, outputs, indices = [], [], []
        for y in range(height):
            for x in range(width):
                r, g, b = self.image[y, x]
                r = r * 2 - 1
                g = g * 2 - 1
                b = b * 2 - 1

                # Concatenate positional encodings.
                p_enc = []
                for l in range(self.L):
                    p_enc.append(x_encoding[l][x])
                    p_enc.append(x_encoding_hf[l][x])

                    p_enc.append(y_encoding[l][y])
                    p_enc.append(y_encoding_hf[l][y])

                inputs.append(p_enc)
                outputs.append([r, g, b])
                indices.append([float(x), float(y)])

        inputs, outputs, indices = np.asarray(inputs), np.asarray(outputs), np.asarray(indices)

        if shuffle:
            np.random.shuffle(indices)
            
            temp_inputs, temp_outputs = [], []
            for x, y in indices:
                index = int(y * self.image.shape[1] + x)
                temp_inputs.append(inputs[index])
                temp_outputs.append(outputs[index])
                
            inputs, outputs = np.asarray(temp_inputs), np.asarray(temp_outputs)
            del temp_inputs, temp_outputs

        return inputs, outputs, indices
    

def GenerateModel(encoding: str, image_name: str, neurons=128, layers=2,
                  epochs=1000, learning_rate=5e-4, batch_size=65536, L=10,
                  dataset='dataset', export_memory=None, model_dir='saved_model',
                  save_epochs=None, verbose=False):
    
    """Generate a NeRF2D model.
    Args:
        encoding: Positional encoding, one of `PositionalEncoding.available_encodings`.
        image_name: The name of the image on which to train.
        neurons: The number of neurons per hidden layer.
        layers: The number of hidden layers.
        epochs: The number of epochs to train.
        learning_rate: The learning rate of the optimizer.
        batch_size: The number of pixels per batch.
        L: The value of L in Equation (4) of the NeRF paper.
        dataset: The directory containing the training image.
        export_memory: The name for the image showing the model's output.
        model_dir: The directory in which to save the model.
        save_epochs: A list of epochs at which to save the model.
        verbose: Boolean to print logging information.
    Returns:
        model: The generated model.
    """
    
    if verbose:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Format input image.
    image_path = os.path.join(dataset, image_name + '.jpg')
    image = Image.open(image_path)
    image = np.asarray(image)
    image = image / 255.0

    # Create dataset of image pixels.
    positional_encoding = Positional_Encoding(image, encoding, L)
    positions, targets, indices = positional_encoding.get_dataset(shuffle=True)
    input_shape = [None, positions.shape[-1]]

    # Design the model.
    channels = image.shape[-1]
    model = Sequential()
    for _ in range(layers):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(channels, activation='linear'))

    # Initialize the model.
    loss_fn = tf.keras.losses.MeanSquaredError()
    metric = tf.keras.metrics.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])
    model.build(input_shape=input_shape)
    model.summary()

    # Apply verbose settings.
    if verbose:
        verbose = 1
        callbacks = []
    else:
        verbose = 0
        def minimal_output(epoch, logs):
            if (epoch + 1) % 100 == 0:
                print('epoch', epoch + 1, '|',
                      'loss', logs['loss'], '|',
                      'error', logs['mean_squared_error'])
                
        json_logging_callback = LambdaCallback(on_epoch_end=minimal_output)
        callbacks = [json_logging_callback]
        
    # Train the model.
    if save_epochs is not None:
        save_epochs = set(save_epochs)  # Remove duplicates
        save_epochs = list(save_epochs)
        save_epochs.sort()
        
        learning_rate_str = '{:.0e}'.format(learning_rate)
        save_dir = f'{model_dir}' \
                f'/{neurons}x{layers}' \
                f'_{encoding}' \
                f'_{learning_rate_str}' \
                f'_{image_name}' \
                f'_by-epoch'
        
        if 0 in save_epochs:
            save_epochs.remove(0)
            
            save_path = save_dir + f'/epoch_0'
            model.save(save_path)
        
        initial_epoch = 0
        for end_epoch in save_epochs:
            num_epochs = end_epoch - initial_epoch
            model.fit(positions, targets, batch_size=batch_size, epochs=end_epoch,
                  verbose=verbose, callbacks=callbacks,
                  initial_epoch=initial_epoch)
            
            save_path = save_dir + f'/epoch_{end_epoch}'
            model.save(save_path)
            
            initial_epoch = end_epoch
        
        # Complete training.
        model.fit(positions, targets, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=callbacks,
                  initial_epoch=initial_epoch)
        
    else:
        model.fit(positions, targets, batch_size=batch_size, epochs=epochs,
                  verbose=verbose, callbacks=callbacks)

    # Save the model.
    if model_dir is not None:
        learning_rate_str = '{:.0e}'.format(learning_rate)
        save_path = f'{model_dir}' \
                    f'/{neurons}x{layers}' \
                    f'_{encoding}' \
                    f'_{learning_rate_str}' \
                    f'_{image_name}'
        
        model.save(save_path)
        print(f'Saved final model to {save_path}')

    # Optionally save the output.
    if export_memory is not None:
        pred_image = np.zeros_like(image)

        # Note: when using strictly cosine encoding or sine encoding, the model output looks wrong,
        #  as it seems mirrored. However, the algorithm works for sin_cos and repeat_xy correctly.
        
        # positional_encoding = Positional_Encoding(image, args.encoding)
        # positions, _, indices = positional_encoding.get_dataset(shuffle=False)

        output = model(positions, training=False)

        # Order of each pixel in shuffled arrangement.
        indices = indices.astype('int')
        indices = indices[:, 1] * image.shape[1] + indices[:, 0]

        # Reformat pixels from [-1, 1) -> [0, 1)
        np.put(pred_image[:, :, 0], indices, np.clip((output[:, 0] + 1) / 2., 0, 1))
        np.put(pred_image[:, :, 1], indices, np.clip((output[:, 1] + 1) / 2., 0, 1))
        np.put(pred_image[:, :, 2], indices, np.clip((output[:, 2] + 1) / 2., 0, 1))

        save_img = np.copy(pred_image[...,::-1] * 255.0)
        cv2.imwrite(export_memory + '.jpg', save_img.astype('uint8'))
        
    return model

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate a NeRF2D model.')
    parser.add_argument('--encoding', type=str, required=True, 
            help='Positional encoding, one of' \
                 '`PositionalEncoding.available_encodings`.')
    parser.add_argument('--image-name', type=str, required=True,
            help='The name of the image on which to train.')
    parser.add_argument('--neurons', type=int, default=128, 
            help='The number of neurons per hidden layer.')
    parser.add_argument('--layers', type=int, default=2, 
            help='The number of hidden layers.')
    parser.add_argument('--epochs', type=int, default=1000, 
            help='The number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=5e-4, 
            help='The learning rate of the optimizer.')
    parser.add_argument('--batch-size', type=int, default=256 * 256, 
            help='The number of pixels per batch.')
    parser.add_argument('--L', type=int, default=10, 
            help='The value of L in Equation (4) of the NeRF paper.')
    parser.add_argument('--dataset', type=str, default='dataset', 
            help='The directory containing the training image.')
    parser.add_argument('--export-memory', type=str, default=None,
            help='The name for the image showing the model\'s output.')
    parser.add_argument('--model-dir', type=str, default='saved_model',
            help='The directory in which to save the model. \'None\' disables backup.')
    parser.add_argument('--save-epochs', type=str, default=None, 
            help='Save the model at each listed epoch. Input in the form of a ' \
                 'comma-separated list. "0" implies saving the model before training.')
    parser.add_argument('--verbose', action='store_true',
            help='Print logging information.')
    args = parser.parse_args()

    # Format save_epochs.
    save_epochs = args.save_epochs.strip().split(',')
    save_epochs = [int(epoch) for epoch in save_epochs]
    
    GenerateModel(args.encoding, args.image_name, neurons=args.neurons,
                  layers=args.layers, epochs=args.epochs,
                  learning_rate=args.learning_rate, batch_size=args.batch_size,
                  L=args.L, dataset=args.dataset, export_memory=args.export_memory,
                  model_dir=args.model_dir, save_epochs=save_epochs,
                  verbose=args.verbose)
