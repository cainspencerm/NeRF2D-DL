import tensorflow as tf
from tensorflow.keras import backend as K
from nerf2d import Positional_Encoding
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def ActivationRegions(model_dir: str, encoding: str, epochs: list, image_name: str,
                      neurons=128, layers=2, learning_rate=5e-4, L=10, 
                      dataset='dataset', output_dir='experiments', verbose=False):
    """Generate experimental results on NeRF2D models. Saves results to provided output directory.
    Args:
        model_dir: The directory of the saved model.
        encoding: Positional encoding, one of `PositionalEncoding.available_encodings`.
        epochs: Four epochs on which to present experimental data formatted as a list of
            integers. 0 implies initial weights before training.
        image_name: Image on which the model was trained.
        neurons: Number of neurons per layer.
        layers: Number of layers.
        learning_rate: Learning rate of the model.
        L: Value of L in Equation (4) of NeRF paper.
        dataset: The directory of the image.
        output_dir: The directory in which to save experimental results.
        verbose: Print the status of the script.
    """
    
    # Format input image.
    image_path = f'{dataset}/{image_name}.jpg'
    image = Image.open(image_path)
    image = np.asarray(image)
    image = image / 255.0
    
    # Get the dataset.
    PE = Positional_Encoding(image, encoding, L)
    positions, _, _ = PE.get_dataset(shuffle=False)

    # Format epochs.
    epochs = set(epochs)  # Remove duplicates
    epochs = list(epochs)
    epochs.sort()
    
    # Format learning rate.
    learning_rate = '{:.0e}'.format(learning_rate)

    activations_colors_all = [[], []]
    unique_activations = [[], []]
    
    for epoch in epochs:
        # Load the model.
        model_path = f'{model_dir}/{neurons}x{layers}_{encoding}_' \
                f'{learning_rate}_{image_name}_by-epoch/epoch_{epoch}'
        model = tf.keras.models.load_model(model_path)

        if verbose:
            print(f'Loaded {model_path}')
        
        for layer in range(layers):
            # Get the activations.
            activations = K.function([model.input], [model.layers[layer].output])

            # Pass in data to layers.
            activations_output = activations([positions])

            # Format the ReLU activations to binary.
            activations_output = np.squeeze(activations_output)
            for i, neuron in enumerate(activations_output):
                for j, output in enumerate(neuron):
                    if output > 0:
                        activations_output[i][j] = 1
                    else:
                        activations_output[i][j] = 0

            # Get the amount of unique activation regions.
            activations_tuples = [tuple(x) for x in activations_output.tolist()]
            unique_patterns = set(activations_tuples)
            unique_activations[layer].append(len(unique_patterns))

            if verbose:
                print(f'Amount of unique activation patterns in layer {layer + 1}',
                      f'of epoch {epoch}: {unique_activations[layer][-1]}')

            # Create a dictionary to assign each unique activation a color.
            colors = {}
            for i, activation in enumerate(unique_patterns):
                colors[activation] = i

            # Assign each vector a color based on their activation pattern.
            activations_colors = []
            for activation in activations_tuples:
                activations_colors.append(colors[activation])

            # Reshape for plot.
            activations_colors = np.array(activations_colors)
            activations_colors = np.reshape(activations_colors, (256, 256))
            activations_colors_all[layer].append(activations_colors)

    # Plot results and save.
    for layer in range(layers):
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{encoding} Layer {layer + 1} Activations over Epochs | {image_name}')
        
        i = 0
        for ax_row in axs:
            for ax in ax_row:
                ax.title.set_text(f'Epoch {epochs[i]} | {unique_activations[layer][i]} Activation Patterns')
                pcm = ax.pcolormesh(activations_colors_all[layer][i], cmap='plasma')
                fig.colorbar(pcm, ax=ax)
                i += 1
                
        save_dir = f'{output_dir}/activations{layer + 1}/'
        file_name = f'{image_name}_{encoding}_{epochs[0]}_{epochs[1]}_{epochs[2]}_{epochs[3]}.jpg'
        fig.savefig(save_dir + file_name)
        plt.close(fig)
    