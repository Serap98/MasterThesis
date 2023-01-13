import time
import numpy as np
import os
from keras import layers, models, callbacks, regularizers, optimizers

# from keras.layers import advanced_activations
from contextlib import redirect_stdout
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def create_CNN_architecture(
    window_size,
    number_of_layers_in_encoder,
    encoder_filters,
    activation_functions,
    kernel_sizes,
    batch_normalizations,
    max_poolings,
    max_pooling_size=2,
    allowed_bottleneck_sizes=[16, 24, 32],
    **kwargs,
):
    TIMESTEPS = window_size
    num_inputs = 1
    input_placeholder = layers.Input(shape=[TIMESTEPS, num_inputs])
    encoded = input_placeholder
    for i in range(number_of_layers_in_encoder):
        encoder_filter = encoder_filters[i]
        activation_function = activation_functions[i]
        kernel_size = kernel_sizes[i]
        batch_normalization = batch_normalizations[i]
        max_pooling = max_poolings[i]

        encoded = layers.Conv1D(
            encoder_filter,
            kernel_size=kernel_size,
            padding="same",
            activation=activation_function,
        )(encoded)
        if max_pooling:
            encoded = layers.MaxPool1D(max_pooling_size)(encoded)
        if batch_normalization:
            encoded = layers.BatchNormalization()(encoded)
    # bottleneck
    encoded = layers.Dense(1, activation="relu")(encoded)
    encoded = layers.BatchNormalization(name=f"embedding")(encoded)
    bottleneck_shape = list(encoded.shape)[1]
    # print(f'Bottleneck size: {bottleneck_shape}')
    if not (bottleneck_shape in allowed_bottleneck_sizes):
        raise Exception(f"Wrong bottleneck shape: {bottleneck_shape}")

    decoded = encoded

    for i in reversed(range(number_of_layers_in_encoder)):
        encoder_filter = encoder_filters[i]
        activation_function = activation_functions[i]
        kernel_size = kernel_sizes[i]
        batch_normalization = batch_normalizations[i]
        decoded = layers.Conv1DTranspose(
            encoder_filter,
            kernel_size=kernel_size,
            padding="same",
            activation=activation_function,
        )(decoded)
        if max_pooling:
            decoded = layers.UpSampling1D(max_pooling_size)(decoded)

    decoded = layers.Conv1DTranspose(
        filters=1, kernel_size=kernel_size, padding="same"
    )(decoded)

    autoencoder = models.Model(inputs=input_placeholder, outputs=decoded)
    return autoencoder, bottleneck_shape


import pandas as pd
import numpy as np


def load_data(main_data_folder, exclude_dataset_for_testing):
    data_folders = os.listdir(main_data_folder)
    train_data_df = pd.DataFrame()
    test_data_df = pd.DataFrame()
    exceptions = {}
    train_length = 0
    test_length = 0
    print(f"Total datasets {len(data_folders)}")
    for f in data_folders:
        try:
            test_df = pd.read_csv(f"{main_data_folder}/{f}/test.csv")
            
            if f == exclude_dataset_for_testing:
                continue
            else:
                test_length += len(test_df)
                train_df = pd.read_csv(f"{main_data_folder}/{f}/train.csv")
            train_length += len(train_df)
            train_data_df = pd.concat(
                [train_data_df, train_df], ignore_index=True
            )  # train_data_df.append(train_df, ignore_index=True)
            test_data_df = pd.concat(
                [test_data_df, test_df], ignore_index=True
            )  # test_data_df.append(test_df, ignore_index=True)

        except Exception as e:
            exceptions[f] = e
    assert train_length == len(
        train_data_df
    ), "Not all training data was appended to final training set"
    assert test_length == len(
        test_data_df
    ), "Not all testing data was appended to final testing set"
    return train_data_df, test_data_df, exceptions


exclude_dataset_for_testing = "InsectSound"
folder_name = "fully_processed_data/w_128_o_64_p_0"
train_data_df, test_data_df, exceptions = load_data(
    folder_name, exclude_dataset_for_testing
)


model_folder = "saved_models"

"""train_data_shape = (3000, 128)
number_of_layers_in_encoder = 2
encoder_filters = [64, 32]
activation_functions = ['relu'] * number_of_layers_in_encoder
kernel_sizes = [5] * number_of_layers_in_encoder
batch_normalizations = [0] * number_of_layers_in_encoder
max_pooling_size=2

model, encoder = create_CNN_architecture(train_data_shape, number_of_layers_in_encoder, encoder_filters,  activation_functions,
                            kernel_sizes, batch_normalizations,max_pooling_size=2)"""

train_data_shape = (3000, 128)
number_of_layers_in_encoder = 3
encoder_filters = [256, 128, 64, 32]
activation_functions = ["relu"] * number_of_layers_in_encoder
kernel_sizes = [5] * number_of_layers_in_encoder
batch_normalizations = [0] * number_of_layers_in_encoder
max_pooling_size = 2

# model, bottleneck_shape = create_CNN_architecture(train_data_shape, number_of_layers_in_encoder, encoder_filters,  activation_functions,
#                            kernel_sizes, batch_normalizations,max_pooling_size=2)


# encoder_filters = [512, 256, 128, 64, 32]
# focus on different kernel sizes
# use embeddings: 16, 32, 22

try_kernel_sizes = [5, 13, 17, 23, 39]

possible_input_sizes = [128, 256]

number_of_layers = [2, 3, 4, 5]

# try any
#                       kernel_sizes, batch_normalizations
model_1 = {
    "name": "model_1",
    "window_size": 128,
    "number_of_layers_in_encoder": 2,
    "input": 128,
    "encoder_filters": [256, 64],
    "kernel_sizes": [23, 17],
    "activation_functions": ["relu", "relu"],
    "batch_normalizations": [False, False],
    "max_poolings": [True] * 2,
}

model_2 = {
    "name": "model_2",
    "window_size": 128,
    "number_of_layers_in_encoder": 3,
    "input": 128,
    "encoder_filters": [256, 128, 64],
    "kernel_sizes": [23, 13, 7],
    "activation_functions": ["relu"] * 3,
    "batch_normalizations": [False] * 3,
    "max_poolings": [True] * 3,
}

model_3 = {
    "name": "model_3",
    "window_size": 128,
    "number_of_layers_in_encoder": 3,
    "input": 128,
    "encoder_filters": [256, 128, 64],
    "kernel_sizes": [13, 13, 5],
    "activation_functions": ["relu"] * 3,
    "batch_normalizations": [False] * 3,
    "max_poolings": [True] * 3,
}

model_4 = {
    "name": "model_4",
    "window_size": 128,
    "number_of_layers_in_encoder": 2,
    "input": 128,
    "encoder_filters": [128, 64],
    "kernel_sizes": [17, 7],
    "activation_functions": ["relu"] * 2,
    "batch_normalizations": [False] * 2,
    "max_poolings": [True] * 2,
}

model_5 = {
    "name": "model_5",
    "window_size": 128,
    "number_of_layers_in_encoder": 2,
    "input": 128,
    "encoder_filters": [128, 128],
    "kernel_sizes": [13, 7],
    "activation_functions": ["relu"] * 2,
    "batch_normalizations": [True] * 2,
    "max_poolings": [True] * 2,
}

model_6 = {
    "name": "model_6",
    "window_size": 128,
    "number_of_layers_in_encoder": 2,
    "input": 128,
    "encoder_filters": [128, 128],
    "kernel_sizes": [13, 7],
    "activation_functions": ["relu"] * 2,
    "batch_normalizations": [True] * 2,
    "max_poolings": [True] * 2,
}

model_7 = {
    "name": "model_7",
    "window_size": 128,
    "number_of_layers_in_encoder": 3,
    "input": 128,
    "encoder_filters": [512, 256, 128],
    "kernel_sizes": [23, 13, 13],
    "activation_functions": ["relu"] * 3,
    "batch_normalizations": [False] * 3,
    "max_poolings": [True] * 3,
}

model_8 = {
    "name": "model_8",
    "window_size": 128,
    "number_of_layers_in_encoder": 4,
    "input": 128,
    "encoder_filters": [256, 256, 128, 128],
    "kernel_sizes": [23, 23, 13, 13],
    "activation_functions": ["relu"] * 4,
    "batch_normalizations": [False] * 4,
    "max_poolings": [False] * 4,
}

model_9 = {
    "name": "model_9",
    "window_size": 128,
    "number_of_layers_in_encoder": 4,
    "input": 128,
    "encoder_filters": [256, 256, 128, 128],
    "kernel_sizes": [23, 23, 13, 5],
    "activation_functions": ["relu"] * 4,
    "batch_normalizations": [True] * 4,
    "max_poolings": [True] * 4,
}

model_10 = {
    "name": "model_10",
    "window_size": 128,
    "number_of_layers_in_encoder": 5,
    "input": 128,
    "encoder_filters": [256, 256, 128, 128, 64],
    "kernel_sizes": [23, 23, 13, 7, 5],
    "activation_functions": ["relu"] * 5,
    "batch_normalizations": [True] * 5,
    "max_poolings": [False] * 5,
}


def generate_multiple_models(list_of_models):
    created_models_dict = {}
    for model_arch in list_of_models:
        created_model, embed = create_CNN_architecture(**model_arch)
        model_name = model_arch["name"]
        print(f"{model_name} has {embed} embedding")
        created_models_dict[model_name] = {
            "model": created_model,
            "embedding_size": embed,
            "architecture": model_arch,
        }
    return created_models_dict


def compile_model(model, optimizer, loss="mse"):
    model.compile(optimizer=optimizer, loss=loss)
    return model



def train_model(
    model,
    model_name,
    train_data,
    test_data,
    main_model_folder,
    epochs=100,
    batch_size=32,
):
    history = model.fit(
        train_data,
        train_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_data, test_data),
        callbacks=[
            callbacks.ModelCheckpoint(
                f"{main_model_folder}/"
                + model_name
                + f"/callbacks"
                + "/epoch{epoch:02d}-loss{val_loss:.3f}.tf"
            ),
            callbacks.ModelCheckpoint(
                f"{main_model_folder}/" + model_name + f"/callbacks" + "/best.tf",
                save_best_only=True,
            ),
        ],
        verbose=0,
    )

    return history


def save_model_data(model, history, main_model_folder):
    def save_model_summary(model, path_to_save):
        with open(f"{path_to_save}/model_summary.txt", "w") as f:
            with redirect_stdout(f):
                model.summary()
        pd.DataFrame.from_dict(history.history).to_csv("history.csv")

    if not os.path.exists(main_model_folder):
        os.mkdir(main_model_folder)
    with open(f'{main_model_folder}' + '/model_structure.json', mode='w') as ofile:
        ofile.write(model.to_json())
    save_model_summary(model, main_model_folder)


def generate_train_evaluate_save(list_of_models, train_data, test_data, main_model_folder, optimizer, epochs=100):
    print("========================= Training model =========================")
    for ind, model_arch in enumerate(list_of_models):
        try:
            model, embed = create_CNN_architecture(**model_arch)
            k = model_arch["name"]
            #model_dict = created_models_dict[k]
            #model = model_dict['model']
            #embedding_size = model_dict['embedding_size']
            embedding_size = embed
            #k = name
            #number_of_layers_in_encoder = model_dict['architecture']['number_of_layers_in_encoder']
            #encoder_filters = model_dict['architecture']['encoder_filters']
            #kernel_sizes = model_dict['architecture']['kernel_sizes']
            number_of_layers_in_encoder = model_arch['number_of_layers_in_encoder']
            encoder_filters = model_arch['encoder_filters']
            kernel_sizes = model_arch['kernel_sizes']
            print(f'Model iteration: {ind} name: {k}')
            print(f'INFO: Layers: {number_of_layers_in_encoder} | embedding size {embedding_size} | Kernel filters {encoder_filters} | Kernel sizes {kernel_sizes}')
            model.compile(optimizer=optimizer, loss='mse')
            folder_name = f'{main_model_folder}/{k}'
            history = train_model(model, k, train_data, test_data, main_model_folder, epochs=epochs)
            save_model_data(model, history, folder_name)
            re = model.evaluate(test_data)
            hist_df = pd.DataFrame.from_dict(history.history)
            lowest_test_val_loss = hist_df.iloc[hist_df['val_loss'].argmin()]
            print(f'Model {k} results {re}')
            vals = lowest_test_val_loss.values
            print(f'Val loss: train {vals[0]} test: {vals[1]}')
            print("========================= Finished training model =========================")
            print('\n')
            del model
        except Exception as e:
            print(f'Error: {e}')



if __name__ == '__main__':
    exclude_dataset_for_testing = "InsectSound"
    folder_name = "fully_processed_data/w_128_o_64_p_0"
    train_data_df, test_data_df, exceptions = load_data(
        folder_name, exclude_dataset_for_testing
    )
    list_of_models = [
        model_1,
        model_2,
        model_3,
        model_4,
        model_5,
        model_6,
        model_7,
        #model_8,
        #model_9,
        #model_10,
    ]

    train_data = train_data_df.values
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    test_data = test_data_df.values
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
    main_model_folder = 'trained_models'
    opt = optimizers.Adam(learning_rate=.0001)
    generate_train_evaluate_save(list_of_models, train_data, test_data, main_model_folder, opt, epochs=10)