import glob
import os
from itertools import chain

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)


def splitting_data(
    df, df_target=None, take_time_stamps=124, overlap=62, zero_padding=True
):
    def windowed_view_adj(
        arr, window=take_time_stamps, overlap=overlap, zero_padding=zero_padding
    ):
        windows = windowed_view(arr, window, overlap)
        if zero_padding:
            re = add_zero_padding(arr, window, overlap)
            return np.append(windows, re, axis=0)
        return windows

    def calculate_number_of_created_samples(
        arr, window=take_time_stamps, overlap=overlap, zero_padding=zero_padding
    ):
        window_step = window - overlap
        new_shape = ((arr.shape[-1] - overlap) // window_step, window)
        return new_shape[0] + 1 if zero_padding else new_shape[0]

    vals = df.values
    vals_shape = vals.shape
    if vals_shape[1] >= take_time_stamps:
        if df_target is None:
            data = list(map(windowed_view_adj, vals))
            return data, None
        else:
            targ_data = df_target.values
            temp_re = [
                (
                    [
                        windowed_view_adj(l),
                        np.array(list(d) * calculate_number_of_created_samples(l)),
                    ]
                )
                for l, d in zip(vals, targ_data)
            ]
            data, data_target = zip(*temp_re)
            data = np.array(data)
            dat_shape = data.shape
            data = data.reshape(dat_shape[0] * dat_shape[1], dat_shape[-1])
            data_target = list(chain(*data_target))
            assert data.shape[0] == len(
                data_target
            ), "Target and data rows are different size!"
            return data, data_target

    else:
        print("Not enough samples")
        return None, None


def windowed_view(arr, window, overlap):
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step, window)
    new_strides = arr.strides[:-1] + (window_step * arr.strides[-1],) + arr.strides[-1:]
    return as_strided(arr, shape=new_shape, strides=new_strides)


def add_zero_padding(arr, window, overlap):
    # need_zeros = len(arr)
    array_len = len(arr)
    window_step = window - overlap
    number_of_els = (arr.shape[-1] - overlap) // window_step
    take_ind = number_of_els * window_step
    number_of_left_elements = array_len - take_ind
    padded_arr = np.array(
        list(arr[take_ind:]) + (window - number_of_left_elements) * [0]
    ).reshape(1, window)
    assert padded_arr.shape == (
        1,
        window,
    ), f"Wrong dimensions after zero padding, expected (1, {window}), got {padded_arr.shape}"
    return padded_arr


def load_files(main_folder_path):
    df_dict = {}
    failed_to_load = []
    # {folder_name: {test: df, train:df}}
    data_folders = os.listdir(main_folder_path)
    for f in data_folders:
        try:
            test_file = glob.glob(f"{main_folder_path}/{f}/*TEST.arff")
            # print(f"test_file: {test_file}")
            train_file = glob.glob(f"{main_folder_path}/{f}/*TRAIN.arff")
            # print(f"train_file: {train_file}")
            if test_file and train_file:
                temp_dict = {
                    "train": pd.DataFrame(arff.loadarff(train_file[0])[0]),
                    "test": pd.DataFrame(arff.loadarff(test_file[0])[0]),
                }
                df_dict[f] = temp_dict
        except:
            print(f)
            failed_to_load.append(f)
    return df_dict, failed_to_load


def load_all_files(
    main_folder_path,
    exclude_all_dataset,
    save_main_folder=None,
    min_allowed_nan_vals=0.1,
):
    data_folders = os.listdir(main_folder_path)
    train_datasets = {}
    test_datasets = {}
    exluded_full_dataset_targets = {}
    exluded_full_dataset = {}
    failed_to_load_datasets = {}
    train_datasets_targets = {}
    test_datasets_targets = {}
    test_dataset_length = 0
    train_dataset_length = 0
    print(f"Total datasets {len(data_folders)}")
    for f in data_folders:
        try:
            joined_data = pd.DataFrame()
            train_and_test_data_joined = pd.DataFrame()
            test_file = glob.glob(f"{main_folder_path}/{f}/*TEST.arff")
            # print(f"test_file: {test_file}")
            train_file = glob.glob(f"{main_folder_path}/{f}/*TRAIN.arff")

            if test_file and train_file:
                train_df = pd.DataFrame(arff.loadarff(train_file[0])[0])

                test_df = pd.DataFrame(arff.loadarff(test_file[0])[0])

                # Dropping rows that contain more than 10 % of nan values
                train_nan = train_df.isna().sum(axis=1) / train_df.count(axis=1)
                train_not_nan = train_nan < min_allowed_nan_vals
                train_df = train_df.loc[train_not_nan]

                test_nan = test_df.isna().sum(axis=1) / test_df.count(axis=1)
                test_not_nan = test_nan < min_allowed_nan_vals
                test_df = test_df.loc[test_not_nan]
                train_len = len(train_df)
                test_len = len(test_df)
                if not train_len or not test_len:
                    re = (
                        "train and test"
                        if not test_len and not train_len
                        else "train"
                        if not train_len
                        else "test"
                    )
                    print(
                        f"Contains too many nan values: {f}, has more than 10 % nan in {re}. Train NaN: {(train_not_nan.mean())}, test NaN: {test_not_nan.mean()}"
                    )
                    failed_to_load_datasets[
                        f
                    ] = f"Dataset Contains too many nan values in {re} datasets. Train NaN: {(train_not_nan.mean())}, test NaN: {(test_not_nan.mean())}"
                    continue

                if f != exclude_all_dataset:

                    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
                    test_df.fillna(test_df.mean(numeric_only=True), inplace=True)
                    assert not test_df.isna().any().sum(), "test df contains nulls"
                    assert not train_df.isna().any().sum(), "train df contains nulls"
                    train_datasets_target = train_df[["target"]]
                    test_datasets_target = test_df[["target"]]
                    train_datasets_targets[f] = train_datasets_target
                    test_datasets_targets[f] = test_datasets_target
                    train_df.drop(labels=["target"], axis=1, inplace=True)
                    test_df.drop(labels=["target"], axis=1, inplace=True)
                    assert (
                        not "target" in test_df.columns
                    ), "did not remove target column in test df"
                    assert (
                        not "target" in train_df.columns
                    ), "did not remove target column in train df"

                    # Processing:
                    train_df = StandardScaler().fit_transform(train_df)
                    test_df = StandardScaler().fit_transform(test_df)
                    train_and_test_data_joined = pd.concat([pd.DataFrame(train_df), pd.DataFrame(test_df)], ignore_index=True)
                    train_df, test_df = train_test_split(train_and_test_data_joined, random_state=42, shuffle=True, test_size=0.2)
                    # Saving
                    train_datasets[f] = pd.DataFrame(train_df)

                    test_datasets[f] = pd.DataFrame(test_df)

                else:

                    joined_data = joined_data.append(train_df, ignore_index=True)
                    joined_data = joined_data.append(test_df, ignore_index=True)
                    exluded_full_dataset_target = joined_data[["target"]]
                    exluded_full_dataset_targets[f] = exluded_full_dataset_target
                    joined_data.drop(labels=["target"], axis=1, inplace=True)

                    before_na = len(joined_data)
                    joined_data = joined_data.loc[
                        (
                            (joined_data.isna().sum(axis=1) / joined_data.count(axis=1))
                            < min_allowed_nan_vals
                        )
                    ]
                    after_na = len(joined_data)
                    print(
                        f"Special dataset: {f}, removed nan samples {after_na - before_na}"
                    )
                    joined_data.fillna(
                        joined_data.mean(numeric_only=True), inplace=True
                    )
                    assert (
                        not joined_data.isna().any().sum()
                    ), "joined_data df contains nulls"
                    joined_data = StandardScaler().fit_transform(joined_data)
                    exluded_full_dataset[f] = joined_data
                    
                if save_main_folder:
                    if not os.path.exists(save_main_folder):
                        os.mkdir(save_main_folder)
                    save_sub_folder = f"{save_main_folder}/{f}"
                    if not os.path.exists(save_sub_folder):
                        os.mkdir(save_sub_folder)
                    # saving files:
                    if not len(joined_data):
                        pd.DataFrame(train_df).to_csv(
                        f"{save_sub_folder}/train.csv", index=False, encoding="utf-8"
                        )
                        pd.DataFrame(test_df).to_csv(
                            f"{save_sub_folder}/test.csv", index=False, encoding="utf-8"
                        )
                        train_datasets_target.to_csv(
                            f"{save_sub_folder}/train_target.csv",
                            index=False,
                            encoding="utf-8",
                        )
                        test_datasets_target.to_csv(
                            f"{save_sub_folder}/test_target.csv",
                            index=False,
                            encoding="utf-8",
                        )
                    else:
                        exluded_full_dataset_target.to_csv(
                            f"{save_sub_folder}/test_target.csv",
                            index=False,
                            encoding="utf-8",
                        )
                        pd.DataFrame(joined_data).to_csv(
                            f"{save_sub_folder}/test.csv", index=False, encoding="utf-8"
                        )
            train_dataset_length += len(train_df)
            test_dataset_length += len(test_df)
            print(f'Dataset {f} | Train {len(train_df)} | Test {len(test_df)}')
        except Exception as e:
            print(f"Failed to load: {f}")
            failed_to_load_datasets[f] = e
            print(f"Failed saving dataset {f}")
            failed_to_load_datasets[f] = e
    print(f'Total training samples {train_dataset_length}')
    print(f'Total testing samples {test_dataset_length}')

    return (
        train_datasets,
        test_datasets,
        train_datasets_targets,
        test_datasets_targets,
        exluded_full_dataset,
        failed_to_load_datasets,
    )


def load_preprocessed_datasets_and_processe(
    main_data_folder,
    exclude_dataset_for_testing,
    save_result_folder=None,
    windows_size=128,
    overlap=64,
    zero_padding=False,
):
    train_data_dict = {}
    test_data_dict = {}
    train_target_dict = {}
    test_target_dict = {}
    data_folders = os.listdir(main_data_folder)
    exceptions = {}
    print(f"Total datasets {len(data_folders)}")
    for f in data_folders:
        try:
            test_df = pd.read_csv(f"{main_data_folder}/{f}/test.csv")
            target_test_df = pd.read_csv(f"{main_data_folder}/{f}/test_target.csv")
            test_shape = test_df.shape
            if test_shape[1] < windows_size:
                exceptions[
                    f
                ] = f"Not enough samples in row, found {test_shape[1]}, expected (window size) {windows_size}"
                continue
            if f == exclude_dataset_for_testing:
                splitted_train, splitted_train_target = pd.DataFrame(), pd.DataFrame()

            else:
                train_df = pd.read_csv(f"{main_data_folder}/{f}/train.csv")
                test_df = pd.read_csv(f"{main_data_folder}/{f}/test.csv")
                target_train_df = pd.read_csv(
                    f"{main_data_folder}/{f}/train_target.csv"
                )

                splitted_train, splitted_train_target = splitting_data(
                    train_df,
                    target_train_df,
                    take_time_stamps=windows_size,
                    overlap=overlap,
                    zero_padding=zero_padding,
                )
            splitted_test, splitted_test_target = splitting_data(
                test_df,
                target_test_df,
                take_time_stamps=windows_size,
                overlap=overlap,
                zero_padding=zero_padding,
            )
            if save_result_folder:
                if not os.path.exists(save_result_folder):
                    os.mkdir(save_result_folder)
                additional_folder = f"{save_result_folder}/w_{windows_size}_o_{overlap}_p_{int(zero_padding)}"
                if not os.path.exists(additional_folder):
                    os.mkdir(additional_folder)
                dataset_folder = f"{additional_folder}/{f}"
                if not os.path.exists(dataset_folder):
                    os.mkdir(dataset_folder)

                # splitted_train = None
                pd.DataFrame(splitted_test).to_csv(
                    f"{dataset_folder}/test.csv", index=False, encoding="utf-8"
                )
                pd.DataFrame(splitted_test_target).to_csv(
                    f"{dataset_folder}/test_target.csv", index=False, encoding="utf-8"
                )
                if len(splitted_train):
                    pd.DataFrame(splitted_train).to_csv(
                        f"{dataset_folder}/train.csv", index=False, encoding="utf-8"
                    )
                    pd.DataFrame(splitted_train_target).to_csv(
                        f"{dataset_folder}/train_target.csv",
                        index=False,
                        encoding="utf-8",
                    )
            else:
                test_target_dict[f] = splitted_test_target
                test_data_dict[f] = splitted_test
                if len(splitted_train):
                    train_target_dict[f] = splitted_train_target
                    train_data_dict[f] = splitted_train

        except Exception as e:
            print(f"Error with {f}: {e}")
            exceptions[f] = e
    return (
        train_data_dict,
        test_data_dict,
        train_target_dict,
        test_target_dict,
        exceptions,
    )


# data_dict, failed_to_load = load_files(f"data")
main_folder_path = "data"


exclude_dataset_for_testing = "InsectSound"
(
    train_datasets,
    test_datasets,
    train_datasets_target,
    test_datasets_target,
    exluded_full_dataset,
    failed_to_load_datasets,
) = load_all_files(main_folder_path, exclude_dataset_for_testing, "processed_datasets")


"""(
    train_data_dict,
    test_data_dict,
    train_target_dict,
    test_target_dict,
    exceptions,
) = load_preprocessed_datasets_and_processe(
    "processed_datasets", exclude_dataset_for_testing,
    save_result_folder='fully_processed_data', windows_size=128, overlap=64
)
"""