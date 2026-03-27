import h5py
import numpy as np
import os

class DatasetLoader:

    @staticmethod
    def load_rml_dataset(filename=r'path/to/Indoor-OTA-12Mod.h5'):
        """
        Load dataset (HDF5 format)

        Parameters:
            filename: Path to the dataset file

        Returns:
            (mods, snrs, lbl): Modulation types, SNR values, label information
            (X_train, Y_train): Training set data and one-hot labels
            (X_val, Y_val): Validation set data and one-hot labels
            (X_test, Y_test): Test set data and one-hot labels
            (train_idx, val_idx, test_idx): Data indices
        """

        print(f"Loading dataset: {filename}")

        if not os.path.exists(filename):
            # Try to load numpy format
            npz_file = filename.replace('.h5', '.npz')
            if not os.path.exists(npz_file):
                raise FileNotFoundError(f"Dataset file not found: {filename}")
            return RMLDatasetLoader.load_rml_dataset_npz(npz_file)

        # Load HDF5 format
        with h5py.File(filename, 'r') as f:
            # Read data
            X_train = f['train_data'][:]
            X_val = f['val_data'][:]
            X_test = f['test_data'][:]

            # Read labels (strings)
            train_labels = [label.decode('utf-8') for label in f['train_labels'][:]]
            val_labels = [label.decode('utf-8') for label in f['val_labels'][:]]
            test_labels = [label.decode('utf-8') for label in f['test_labels'][:]]

            # Read SNR labels
            train_snrs = f['train_snrs'][:]
            val_snrs = f['val_snrs'][:]
            test_snrs = f['test_snrs'][:]

            # Read modulation types
            mods = [mod.decode('utf-8') for mod in f['modulation_types'][:]]

            # Read target SNR range
            target_snrs = f['target_snrs'][:]

        # Reshape data to (number of samples, 2, 128) format
        def reshape_to_iq(data):
            if data.shape[1] == 256:
                return data.reshape(-1, 2, 128)
            return data

        X_train = reshape_to_iq(X_train)
        X_val = reshape_to_iq(X_val)
        X_test = reshape_to_iq(X_test)

        # print(f"Data loading completed!")
        # print(f"Modulation types: {mods}")
        # print(f"SNR range: {sorted(np.unique(target_snrs).tolist())}")
        # print(f"Training set: {X_train.shape}")
        # print(f"Validation set: {X_val.shape}")
        # print(f"Test set: {X_test.shape}")

        # Create one-hot encoded labels
        def to_onehot(labels, mods_list):
            num_classes = len(mods_list)
            onehot = np.zeros((len(labels), num_classes))

            # Create mapping from modulation type to index
            mod_to_idx = {mod: idx for idx, mod in enumerate(mods_list)}

            for i, label in enumerate(labels):
                idx = mod_to_idx[label]
                onehot[i, idx] = 1

            return onehot

        Y_train = to_onehot(train_labels, mods)
        Y_val = to_onehot(val_labels, mods)
        Y_test = to_onehot(test_labels, mods)

        # Create indices
        train_len = X_train.shape[0]
        val_len = X_val.shape[0]
        test_len = X_test.shape[0]

        train_idx = list(range(0, train_len))
        val_idx = list(range(train_len, train_len + val_len))
        test_idx = list(range(train_len + val_len, train_len + val_len + test_len))

        # Create label list (for dictionary structure)
        lbl = []
        snrs_all = np.concatenate([train_snrs, val_snrs, test_snrs])
        labels_all = train_labels + val_labels + test_labels

        for label, snr in zip(labels_all, snrs_all):
            lbl.append((label, int(snr)))

        # Create data dictionary (grouped by (mod, snr))
        Xd = {}
        X_all = np.concatenate([X_train, X_val, X_test], axis=0)
        snrs_all = np.concatenate([train_snrs, val_snrs, test_snrs])
        labels_all = np.array(train_labels + val_labels + test_labels)

        for i in range(len(X_all)):
            mod = labels_all[i]
            snr = int(snrs_all[i])
            key = (mod, snr)

            if key not in Xd:
                Xd[key] = []

            Xd[key].append(X_all[i])

        # Convert lists to numpy arrays
        for key in Xd:
            Xd[key] = np.array(Xd[key])

        # Create list of all SNRs
        snrs = sorted(np.unique(target_snrs).tolist())

        return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx,
                                                                                         test_idx), Xd

    @staticmethod
    def load_rml_dataset_npz(filename=r'path/to/Indoor-OTA-12Mod.h5'):
        """
        Load dataset from NPZ file
        """
        print(f"Loading NPZ dataset: {filename}")

        data = np.load(filename, allow_pickle=True)

        # Read data
        X_train = data['train_data']
        X_val = data['val_data']
        X_test = data['test_data']

        # Read labels
        train_labels = data['train_labels']
        val_labels = data['val_labels']
        test_labels = data['test_labels']

        # Read SNR labels
        train_snrs = data['train_snrs']
        val_snrs = data['val_snrs']
        test_snrs = data['test_snrs']

        # Read modulation types
        mods = data['modulation_types'].tolist()

        # Read target SNR range
        target_snrs = data['target_snrs']

        # Reshape data to (number of samples, 2, 128) format
        def reshape_to_iq(data):
            if data.shape[1] == 256:
                return data.reshape(-1, 2, 128)
            return data

        X_train = reshape_to_iq(X_train)
        X_val = reshape_to_iq(X_val)
        X_test = reshape_to_iq(X_test)

        print(f"Data loading completed!")
        print(f"Modulation types: {mods}")
        print(f"SNR range: {sorted(np.unique(target_snrs).tolist())}")
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")

        # Create one-hot encoded labels
        def to_onehot(labels, mods_list):
            num_classes = len(mods_list)
            onehot = np.zeros((len(labels), num_classes))

            # Create mapping from modulation type to index
            mod_to_idx = {mod: idx for idx, mod in enumerate(mods_list)}

            for i, label in enumerate(labels):
                idx = mod_to_idx[label]
                onehot[i, idx] = 1

            return onehot

        Y_train = to_onehot(train_labels, mods)
        Y_val = to_onehot(val_labels, mods)
        Y_test = to_onehot(test_labels, mods)

        # Create indices
        train_len = X_train.shape[0]
        val_len = X_val.shape[0]
        test_len = X_test.shape[0]

        train_idx = list(range(0, train_len))
        val_idx = list(range(train_len, train_len + val_len))
        test_idx = list(range(train_len + val_len, train_len + val_len + test_len))

        # Create label list (for dictionary structure)
        lbl = []
        snrs_all = np.concatenate([train_snrs, val_snrs, test_snrs])
        labels_all = np.concatenate([train_labels, val_labels, test_labels])

        for label, snr in zip(labels_all, snrs_all):
            lbl.append((label, int(snr)))

        # Create data dictionary (grouped by (mod, snr))
        Xd = {}
        X_all = np.concatenate([X_train, X_val, X_test], axis=0)
        snrs_all = np.concatenate([train_snrs, val_snrs, test_snrs])
        labels_all = np.concatenate([train_labels, val_labels, test_labels])

        for i in range(len(X_all)):
            mod = labels_all[i]
            snr = int(snrs_all[i])
            key = (mod, snr)

            if key not in Xd:
                Xd[key] = []

            Xd[key].append(X_all[i])

        # Convert lists to numpy arrays
        for key in Xd:
            Xd[key] = np.array(Xd[key])

        # Create list of all SNRs
        snrs = sorted(np.unique(target_snrs).tolist())

        return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx,
                                                                                         test_idx), Xd


if __name__ == '__main__':

    # Load dataset
    try:
        (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx,
                                                                                  test_idx), Xd = DatasetLoader.load_rml_dataset()

        print(f"\nDataset loaded successfully!")
        print(f"Modulation types: {mods}")
        print(f"SNR values: {snrs}")
        print(f"Training set: {X_train.shape}, {Y_train.shape}")
        print(f"Validation set: {X_val.shape}, {Y_val.shape}")
        print(f"Test set: {X_test.shape}, {Y_test.shape}")

        # Test data dictionary
        print(f"\nData dictionary example:")
        for i, key in enumerate(list(Xd.keys())[:3]):
            print(f"  {key}: {Xd[key].shape}")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()