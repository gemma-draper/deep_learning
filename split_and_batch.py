import torch
import multiprocessing

def split_data(data, train_size=0.8, val_size=0.1):
    """
    Input: data tensor.
    Output: train, validation and test data tensors.
    """
    # calculate size of data splits
    n = len(data)
    train_len = int(n * train_size)
    val_len = int(n * val_size)

    # split the data
    train_set, val_set, test_set = torch.utils.data.random_split(data, [train_len, val_len, len(data) - train_len - val_len])
    
    return train_set, val_set, test_set

def batch_data(data, batch_size=32, shuffle=False):
    """
    Given data tensor, returns generator for batched data.
    """
    # assign half the available cpu for batching 
    cpu = multiprocessing.cpu_count()/2

    # create loader
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=cpu)

    return data_loader

def split_and_batch(data, batch_size=32, shuffle_train=True):
    """
    Given a data tensor, split into train, val and test sets.
    Return batched data generator for each set.
    """

    train_set, val_set, test_set = split_data(data)
    train_loader = batch_data(train_set, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = batch_data(val_set, batch_size=batch_size)
    test_loader = batch_data(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader
    
if __name__=="__main__":
    pass