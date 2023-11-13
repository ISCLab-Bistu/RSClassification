from sklearn.model_selection import train_test_split


# dataframe
def split_dataframe(data, split_ratio=(0.7, 0.3)):
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=0, stratify=data.labels)

    return data_train, data_test
