# dataset settings
dataset_type = 'RamanSpectral'  # 

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='Resampling', start=1000, end=4000),
    dict(type='Normalize', method='intensity'),  # normalization
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
    # dict(type='Collect', keys=['data', 'labels'])
    # The process of deciding which keys in the data should be passed to the detector
]
# 
test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=128,  # single GPU the Batch size
    workers_per_gpu=2,  # single GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/LR-Raman/raman_data/lr-raman_test.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/LR-Raman/raman_data/lr-raman_test.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/LR-Raman/raman_data/lr-raman_test.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)
