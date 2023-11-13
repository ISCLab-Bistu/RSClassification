# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Normalize', method='intensity'),  # normalization
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
    # dict(type='Collect', keys=['data', 'labels'])  # 
]
# 
test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=32,  # single GPU the Batch size
    workers_per_gpu=2,  # single GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/LiNbO3/results/linbo3.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/LiNbO3/results/linbo3.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/LiNbO3/results/linbo3.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)
