# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Normalize', method='intensity'),  # normalization
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data  torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels  torch.Tensor
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=32,  # single GPU the Batch size
    workers_per_gpu=2,  # single GPU the 
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/graphene/results/graphene.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/graphene/results/graphene.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/graphene/results/graphene.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)
