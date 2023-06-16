# dataset settings
dataset_type = 'RamanSpectral'  # 

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='FlipSpx'),  # flipxshaft
    # dict(type='Resample'),  # Resampling spectrum（These two won't be needed for a while）
    dict(type='Normalize', method='intensity'),  # 
    # dict(type='Smooth'),  # 
    # dict(type='BaseLine'),  # Baseline removal
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
    # dict(type='Collect', keys=['data', 'labels'])
    # The process of deciding which keys in the data should be passed to the detector
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=256,  # GPU the Batch size
    workers_per_gpu=2,  # single GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/Microplastics/results/Microplastics.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/Microplastics/results/Microplastics.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/Microplastics/Microplastics_test.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)
