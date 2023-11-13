# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='Resampling', start=800, end=1800),
    # dict(type='SpectrumToZero'),
    dict(type='Normalize', method='intensity'),  # 
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='Resampling', start=800, end=1800),  # Attention sumtrainBe consistent
    # dict(type='SpectrumToZero'),
    dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=128,  # GPU  Batch size
    workers_per_gpu=2,  # single GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/diabetes/results/diabetes.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/diabetes/results/diabetes.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/diabetes/results/diabetes.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)

evaluation = dict(  # Computational accuracy
    interval=1,
    metric='accuracy',
    metric_options={'topk': (1,)},
    save_best="auto",
    start=1
)
