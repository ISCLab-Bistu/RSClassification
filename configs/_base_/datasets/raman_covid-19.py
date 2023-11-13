# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='AddNoise', noise_std=0.0005),
    dict(type='MoveRaman', max_shift=2),
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[400, 2112]], method='als', lam=10 ** 5, p=0.05),
    dict(type='Normalize', method='minmax'),  # normalization()
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data  torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
    # dict(type='Collect', keys=['data', 'labels'])
    # The process of deciding which keys in the data should be passed to the detector
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='MoveRaman'),
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[400, 2112]], method='als', lam=10 ** 5, p=0.05),
    dict(type='Normalize', method='minmax'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=32,  # single GPU the Batch size
    workers_per_gpu=2,  # GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/COVID-19/results/covid-19_test.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/COVID-19/results/covid-19_test.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/COVID-19/results/covid-19_test.csv',
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
