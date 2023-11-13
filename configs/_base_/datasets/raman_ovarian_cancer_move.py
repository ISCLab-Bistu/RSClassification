# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='AddNoise'),
    dict(type='MoveRaman', move_ranges=[(600, 800), (800, 1400), (1400, 1800)]),
    # dict(type='IntensityFactory'),
    dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    dict(type='RemoveBaseline', roi=[[400, 2003]], method='als', lam=10 ** 5, p=0.05),
    dict(type='Normalize', method='intensity'),  #
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='AddNoise'),
    dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    dict(type='RemoveBaseline', roi=[[400, 2003]], method='als', lam=10 ** 5, p=0.05),
    dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=64,  # single GPU the Batch size
    workers_per_gpu=2,  # GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/ovarian_cancer/results/ovarian_cancer.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/ovarian_cancer/results/ovarian_cancer.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/ovarian_cancer/results/ovarian_cancer.csv',
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
