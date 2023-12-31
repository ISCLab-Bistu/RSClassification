# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='AddNoise'),
    # dict(type='MoveRaman', move_ranges=[(300, 1000), (1000, 2000), (2000, 3000), (3000, 3500)]),
    # dict(type='IntensityFactory'),
    # dict(type='Resampling', start=0, end=1840),
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[0, 2089]], method='als', lam=10 ** 5, p=0.05),
    dict(type='Normalize', method='minmax'),  # normalization
    # dict(type='GANRaman', train_label=0),  # GAN Network
    # dict(type='GANRaman', train_label=1),
    # dict(type='GANRaman', train_label=2),
    # dict(type='GANRaman', train_label=3),
    # dict(type='GANRaman', train_label=4),
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels  torch.Tensor
    # dict(type='Collect', keys=['data', 'labels'])
    # The process of deciding which keys in the data should be passed to the detector
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='AddNoise', noise_std=0.5),
    # dict(type='Resampling', start=0, end=1840),
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[0, 2089]], method='als', lam=10 ** 5, p=0.05),
    dict(type='Normalize', method='minmax'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=16,  # single GPU the Batch size
    workers_per_gpu=2,  # single GPU  Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/cell_spectrum/results/cell_spectrum.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/cell_spectrum/results/cell_spectrum.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/cell_spectrum/results/cell_spectrum.csv',
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
