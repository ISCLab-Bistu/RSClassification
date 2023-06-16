# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),  # 
    # dict(type='MoveRaman'),
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[600, 1750]], method='als', lam=10 ** 5, p=0.05),
    # dict(type='FlipSpx'),  # xshaft
    # dict(type='Resample'),  # Resampling spectrum（These two won't be needed for a while）
    dict(type='Normalize', method='intensity'),  # normalization
    # dict(type='Smooth'),  # Smooth spectrum
    # dict(type='BaseLine'),  # 
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
    # dict(type='Collect', keys=['spectrum', 'labels'])  # The process of deciding which keys in the data should be passed to the detector
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='MoveRaman'),
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[600, 1750]], method='als', lam=10 ** 5, p=0.05),
    dict(type='Normalize', method='intensity'),
    # dict(type='Smooth'),  # Smooth spectrum
    # dict(type='BaseLine'),  # 
    dict(type='DataToFloatTensor', keys=['spectrum']),
    # dict(type='Collect', keys=['data'])  # test Time-invariant transmission labels
]
data = dict(
    samples_per_gpu=64,  # single GPU the Batch size
    workers_per_gpu=2,  # single GPU the 
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/single_cell/results/single_cell.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/single_cell/results/single_cell.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.7, 1),
        file_path='data/single_cell/results/single_cell.csv',
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
