# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    dict(type='RemoveBaseline', roi=[[-29, 4096]], method='arPLS', lam=10 ** 6, ratio=0.001),
    dict(type='Normalize', method='intensity'),  # 
    # dict(type='Resampling', start=397.79, end=4000.64, step=5.79),
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data  torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
    # dict(type='Collect', keys=['data', 'labels'])  # The process of deciding which keys in the data should be passed to the detector
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    dict(type='RemoveBaseline', roi=[[-29, 4096]], method='arPLS', lam=10 ** 6, ratio=0.001),
    dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=128,  # single GPU the Batch size
    workers_per_gpu=2,  # GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/old_oral_cancer/results/oral_cancer1.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/old_oral_cancer/results/oral_cancer1.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/old_oral_cancer/results/oral_cancer1.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)

evaluation = dict(  # Computational accuracy
    interval=1,
    metric='precision',
    metric_options={'topk': (1,)},
    save_best="auto",
    start=1
)
