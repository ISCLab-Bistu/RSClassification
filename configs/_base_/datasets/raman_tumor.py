# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='Resampling', start=397.79, end=3282.36, step=5.79),
    dict(type='Normalize', method='intensity'),  # normalization
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels  torch.Tensor
    # dict(type='Collect', keys=['data', 'labels'])  # 
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='Resampling', start=397.79, end=3282.36, step=5.79),  # Attention sumtrainBe consistent
    dict(type='Normalize', method='intensity'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]

data = dict(
    samples_per_gpu=256,  # single GPU the Batch size
    workers_per_gpu=4,  #  GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/tumor/results/tumor.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/tumor/results/tumor.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/tumor/results/tumor.csv',
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