# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# 
train_pipeline = [
    dict(type='LoadDataFromFile'),  # Read data
    # dict(type='FlipSpx'),  # flipxshaft
    # dict(type='Resample'),  # Resampling spectrum（These two won't be needed for a while）
    # dict(type='Normalize', method='intensity'),  # 
    # dict(type='Smooth'),  # Smooth spectrum
    dict(type='RemoveBaseline', roi=[[400, 500], [750, 1000], [1400, 1500], [1500, 2500], [3000, 3200]], method='poly',
         polynomial_order=4),  # Baseline removal
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
    # dict(type='Collect', keys=['spectrum', 'labels'])  #
    # The process of deciding which keys in the data should be passed to the detector
]
# 
test_pipeline = [
    dict(type='LoadDataFromFile'),
    # dict(type='Normalize', method='intensity'),
    # dict(type='Smooth'),  # Smooth spectrum
    # dict(type='BaseLine'),  # Baseline removal
    dict(type='DataToFloatTensor', keys=['spectrum']),
    # dict(type='Collect', keys=['spectrum'])  # test Time-invariant transmission labels
]
data = dict(
    samples_per_gpu=32,  # single GPU the Batch size
    workers_per_gpu=2,  # GPU the Thread count
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/formic_acid/results/formic_acid.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 0.9),
        file_path='data/formic_acid/results/formic_acid.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_size=(0.9, 1),
        file_path='data/formic_acid/results/formic_acid.csv',
        pipeline=test_pipeline,
        test_mode=True
    )
)
