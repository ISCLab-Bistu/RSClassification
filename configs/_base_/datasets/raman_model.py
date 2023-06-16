# Basic configuration（）
dataset_type = 'RamanSpectral'  # 

data_norm_cfg = dict(  # Normalized configuration，Used to normalize Raman data。
    mean=[123.675],  # The average value used to pre-train the backbone network model in pre-training。
    std=[58.395])  # The standard deviation used to pre-train the backbone network model in pre-training。
# Training data pipeline
train_pipeline = [
    dict(type='load_data'),  # 
    # dict(type='FlipSpx'),  # flipxshaft
    # dict(type='Resample'),  # Resampling spectrum（These two won't be needed for a while）
    dict(type='Normalisation', **data_norm_cfg),  # normalization
    dict(type='Smooth'),  # 
    dict(type='BaseLine'),  # Baseline removal
    dict(type='DataToTensor', keys=['data']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
    dict(type='Collect', keys=['data', 'labels'])
    # The process of deciding which keys in the data should be passed to the detector
]
# Test data pipeline
test_pipeline = [
    dict(type='load_data'),
    dict(type='Normalisation', **data_norm_cfg),
    dict(type='Smooth'),  # 
    dict(type='BaseLine'),  # 
    dict(type='DataToTensor', keys=['data']),
    dict(type='Collect', keys=['data'])  # test Time-invariant transmission labels
]
data = dict(
    samples_per_gpu=32,  # single GPU the Batch size
    workers_per_gpu=2,  # single GPU the 
    train=dict(  # Training data information
        type=dataset_type,  # Data set name
        # data_prefix='data/single_cell/train',  # Data set catalog，When not exist ann_file when，
        pipeline=train_pipeline),  # The data set needs to go through Data pipeline
    val=dict(  # Verify data set information
        type=dataset_type,
        # data_prefix='data/single_cell/val',
        # ann_file='data/single_cell/meta/val.txt',  # Mark file path，exist ann_file ，Category information is not automatically obtained through folders
        pipeline=test_pipeline),
    test=dict(  # Test data set information
        type=dataset_type,
        # data_prefix='data/single_cell/val',
        # ann_file='data/single_cell/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(  # evaluation hook The configuration of
    interval=1,  # Interval during validation，The unit is epoch  iter， Depend on runner type。
    metric='accuracy')  # Metrics used during validation。
