# dataset settings
dataset_type = 'RamanSpectral'  # Data set name

# Training data pipeline
train_pipeline = [
    dict(type='LoadDataFromFile'),  #
    # dict(type='Smoothing', method="savgol", window_length=5, polyorder=2),
    # dict(type='RemoveBaseline', roi=[[0, 4000]], method='airPLS', lam=10 ** 5, p=0.05),
    dict(type='Normalize', method='minmax'),  # normalization
    dict(type='DataToFloatTensor', keys=['spectrum']),  # data Turn into torch.Tensor
    dict(type='ToTensor', keys=['labels']),  # labels Turn into torch.Tensor
]
# Test data pipeline
test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Normalize', method='minmax'),
    dict(type='DataToFloatTensor', keys=['spectrum']),
]
data = dict(
    samples_per_gpu=32,  # single GPU the Batch size
    workers_per_gpu=2,  # single GPU the
    train=dict(
        type=dataset_type,
        data_size=(0, 0.7),
        file_path='data/single_cell/results/single_cell.csv',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_size=(0.7, 1),
        file_path='data/single_cell/results/single_cell.csv',
        pipeline=test_pipeline,
    ),
)

model = dict(
    type='SearchCNNController',
    input_channels=1,
    init_channels=16,
    n_classes=6,
    n_layers=8,
    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
)

work_dir = 'single_cell_darts'

# Log configuration information。
log_config = dict(
    interval=100,  # Interval for printing logs，  iters
    hooks=[
        dict(type='TextLoggerHook'),  # Text recorder for recording training process(logger)。
        # dict(type='TensorboardLoggerHook')  # Equally support Tensorboard Logs
    ])
log_level = 'INFO'  # Log output level
epochs = 300
print_freq = 50  # print frequency

# weight optimizer
w_optimizer = dict(
    w_lr=0.025, w_lr_min=0.001, w_momentum=0.9, w_weight_decay=3e-4, w_grad_clip=5
)

# alpha optimizer
alpha_optimizer = dict(
    alpha_lr=0.025, alpha_weight_decay=0.001
)
