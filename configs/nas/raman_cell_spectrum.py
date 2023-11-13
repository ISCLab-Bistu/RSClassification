_base_ = [
    '../_base_/datasets/raman_cell_spectrum.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RamanClassifier',
    backbone=dict(type='AugmentCNN',
                  input_size=2090,
                  input_channels=1,
                  init_channels=32,
                  n_classes=5,
                  n_layers=32),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)

work_dir = 'cell_spectrum_augment'
