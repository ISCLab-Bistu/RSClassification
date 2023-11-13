_base_ = [
    '../_base_/datasets/raman_single_cell.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RamanClassifier',
    backbone=dict(type='AugmentCNN',
                  input_size=815,
                  input_channels=1,
                  init_channels=64,
                  n_classes=6,
                  n_layers=8),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)

work_dir = 'single_cell_augment'
