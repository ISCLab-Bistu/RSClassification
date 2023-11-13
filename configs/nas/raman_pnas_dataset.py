_base_ = [
    '../_base_/datasets/raman_pnas_dataset.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RamanClassifier',
    backbone=dict(type='AugmentCNN',
                  input_size=815,
                  input_channels=1,
                  init_channels=32,
                  n_classes=2,
                  n_layers=8),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)

work_dir = 'pnas_dataset_augment'
