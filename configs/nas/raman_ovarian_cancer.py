_base_ = [
    '../_base_/datasets/raman_ovarian_cancer.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='RamanClassifier',
    backbone=dict(type='AugmentCNN',
                  input_size=1480,
                  input_channels=1,
                  init_channels=16,
                  n_classes=2,
                  n_layers=16),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)

work_dir = 'ovarian_cancer_augment'
