# VGG
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='VGG', depth=19, norm_cfg=dict(type='BN1d'), num_classes=6, input_linear=25),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)
