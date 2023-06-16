# ResNet50
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)
