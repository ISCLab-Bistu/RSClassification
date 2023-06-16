model = dict(
    type='RamanClassifier',  # 
    backbone=dict(
        type='ResNet',
        depth=50,
        strides=(1, 2, 2, 2),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)
