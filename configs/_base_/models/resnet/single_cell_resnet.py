model = dict(
    type='RamanClassifier',  # Classifier type
    backbone=dict(
        type='ResNet',  # Backbone network type
        depth=50,
        strides=(1, 2, 2, 2),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)

