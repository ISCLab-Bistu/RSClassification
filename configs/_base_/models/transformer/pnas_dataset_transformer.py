# ResNetV2
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='Ml4fTransformer', input_dim=1193, num_classes=2),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1193,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)
