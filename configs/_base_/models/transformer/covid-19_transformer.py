# ResNetV2
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='Ml4fTransformer', input_dim=900, num_classes=3),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=900,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)
