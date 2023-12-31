# ResNetV2
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='Ml4fTransformer', input_dim=2090, num_classes=5),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2090,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)