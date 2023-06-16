# ResNetV2
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='ResNetV2', input_dim=1193, num_classes=2),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)
