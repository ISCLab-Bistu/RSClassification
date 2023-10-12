model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='Ml4fTransformer',
        input_dim=815,
        num_classes=6,
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=815,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)
