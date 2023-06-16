model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='AlexNet',
        num_classes=6,
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)
