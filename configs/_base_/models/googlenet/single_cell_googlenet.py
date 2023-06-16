# googlenet needs to modify the number of training rounds > 1000
model = dict(
    type='RamanClassifier',
    backbone=dict(
        type='GoogLeNet',
        num_classes=6,
        input_dim=815,
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)
