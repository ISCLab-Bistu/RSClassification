model = dict(
    type='RamanClassifier',  # Classifier type
    backbone=dict(
        type='ResNetV2',  # Backbone network type
        input_dim=2890,
        num_classes=2),
    loss=dict(type='CrossEntropyLoss', loss_weight=1.0)  # Loss function configuration information
)
