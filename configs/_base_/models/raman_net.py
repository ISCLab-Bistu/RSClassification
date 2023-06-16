# Basic configuration（Reserved edition）
model = dict(
    type='DataClassifier',  # 
    backbone=dict(
        type='ResNet',  # Backbone network type
        # depth=50,  # Backbone network depth， ResNet Generally have18, 34, 50, 101, 152 Can choose
        num_stages=4,  # Backbone network state(stages)， head The input of。
        out_indices=(3,),  # Output feature map output index。The further away from the input image，Larger index
        frozen_stages=-1,
        # Network fine-tuning time，stage（The inverse propagation algorithm is not implemented during training），ifnum_stages=4，backbonecontainstem with 4 a stages。frozen_stages-1when，Unfrozen network； for0when，Frozen stem； 1when，Frozen stem and stage1； for4when，backbone
        style='pytorch'),  # ，'pytorch' It means the step size is2The layer of is 3x3 convolution， 'caffe' It means the step size is2The layer of is 1x1 。
    neck=dict(type='GlobalAveragePooling'),  # Neck network type
    head=dict(
        type='LinearClsHead',  # Linear sort head，
        num_classes=2,  # Number of output classes，This is consistent with the number of categories in the dataset
        in_channels=2048,  # Input channel number，This is related to... neck The output channel is consistent
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),  # Loss function configuration information
        topk=(1, 5),  # Evaluation index，Top-k Accuracy rate，  top1 with top5 Accuracy rate
    ))
