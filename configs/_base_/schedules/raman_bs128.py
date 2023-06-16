# Optimizer configuration
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.01)
# optimizer = dict(type='AdamW', lr=0.005, weight_decay=0.0002)
optimizer_config = dict(grad_clip=None)
# Parameter learning strategy
# lr_config = dict(policy='step', step=[40, 70, 90])
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner',  # Class of runner that will be used, such as IterBasedRunner or
              # EpochBasedRunner.
              max_epochs=400)  # Total runner turns, using 'max_iters' for IterBasedRunner
