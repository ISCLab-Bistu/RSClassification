optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='step', step=[40, 70, 90])
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.002,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner',
              max_epochs=300)
