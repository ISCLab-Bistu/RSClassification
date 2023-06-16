# Be used foroptim_wrapperBuild the configuration file encapsulated by the optimizer。support PyTorch All optimizers in，At the same time their parameters and PyTorch 。
# Base configuration (keep comments)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    scheduler=dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1),
    paramwise_cfg=dict(norm_decay_mult=0))
# Configuration file for the optimizer hook
# optimizer_config = dict(grad_clip=None)  # (grad_clip)。
# The learning rate adjustment configuration is used to register the LrUpdater hook
# lr_config = dict(policy='step',  # The scheduler also supports CosineAnnealing, Cyclic, etc.
#                  step=[30, 60, 90])  #  At epochs of 30, 60, and 90, lr decays
runner = dict(type='EpochBasedRunner',
              max_epochs=100)
