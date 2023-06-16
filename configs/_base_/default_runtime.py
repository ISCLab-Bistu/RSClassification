# Checkpoint hook Configuration file of。
checkpoint_config = dict(
    interval=1)  # Save interval is 1，Unit will be based on runner Different variation，Can be for epoch perhaps iter。
# Log configuration information。
log_config = dict(
    interval=100,  # Interval for printing logs，  iters
    hooks=[
        dict(type='TextLoggerHook'),  # Text recorder for recording training process(logger)。
        # dict(type='TensorboardLoggerHook')  # Equally support Tensorboard Logs
    ])

launcher = 'pytorch'
log_level = 'INFO'  # Log output level。
resume_from = None  # Restores a checkpoint from a given path(checkpoints)，。
load_from = None
workflow = [('train', 2), ('val', 1)]  # runner Workflow of
