_base_ = [
    '../_base_/models/lr-raman_resnet.py', '../_base_/datasets/raman_lr-raman.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'lr-raman'
