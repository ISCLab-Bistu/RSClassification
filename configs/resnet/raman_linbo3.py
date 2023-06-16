_base_ = [
    '../_base_/models/linbo3_resnet.py', '../_base_/datasets/raman_linbo3.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'lr-raman'
