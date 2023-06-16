_base_ = [
    '../_base_/models/vgg/tumor_vgg.py', '../_base_/datasets/raman_tumor.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'tumor_vgg'
