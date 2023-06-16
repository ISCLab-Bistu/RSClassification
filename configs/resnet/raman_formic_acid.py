_base_ = [
    '../_base_/models/formic_acid_resnet.py', '../_base_/datasets/raman_formic_acid.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'formic_acid'
