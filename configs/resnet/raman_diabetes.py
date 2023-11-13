_base_ = [
    '../_base_/models/diabetes_resnet.py', '../_base_/datasets/raman_diabetes.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'diabetes'
