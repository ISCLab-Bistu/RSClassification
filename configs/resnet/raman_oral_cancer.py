_base_ = [
    '../_base_/models/resnet/oral_cancer_resnet.py', '../_base_/datasets/raman_oral_cancer.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'oral_cancer_resnet'
