_base_ = [
    '../_base_/models/resnet/ovarian_cancer_resnet.py', '../_base_/datasets/raman_ovarian_cancer.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'ovarian_cancer_resnet'
