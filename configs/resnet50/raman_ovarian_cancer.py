_base_ = [
    '../_base_/models/resnet50/ovarian_cancer_resnet.py', '../_base_/datasets/raman_ovarian_cancer.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = '50ovarian_cancer_resnet'
