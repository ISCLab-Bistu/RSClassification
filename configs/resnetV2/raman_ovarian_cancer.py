_base_ = [
    '../_base_/models/resnetV2/ovarian_cancer_resnetV2.py', '../_base_/datasets/raman_ovarian_cancer.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'ovarian_cancer_resnetV2'
