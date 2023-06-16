_base_ = [
    '../_base_/models/resnetV2/oral_cancer_resnetV2.py', '../_base_/datasets/raman_oral_cancer.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'oral_cancer_resnetV2'
