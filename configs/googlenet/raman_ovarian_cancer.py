_base_ = [
    '../_base_/models/googlenet/ovarian_cancer_googlenet.py', '../_base_/datasets/raman_ovarian_cancer.py',
    '../_base_/schedules/raman_trans.py', '../_base_/default_runtime.py'
]

work_dir = 'ovarian_cancer_googlenet'
