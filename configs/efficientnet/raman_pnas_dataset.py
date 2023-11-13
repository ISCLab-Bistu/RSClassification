_base_ = [
    '../_base_/models/efficientnet/pnas_dataset_efficientnet.py', '../_base_/datasets/raman_pnas_dataset.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'pnas_dataset_efficientnet'
