_base_ = [
    '../_base_/models/resnet50/pnas_dataset_resnet.py', '../_base_/datasets/raman_pnas_dataset.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = '50pnas_dataset_resnet'
