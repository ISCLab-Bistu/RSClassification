_base_ = [
    '../_base_/models/mobilenet/pnas_dataset_mobilenetV2.py', '../_base_/datasets/raman_pnas_dataset.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'pnas_dataset_mobilenetV2'
