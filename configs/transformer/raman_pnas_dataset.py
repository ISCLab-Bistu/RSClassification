_base_ = [
    '../_base_/models/transformer/pnas_dataset_transformer.py', '../_base_/datasets/raman_pnas_dataset.py',
    '../_base_/schedules/raman_trans.py', '../_base_/default_runtime.py'
]

work_dir = 'pnas_dataset_transformer'
