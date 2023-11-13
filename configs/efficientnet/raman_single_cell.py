_base_ = [
    '../_base_/models/efficientnet/single_cell_efficientnet.py', '../_base_/datasets/raman_single_cell.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'single_cell_efficientnet'
