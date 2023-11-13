_base_ = [
    '../_base_/models/googlenet/single_cell_googlenet.py', '../_base_/datasets/raman_single_cell.py',
    '../_base_/schedules/raman_trans.py', '../_base_/default_runtime.py'
]

work_dir = 'single_cell_googlenet'
