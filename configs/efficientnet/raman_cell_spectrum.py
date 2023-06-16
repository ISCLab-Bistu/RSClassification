_base_ = [
    '../_base_/models/efficientnet/cell_spectrum_efficientnet.py', '../_base_/datasets/raman_cell_spectrum.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'cell_spectrum_efficientnet'
