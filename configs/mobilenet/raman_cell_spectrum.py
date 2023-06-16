_base_ = [
    '../_base_/models/mobilenet/cell_spectrum_mobilenetV2.py', '../_base_/datasets/raman_cell_spectrum.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'cell_spectrum_mobilenetV2'
