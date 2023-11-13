_base_ = [
    '../_base_/models/transformer/cell_spectrum_transformer.py', '../_base_/datasets/raman_cell_spectrum.py',
    '../_base_/schedules/raman_trans.py', '../_base_/default_runtime.py'
]

work_dir = 'cell_spectrum_transformer'
