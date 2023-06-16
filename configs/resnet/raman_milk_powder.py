_base_ = [
    '../_base_/models/milk_powder_resnet.py', '../_base_/datasets/raman_milk_powder.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'milk_powder'
