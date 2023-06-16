_base_ = [
    '../_base_/models/microplastics_resnet.py', '../_base_/datasets/raman_microplastics.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'Microplastics'  # The directory file address used to hold the model checkpoints and logs for the current experiment。
