_base_ = [
    '../_base_/models/efficientnet/covid-19_efficientnet.py', '../_base_/datasets/raman_covid-19.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'covid-19_efficientnet'
