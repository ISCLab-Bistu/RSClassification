_base_ = [
    '../_base_/models/resnetV2/covid-19_resnetV2.py', '../_base_/datasets/raman_covid-19.py',
    '../_base_/schedules/raman_bs128.py', '../_base_/default_runtime.py'
]

work_dir = 'covid-19_resnetV2'
