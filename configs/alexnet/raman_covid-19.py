_base_ = [
    '../_base_/models/alexnet/covid-19_alexnet.py', '../_base_/datasets/raman_covid-19.py',
    '../_base_/schedules/raman_trans.py', '../_base_/default_runtime.py'
]

work_dir = 'covid-19_alexnet'
