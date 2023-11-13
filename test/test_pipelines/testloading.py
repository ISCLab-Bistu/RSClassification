from rmsm import build_from_cfg

from rmsm.datasets.builder import PIPELINES
import rmsm.datasets.pipelines

load = dict(type='LoadDataFromFile')
load = build_from_cfg(load, PIPELINES)
print("-----------------------")
print(load)
