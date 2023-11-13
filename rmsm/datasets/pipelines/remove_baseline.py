import numpy as np

from ..builder import PIPELINES
from ..rampy import baseline


@PIPELINES.register_module()
class RemoveBaseline(object):
    def __init__(self, roi=[[0, 100], [200, 220], [280, 290], [420, 430], [480, 500]], method='poly', **kwargs):
        self.method = method
        self.roi = roi
        self.kwargs = kwargs

    def __call__(self, results):
        # results
        data = []
        x = results['raman_shift']
        print("Baseline")
        for i in range(len(results['labels'])):
            y = results['spectrum'][i]
            # baseline correction
            roi = np.array(self.roi)
            ycalc_poly, base_poly = baseline(x, y, roi, method=self.method, **self.kwargs)  # 
            ycalc_poly = np.squeeze(ycalc_poly)
            data.append(ycalc_poly)
        data = np.array(data)  # Convert to numpy
        results['spectrum'] = data

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'method={self.method}, '
                    f'roi={self.roi})')
        return repr_str
