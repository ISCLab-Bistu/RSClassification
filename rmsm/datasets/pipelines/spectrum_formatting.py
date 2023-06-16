import numpy as np

from ..builder import PIPELINES
from ..rampy import flipsp, resample


@PIPELINES.register_module()
class Flipping(object):

    def __call__(self, results):
        x = results['raman_shift']
        # Flip spectrum
        raman_shift = flipsp(x)
        results['raman_shift'] = raman_shift

        return results


@PIPELINES.register_module()
class Resampling(object):
    def __init__(self, start=400, end=1300, step=1.0, **kwargs):
        self.start = start
        self.end = end
        self.step = step

    def __call__(self, results):
        spectrum = []
        raman_shift = []
        x = results['raman_shift']
        for i in range(len(results['labels'])):
            y = results['spectrum'][i]
            # Resampled spectrum
            x_new = np.arange(self.start, self.end, self.step)  # we generate the new X values with numpy.arange()
            y_new = resample(x, y, x_new, fill_value="extrapolate")
            spectrum.append(y_new)
            raman_shift.append(x_new)

        spectrum = np.array(spectrum)  # numpy
        raman_shift = np.array(raman_shift)

        results['spectrum'] = spectrum
        results['raman_shift'] = raman_shift

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'start={self.start}, '
                    f"end='{self.end}', "
                    f'step={self.step})')
        return repr_str


@PIPELINES.register_module()
class SpectrumToZero(object):
    """Format the corresponding spectrum, Set the negative spectral value to zero.

    """

    def __call__(self, results):
        spectrum = []
        x = results['raman_shift']
        for i in range(len(results['labels'])):
            y = results['spectrum'][i]
            y[y < 0] = 0
            spectrum.append(y)

        spectrum = np.array(spectrum)

        results['spectrum'] = spectrum

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        return repr_str
