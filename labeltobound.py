from skimage import segmentation
import numpy as np


class StandardLabelToBoundary:
    def __init__(self, ignore_index=None, append_label=False, blur=False, sigma=1, mode='thick', foreground=False,
                 **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.blur = blur
        self.sigma = sigma
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype('int32')
        if self.blur:
            boundaries = blur_boundary(boundaries, self.sigma)

        results = []
        if self.foreground:
            foreground = (m > 0).astype('uint8')
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)

def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input

def ltb (m):
    boundaries = segmentation.find_boundaries(m, connectivity=2, mode='thick')
    boundaries = boundaries.astype('int32')
    results = []
    results.append(_recover_ignore_index(boundaries, m, None))
    return np.stack(results, axis=0)

