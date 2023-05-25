import mmcv
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PadShortSide(object):
    """Pad the image & mask.

    Pad to the minimum size that is equal or larger than a number.
    Added keys are "pad_shape", "pad_fixed_size",

    Args:
        size (int, optional): Fixed padding size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self, size=None, pad_val=0, seg_pad_val=255):
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        h, w = results['img'].shape[:2]
        new_h = max(h, self.size)
        new_w = max(w, self.size)
        padded_img = mmcv.impad(results['img'],
                                shape=(new_h, new_w),
                                pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        # results['unpad_shape'] = (h, w)

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_results', []):
            results[key] = mmcv.impad(results[key],
                                      shape=results['pad_shape'][:2],
                                      pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
          results (dict): Result dict from loading pipeline.

        Returns:
          dict: Updated result dict.
        """
        h, w = results['img'].shape[:2]
        if h >= self.size and w >= self.size:
            pass
        else:
            self._pad_img(results)
            self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, pad_val={self.pad_val})'
        return repr_str
