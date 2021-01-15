#from reconstruction import sample_reconstruct_methods
from VISoR_Brain.positioning.visor_sample import *
from VISoR_Brain.format.raw_data import RawData
import time, typing
from VISoR_Reconstruction.misc import ROOT_DIR
_all_methods = {}


def _get_all_methods():
    global _all_methods
    methods_root = os.path.join(ROOT_DIR, 'reconstruction', 'sample_reconstruct_methods')
    dirs = [d for d in os.listdir(methods_root)
            if os.path.isdir(os.path.join(methods_root, d))]
    _all_methods = {}
    for d in dirs:
        methods = {}
        files = os.listdir(os.path.join(methods_root, d))
        for f in files:
            if f.split('.')[-1] == 'py':
                try:
                    method = __import__('VISoR_Reconstruction.reconstruction.sample_reconstruct_methods.{}.{}'.format(d, f[:-3]),
                                        globals(), locals(), ['reconstruct']).reconstruct
                except AttributeError:
                    continue
                methods[f[:-3]] = method
        if len(methods) > 0:
            _all_methods[d] = methods


_get_all_methods()


def get_all_methods():
    return _all_methods


def create_reference(rawdata: RawData) -> VISoRSample:
    sample_data = VISoRSample()
    sample_data.column_source = rawdata.file
    sample_data.column_images = get_column_images(rawdata)
    r = sample_data.raw_data
    for i in range(len(r.columns)):
        affine_t = [0, 1 / r.pixel_size, 0,
                    0, 0, 1 / r.pixel_size / np.cos(r.angle),
                    1 / r.column_spacing[i], 0, 1 / r.column_spacing[i] / np.tan(r.angle)]

        p0 = r.column_pos0[i]
        af = sitk.AffineTransform(3)
        af.SetMatrix(affine_t)

        if r.column_spacing[i] < 0:
            p0 = (r.column_pos1[i][0], p0[1], p0[2])
        tl = np.subtract(r.pos0, p0).tolist()
        tl = af.TransformPoint(tl)
        af.Translate(tl)
        sample_data.transforms[i] = af
    sample_data.calculate_transforms()
    sample_data.calculate_spheres()
    return sample_data


def sitich(sample_data: VISoRSample, method: str = 'default', *args, **kwargs):
    if method == 'default':
        method = 'elastix_align'
    reconstruct = _all_methods['stitch'][method]
    reconstruct(sample_data, *args, *kwargs)
    sample_data.calculate_transforms()
    sample_data.calculate_spheres()
    return sample_data


def align_channels(sample_data: VISoRSample, reference: VISoRSample, method: str = 'default', *args, **kwargs):
    if method == 'default':
        method = 'channel_elastix_align'
    reconstruct = _all_methods['align_channels'][method]
    reconstruct(sample_data, reference, *args, **kwargs)
    sample_data.calculate_transforms()
    sample_data.calculate_spheres()
    sample_data.sphere = reference.sphere
    return sample_data


def reconstruct_image(sample_data: VISoRSample, pixel_size, roi=None, method: str='gpu_resample', rawdata: RawData=None,
                      *args, **kwargs):
    reconstruct = _all_methods['image_reconstruct'][method]
    if rawdata is not None:
        sample_data.column_source = rawdata.file
    sample_data.load_columns()
    if roi is None:
        roi = sample_data.sphere
    image = reconstruct(sample_data, roi, pixel_size, *args, **kwargs)
    return image


def reconstruct_sample(rawdata: RawData, methods: typing.Union[str, dict] = 'default',
                       reference_slice: VISoRSample = None, *args, **kwargs) -> VISoRSample:
    if methods == 'default':
        methods = {'stitch': 'default'}
    if methods == 'elastix_align':
        methods = {'stitch': 'elastix_align'}
    sample_data = VISoRSample()
    sample_data.column_source = rawdata.file
    sample_data.column_images = get_column_images(rawdata)
    sample_data = create_reference(rawdata)
    if 'stitch' in methods:
        sample_data = sitich(sample_data, methods['stitch'], *args, **kwargs)
    if 'align_channels' in methods and reference_slice is not None:
        sample_data = align_channels(sample_data, reference_slice, methods['align_channels'], *args, **kwargs)
    return sample_data

