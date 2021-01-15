import SimpleITK as sitk
import tifffile
import os, types
import concurrent.futures

ROOT_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = 'F:/TEST_DATA/visor_brain_test_data'

VERSION = '0.5.9'

def image_generator_func(dst:list, func:callable, args=None, kwargs=None, rerun=False) -> dict:
    imgmap = {}
    if not rerun:
        try:
            for d in dst:
                if d is not None:
                    imgmap[d] = sitk.ReadImage(d)
            while 1:
                yield imgmap
        except RuntimeError as e:
        #    print(e)
            pass
        #except Exception as e:
        #    print(e)
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    for i in range(len(args)):
        if isinstance(args[i], types.GeneratorType):
            args[i] = args[i].__next__()
    for k in kwargs:
        if isinstance(kwargs[k], types.GeneratorType):
            kwargs[k] = kwargs[k].__next__()
    imgs = func(*args, **kwargs)
    if type(imgs) is sitk.Image:
        imgmap[dst[0]] = imgs
    else:
        for i in range(len(dst)):
            imgmap[dst[i]] = imgs[i]
    for d in dst:
        if d is None:
            continue
        sitk.WriteImage(imgmap[d], d)
    while 1:
        yield imgmap


def image_generator(key, gen):
    while 1:
        yield gen.__next__()[key]


def create_image_generator(dst:list, func:callable, args=None, kwargs=None, rerun=False):
    gen = image_generator_func(dst, func, args, kwargs, rerun)
    return {d:image_generator(d, gen) for d in dst}
