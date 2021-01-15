import json, re
import os, sys, multiprocessing
import numpy as np
import yaml, tifffile
import SimpleITK as sitk
from distutils.version import LooseVersion
from VISoR_Brain.lib import flsmio
import time, importlib


#import win32file

#win32file._setmaxstdio(2048)

class RawData:
    _rct = 0

    def __init__(self, data_file, device_file=None, real_path=None):
        self.name = os.path.basename(data_file).split('.')[0]
        self.device_file = device_file
        self.info = {}
        self.path = real_path
        self.file = data_file
        self.columns = []
        self.overlap_columns = []
        self.column_label = []
        self.column_pos0 = []
        self.column_pos1 = []
        self.roi = [0, 0, 2048, 2048]
        self.thumbnail_roi = [0, 0, 512, 512]
        self.x_coefficient = 1000
        self.y_coefficient = 1000
        self.pixel_size = 1
        self.column_spacing = []
        self.angle = -45 / 180 * np.pi
        self.pos0 = (0, 0, 0)
        self.pos1 = (0, 0, 0)
        self.image_files = {}
        self.scales = {}
        self.wave_length = 0
        self.caption = 'unknown'
        self.version = '0.0.0'
        self.z_index = -1
        self.size = [0, 0, 0, 0]
        self._load_func = self._load
        self._fastlsm_data(data_file, device_file)

        self._cached_thumb = None
        self._construct_count()

    @classmethod
    def _construct_count(cls):
        cls._rct += 1
        if cls._rct > 200:
            importlib.reload(flsmio)
            cls._rct = 0

    def _device(self, device_file):
        f = open(device_file)
        d = yaml.load(f)
        ch = d['channels'][0]
        self.x_coefficient = d['stage']['x_coefficient']
        self.y_coefficient = d['stage']['y_coefficient']
        self.roi = [ch['roi'][0], ch['roi'][1], ch['roi'][2], ch['roi'][3]]
        self.pixel_size =ch['pixel_size']
        self.angle = float(ch['angle']) / 180 * np.pi

    def _fastlsm_data(self, data_file, device_file=None):
        if self.path is None:
            self.path = os.path.dirname(data_file)
        f = open(data_file)
        flsm_info = json.load(f)
        self.info = flsm_info
        try:
            self.version = flsm_info['version']
        except:
            self.version = '1.0.0'

        # Fake raw data file
        if LooseVersion(self.version) < LooseVersion('2.4.0'):
            raise
            return
        else:
            self._fastlsm_data_v2(data_file)

    def _fastlsm_data_v2(self, flsm_file):

        self.angle = 45 / 180 * np.pi
        self._load_func = self._load

        reader = flsmio.FlsmReader(flsm_file)
        self.caption = reader.value('caption')
        self.wave_length = reader.value('wave_length')
        self.z_index = int(reader.value('slices_index'))
        spacing = float(reader.value("exposure")) * float(reader.value("velocity"))
        self.pixel_size = float(reader.value("pixel_size"))
        self.image_files['raw'] = reader
        self.image_files['thumbnail'] = reader
        all_size = reader.size()
        n_stacks = int(all_size[0])
        n_images = int(all_size[1])
        width = int(all_size[2])
        height = int(all_size[3])
        thumbnail_width = int(all_size[4])
        thumbnail_height = int(all_size[5])
        self.size = [n_stacks, width, height, n_images]
        self.scales['raw'] = 1
        self.scales['thumbnail'] = width / thumbnail_width
        self.image_files['raw'] = None
        self.image_files['thumbnail'] = None
        self.roi = [0, 0, width, height]
        self.thumbnail_roi = [0, 0, thumbnail_width, thumbnail_height]
        for i in range(n_stacks):
            self.column_spacing.append(spacing)
            #pos = None
            j = n_images // 2
            #for j in range(1, n_images):
            #    image = reader.thumbnail(i, j)
            #    if image is not None:
            #        pos = image.position()
            #        break
            image = reader.thumbnail(i, j)
            pos = image.position()
            pos0 = [pos[0] - j * spacing, pos[1], 0]
            pos1 = [pos[0] + n_images * spacing,
                    pos[1] + self.pixel_size * width,
                    np.cos(self.angle) * self.pixel_size * height]
            pos0, pos1 = [min(pos0[i], pos1[i]) for i in range(3)], [max(pos0[i], pos1[i]) for i in range(3)]
            self.column_pos0.append(pos0)
            self.column_pos1.append(pos1)
            self.columns.append(range(n_images))
        self.pos0 = [min(self.column_pos0, key=lambda x: x[i])[i] for i in range(3)]
        self.pos1 = [max(self.column_pos1, key=lambda x: x[i])[i] for i in range(3)]

    def release(self):
        if self._load_func == self._load:
            #self.image_files['raw'].release()
            del self.image_files['raw']
            self.image_files['raw'] = None

    def load(self, idx, range_=None, source_type='auto', **kwargs):
        if range_ == None:
            range_ = (0, len(self.columns[idx]))
        if isinstance(range_, int):
            range_ = [range_, range_ + 1]
        return self._load_func(idx, range_, source_type, **kwargs)

    def _load(self, idx, range_=None, source_type='auto', output_format='sitk'):
        if self.image_files['raw'] is None:
            reader = flsmio.FlsmReader(self.file)
            self.image_files['raw'] = reader
            self.image_files['thumbnail'] = reader
        #t1 = time.time()
        size = [self.roi[2], self.roi[3]]
        image_func = self.image_files['raw'].raw
        if source_type == 'thumbnail':
            size = [self.thumbnail_roi[2], self.thumbnail_roi[3]]
            image_func = self.image_files['raw'].thumbnail

        if output_format == 'sitk':
            images = []
            for i in range(0, range_[1] - range_[0]):
                image = image_func(idx, i + range_[0])
                if image is None:
                    image = sitk.Image(size, sitk.sitkUInt16)
                else:
                    image.decode()
                    image = np.array(image, copy=False)
                    image = sitk.GetImageFromArray(image)
                images.append(image)
            images = sitk.JoinSeries(images)
            if source_type == 'thumbnail':
                images.SetSpacing([self.scales[source_type], self.scales[source_type], 1])
        else:
            images = np.zeros((range_[1] - range_[0], size[1], size[0]), np.uint16)
            for i in range(0, range_[1] - range_[0]):
                image = image_func(idx, i + range_[0])
                if image is None:
                    continue
                image.decode()
                image = np.array(image, copy=False)
                np.copyto(images[i], image)
        return images


    def cache(self):
        self._cached_thumb = self.image_files['thumbnail'].asarray(maxworkers=12)
        shape = self._cached_thumb.shape
        self._cached_thumb = np.transpose(self._cached_thumb, (1, 0, 2, 3))
        self._cached_thumb = np.reshape(self._cached_thumb, (shape[0] * shape[1], shape[2], shape[3]))

    def delete_cache(self):
        self._cached_thumb = None

    def __getitem__(self, item):
        assert isinstance(item, int)
        return RawDataStack(item, self)


def read_ome_tiff(source: tifffile.TiffFile, ifd_list):
    #imgs = {}
    import asyncio
    async def read_page(ifd):
        img = sitk.GetImageFromArray(source.pages[ifd].asarray())
        return ifd, img
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for ifd in ifd_list:
        tasks.append(asyncio.ensure_future(read_page(ifd)))
    loop.run_until_complete(asyncio.wait(tasks))
    imgs = {t.result()[0]:t.result()[1] for t in tasks}
    imgs = [imgs[ifd] for ifd in ifd_list]
    return imgs


class RawDataStack:
    def __init__(self, index, parent:RawData):
        self.index = index
        self.parent = parent
        self.source_type = 'auto'

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.parent.load(self.index, [item.start, item.stop], source_type=self.source_type)


def load_raw_data(flsm_file):
    raw_data = RawData(flsm_file)
    return raw_data
