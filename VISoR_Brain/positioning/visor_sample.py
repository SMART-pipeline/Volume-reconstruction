import yaml, tempfile, tarfile, json
import numpy as np
from VISoR_Brain.format import raw_data
from contextlib import contextmanager
from distutils.version import LooseVersion
from VISoR_Brain.misc import *


class VISoRColumnImage:
    def __init__(self):
        self.index = 0
        self.data = None
        self.sphere = [(0, 0, 0), (0, 0, 0)]

    def get_image(self, roi: list, source: str = 'auto'):
        r0, r1 = np.minimum(roi[0], roi[1]), np.maximum(roi[0], roi[1])
        r0, r1 = np.int32(np.maximum(r0, self.sphere[0])), np.int32(np.minimum(r1, self.sphere[1]))
        self.data.source_type = source
        img = self.data[r0[2]:r1[2]]
        if isinstance(img, sitk.Image):
            spacing = img.GetSpacing()
            img.SetOrigin([img.GetOrigin()[0],
                           img.GetOrigin()[1],
                           img.GetOrigin()[2] + float(r0[2]) / spacing[2]])
            return img
        return img


    def in_sphere(self, pos):
        if np.min(np.greater(pos, self.sphere[0])) == 0:
            return False
        if np.min(np.less(pos, self.sphere[1])) == 0:
            return False
        return True

    def have_sphere_overlap(self, roi):
        r0, r1 = np.minimum(roi[0], roi[1]), np.maximum(roi[0], roi[1])
        if np.max(np.greater(r0, self.sphere[1])) == 0 and np.max(np.less(r1, self.sphere[0])) == 0:
            return True
        return False


def get_column_images(rawdata: raw_data.RawData):
    column = {}
    for i in range(len(rawdata.columns)):
        st = VISoRColumnImage()
        st.data = rawdata[i]
        st.index = i
        st.sphere = [[0, 0, 0], [int(rawdata.roi[2]), int(rawdata.roi[3]), len(rawdata.columns[i])]]
        column[i] = st
    return column



class VISoRSample:
    def __init__(self, filename=None):
        self.transforms = {}
        self.inverse_transforms = {}
        self.column_spheres = {}
        self.sphere = [[0, 0, 0], [0, 0, 0]]
        self.column_images = {}
        self.image = None
        self.column_source = None
        self.image_source = None
        self.device_file = None
        self.image_origin = [0, 0, 0]
        self.image_spacing = [1, 1, 1]
        self.version = VERSION

        # Private attributes for optimization
        self._column_spheres = None

        if filename is not None:
            self.load(filename)

    @contextmanager
    def load(self, filename):
        if filename.split('.')[-1] == 'tar':
            with tempfile.TemporaryDirectory() as tmpdir:
                self.version = '0.1'
                tar = tarfile.open(filename, 'r')
                tar.extractall(tmpdir)
                info_file = open(os.path.join(tmpdir, 'info.txt'), 'r')
                _info = yaml.load(info_file)
                info_file.close()
                self.sphere = _info['sphere']
                self.column_spheres = _info['stack_spheres']
                self._update_spheres()
                self.column_source = _info['stack_source']
                if _info['image_source'] is not None:
                    self.image_source = os.path.join(os.path.dirname(filename), _info['image_source'])
                    self.image_origin = _info['image_origin']
                    self.image_spacing = _info['image_spacing']
                self.device_file = _info['device_file']
                for i in range(len(_info['transforms'])):
                    tf_path = os.path.join(tmpdir, _info['transforms'][i])
                    tf = sitk.ReadTransform(tf_path)
                    self.transforms[i] = tf
                self.calculate_transforms()
                return
        with open(filename, 'r') as info_file:
            _info = json.load(info_file)
            self.sphere = _info['sphere']
            if 'version' in _info:
                self.version = _info['version']
            else:
                self.version = '0.2.0'
            self.column_spheres = {int(i): v for i, v in _info['column_spheres'].items()}
            self._update_spheres()
            self.column_source = _info['stack_source']
            if _info['image_source'] is not None:
                self.image_source = os.path.join(os.path.dirname(filename), _info['image_source'])
                self.image_origin = _info['image_origin']
                self.image_spacing = _info['image_spacing']
            if LooseVersion(self.version) < LooseVersion('0.4.0'):
                self.device_file = _info['device_file']
            for i in range(len(_info['transforms'])):
                tf = sitk.AffineTransform(3)
                tf.SetParameters(_info['transforms'][str(i)])
                self.transforms[i] = tf
            self.calculate_transforms()

    def load_columns(self):
        r = raw_data.RawData(self.column_source, self.device_file)
        self.column_images = get_column_images(r)

    def load_image(self):
        img = sitk.ReadImage(self.image_source)
        self.set_image(img)

    @property
    def raw_data(self) -> raw_data.RawData:
        if len(self.column_images) == 0:
            raise AttributeError('Raw data not loaded')
        return self.column_images[0].data.parent

    @contextmanager
    def save(self, filename):
        _info = {'sphere': self.sphere,
                 'transforms': {},
                 'image_source': None,
                 'stack_source': self.column_source}

        if self.image_source is not None:
            _info['image_source'] = os.path.relpath(self.image_source, os.path.dirname(filename))
            _info['image_origin'] = self.image_origin
            _info['image_spacing'] = self.image_spacing

        _info['version'] = VERSION
        # old way
        if filename.split('.')[-1] == 'tar':
            _info['stack_spheres'] = self.column_spheres
            with tempfile.TemporaryDirectory() as tmpdir:
                t_path = os.path.join(tmpdir, 'transforms')
                os.mkdir(t_path)
                for i in self.transforms.keys():
                    tf_path = os.path.join(t_path, str(i) + '.txt')
                    sitk.WriteTransform(self.transforms[i], tf_path)
                    _info['transforms'][i] = os.path.relpath(tf_path, tmpdir)

                info = yaml.dump(_info)
                info_file = open(os.path.join(tmpdir, 'info.txt'), 'w')
                info_file.write(info)
                info_file.close()
                tar = tarfile.open(filename, 'w')
                for root,dir_,files in os.walk(tmpdir):
                    for file in files:
                        fullpath = os.path.join(root, file)
                        tar.add(fullpath, os.path.relpath(fullpath, tmpdir))
                tar.close()
            return
        # new way
        _info['column_spheres'] = self.column_spheres
        for i in self.transforms:
            _info['transforms'][i] = self.transforms[i].GetParameters()
        with open(filename, 'w') as f:
            json.dump(_info, f, indent=2)



    def save_image(self, filename):
        sitk.WriteImage(self.image, filename)
        self.image_source = filename

    def in_stack_sphere(self, sample_pos, strict=True):
        st = len(self.transforms)
        spheres = self._column_spheres

        def _in_stack_sphere():
            res = []
            if strict:
                for i in range(st):
                    if spheres[i * 6] < sample_pos[0] < spheres[i * 6 + 3]:
                        if spheres[i * 6 + 1] < sample_pos[1] < spheres[i * 6 + 4]:
                            if spheres[i * 6 + 2] < sample_pos[2] < spheres[i * 6 + 5]:
                                res.append(i)
            else:
                if sample_pos[1] <= spheres[4]:
                    res.append(0)
                if sample_pos[1] > spheres[st * 6 - 5]:
                    res.append(st - 1)
                for i in range(1, st - 1):
                    if spheres[i * 6 + 1] <= sample_pos[1] < spheres[i * 6 + 4]:
                        res.append(i)
            return res
        return _in_stack_sphere()

    def get_sample_position_from_image(self, image_pos, image_spacing=None, image_origin=None):
        if image_spacing is None:
            image_spacing = self.image_spacing
        if image_origin is None:
            image_origin = self.image_origin
        return (np.array(image_pos)
                * np.array(image_spacing)
                + np.array(image_origin)).tolist()

    def get_image_position(self, sample_pos, image_spacing=None, image_origin=None):
        if image_spacing is None:
            image_spacing = self.image_spacing
        if image_origin is None:
            image_origin = self.image_origin
        return ((np.array(sample_pos) - np.array(image_origin)) / np.array(image_spacing)).tolist()

    # Give a position in raw data, return the position in physical space.
    def get_slice_position(self, index, stack_pos):
        return self.inverse_transforms[index].TransformPoint(stack_pos)

    # Give a position in physical space, return the position in raw data.
    def get_column_position(self, slice_pos, index=None, strict=False):
        if index is None:
            index = self.in_stack_sphere(slice_pos, strict)
            if len(index) == 0:
                return None, None
            index = index[0]
        p = self.transforms[index].TransformPoint(slice_pos)
        return index, p

    def create_reference(self):
        sl = VISoRSample()
        sl.transforms = self.transforms
        sl.sphere = self.sphere
        sl.column_spheres = self.column_spheres
        sl._update_spheres()
        return sl

    # Calculate the region of entire slice
    def calculate_spheres(self):
        self.column_spheres.clear()
        spheres = []
        for i in range(len(self.column_images)):
            sp = self.column_images[i].sphere
            tf = self.transforms[i].GetInverse()
            plist = [[sp[j][0], sp[k][1], sp[l][2]] for j in range(2) for k in range(2) for l in range(2)]
            plist = [tf.TransformPoint(p) for p in plist]
            self.column_spheres[i] = [np.min(plist, 0).tolist(), np.max(plist, 0).tolist()]
            spheres.append(self.column_spheres[i])
        sp = np.array(spheres)
        self.sphere = [np.min(sp, 0)[0].tolist(), np.max(sp, 0)[1].tolist()]
        self._update_spheres()

    def calculate_transforms(self):
        self.inverse_transforms = {i: self.transforms[i].GetInverse() for i in self.transforms}

    def _update_spheres(self):
        self._column_spheres = [self.column_spheres[i][j][k]
                                for i in range(len(self.column_spheres))
                                for j in range(2)
                                for k in range(3)]

    def set_image(self, image:sitk.Image):
        #self.sphere[0] = origin
        #self.sphere[1] = (np.int32(origin) + np.int32(image.GetSize())).tolist()
        self.image = image
        self.image_origin = list(self.image.GetOrigin())
        self.image_spacing = list(self.image.GetSpacing())

    def clear_sample_image(self):
        self.image = None

    def clear_stack_image(self):
        self.column_images.clear()

    def assign_stack_image(self, stack:VISoRColumnImage, index: int):
        if index > len(self.transforms) or index < 0:
            raise IndexError
        self.column_images[index] = stack

    def get_stack_shift(self, index: int):
        return self.raw_data.column_spacing[index] / self.raw_data.pixel_size * np.sin(self.raw_data.angle)

    def get_shifted_stack_image(self, index:int, roi):
        t1 = sitk.AffineTransform(3)
        t1.Shear(1, 2, self.get_stack_shift(index))
        plist = [[roi[j][0], roi[k][1], roi[l][2]]
                 for j in range(2) for k in range(2) for l in range(2)]
        plist = np.array([t1.GetInverse().TransformPoint(p) for p in plist])
        roi[1][2] += 1
        src = self.column_images[index].get_image(roi)
        src_origin = np.min(plist, 0).tolist()
        src_size = np.int32(np.max(plist, 0) - np.min(plist, 0)).tolist()
        src_spacing = np.array(src.GetSpacing())
        src = sitk.Resample(src, src_size, t1, sitk.sitkNearestNeighbor, src_origin, src_spacing.tolist())
        return src

    # Reconstruct the image of the given region in physical space.
    # Deprecated
    def get_sample_image(self, roi, pixel_size, stack_index=None, source='auto', block_size=None):
        min_idx = 0

        # Reconstruct each blocks.
        def _get_sample_image(roi, pixel_size, source, column_index=None):
            import torch
            roi = [roi[0].copy(), roi[1].copy()]
            roi_size = np.int32((np.array(roi[1]) - np.array(roi[0])) / pixel_size).tolist()
            ref = sitk.Image(roi_size, sitk.sitkFloat32)
            ref.SetOrigin(roi[0])
            ref.SetSpacing([pixel_size, pixel_size, pixel_size])
            out = torch.cuda.FloatTensor(roi_size[2], roi_size[1], roi_size[0]).zero_()
            out += 1
            idx_range = range(len(self.column_images))
            if column_index is not None:
                idx_range = [column_index]
            _min_idx = len(self.column_images)
            for i in idx_range:
                tf = self.transforms[i]
                t1 = sitk.AffineTransform(3)
                r = self.raw_data
                sr = r.column_spacing[i] / r.pixel_size * np.sin(r.angle)
                t1.Shear(1, 2, sr)
                t2 = sitk.AffineTransform(tf)
                t2.Shear(1, 2, -sr, False)
                plist = [roi[0], roi[1], [roi[0][0], roi[0][1], roi[1][2]], [roi[1][0], roi[1][1], roi[0][2]]]
                plist = np.array([tf.TransformPoint(p) for p in plist])
                roi_col = [np.min(plist - 1, 0).tolist(), np.max(plist + 2, 0).tolist()]
                if not self.column_images[i].have_sphere_overlap(roi_col):
                    continue
                if int(roi_col[1][2]) < self.column_images[i].sphere[0][2] + 1 \
                        or int(roi_col[0][2] > self.column_images[i].sphere[1][2]):
                    continue
                print('Generating image of stack {0}'.format(i))
                src = self.column_images[i].get_image(roi_col, source)
                if i < _min_idx:
                    _min_idx = i
                src_spacing = np.array(src.GetSpacing())
                p1 = np.array(src.GetOrigin())
                p2 = np.array(src.GetSize()) * src_spacing + p1
                plist = [p1, p2]
                plist = [[plist[j][0], plist[k][1], plist[l][2]]
                         for j in range(2) for k in range(2) for l in range(2)]
                plist = np.array([t1.GetInverse().TransformPoint(p) for p in plist])
                src_origin = np.min(plist, 0).tolist()
                src_size = np.int32((np.max(plist, 0) - np.min(plist, 0)) / src_spacing).tolist()
                resample_method = sitk.sitkNearestNeighbor
                if source == 'thumbnail':
                    resample_method = sitk.sitkLinear
                src = sitk.Resample(src, src_size, t1, resample_method, src_origin, src_spacing.tolist())
                out_ = sitk.Resample(src, ref, t2, sitk.sitkLinear)
                out_ = sitk.GetArrayFromImage(out_)
                out_ = torch.from_numpy(np.float32(out_))
                out_ = out_.cuda()
                # out = sitk.Cast(out, sitk.sitkFloat32)
                out = (out_ * out_ + out * out) / (out + out_)
            nonlocal min_idx
            if min_idx < _min_idx:
                min_idx = _min_idx
            out -= 1
            out = out.cpu().numpy()
            out = np.uint16(out)
            return out

        # Divide the given region into blocks.
        if block_size is None:
            block_size = int(min(512 * pixel_size, 2048))
        if isinstance(block_size, int):
            block_size = [block_size, block_size, block_size]
        roi = [roi[0].copy(), roi[1].copy()]
        roi_size = np.int32((np.array(roi[1]) - np.array(roi[0])) / pixel_size)
        out = np.zeros(np.flip(roi_size, 0), np.uint16)
        block_count = np.ceil((roi[1][0] - roi[0][0]) / block_size[0]) \
                      * np.ceil((roi[1][1] - roi[0][1]) / block_size[1]) \
                      * np.ceil((roi[1][2] - roi[0][2]) / block_size[2])
        ct = 0

        # Reconstruct image blockwise
        for j in np.arange(roi[0][1], roi[1][1], block_size[1]):
            for i in np.arange(roi[0][0], roi[1][0], block_size[0]):
                for k in np.arange(roi[0][2], roi[1][2], block_size[2]):
                    block_roi = [[i, j, k], [i + block_size[0], j + block_size[1], k + block_size[2]]]
                    block_roi[1] = np.minimum(block_roi[1], roi[1]).tolist()
                    block_image_roi = [np.int32(np.subtract(block_roi[0], roi[0]) / pixel_size),
                                       np.int32(np.subtract(block_roi[1], roi[0]) / pixel_size)]
                    if np.min(block_image_roi[1] - block_image_roi[0]) <= 0:
                        continue
                    print('Generating block {0}/{1}'.format(ct + 1, int(block_count)), *block_image_roi)
                    block = _get_sample_image(block_roi, pixel_size, source, stack_index)
                    np.copyto(out[block_image_roi[0][2]:block_image_roi[0][2] + block.shape[0],
                                  block_image_roi[0][1]:block_image_roi[0][1] + block.shape[1],
                                  block_image_roi[0][0]:block_image_roi[0][0] + block.shape[2]], block)
                    ct += 1
        out = sitk.GetImageFromArray(out)
        out.SetOrigin(roi[0])
        out.SetSpacing([pixel_size, pixel_size, pixel_size])
        return out


def load_visor_sample(filename: str):
    sl = VISoRSample()
    sl.load(filename)
    return sl

