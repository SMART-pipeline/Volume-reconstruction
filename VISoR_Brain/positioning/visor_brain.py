from VISoR_Brain.positioning.visor_sample import *
import warnings


class VISoRBrain:
    def __init__(self, path=None):
        self.slices = {}
        self._transforms = {}
        self.transform_source = {}
        self.atlas_transform = None
        self.slice_spheres = {}
        self.sphere_map = None
        self.sphere = [[0, 0, 0], [0, 0, 0]]
        self.version = VERSION

        if path is not None:
            self.load(path)

    def save(self, file_path):
        _info = {}
        _info['sphere'] = self.sphere
        _info['slices_spheres'] = self.slice_spheres
        _info['transforms'] = {}
        _info['slices'] = {}
        _info['version'] = VERSION
        path = os.path.dirname(file_path)
        if not os.path.exists(path):
            os.mkdir(path)

        if self.sphere_map is not None:
            _info['sphere_map'] = os.path.join(path, 'sphere_map.mha')
            sphere_map = sitk.GetImageFromArray(self.sphere_map)
            sitk.WriteImage(sphere_map, _info['sphere_map'])
        else:
            _info['sphere_map'] = None

        t_path = os.path.join(path, 'slices')
        if not os.path.exists(t_path):
            os.mkdir(t_path)
        for k, f in self.slices.items():
            p = os.path.join(t_path, str(k) + '.txt')
            f.save(p)
            _info['slices'][k] = (os.path.relpath(p, path))
        t_path = os.path.join(path, 'transforms')
        if not os.path.exists(t_path):
            os.mkdir(t_path)
        for k, f in self._transforms.items():
            p = os.path.join(t_path, '{0}.mha').format(k)
            if f is not None:
                sitk.WriteImage(sitk.Cast(f.GetDisplacementField(), sitk.sitkVectorFloat32), p)
            _info['transforms'][k] = (os.path.relpath(p, path))
        if self.atlas_transform is not None:
            sitk.WriteTransform(self.atlas_transform, os.path.join(path, 'atlas_transform.txt'))
            _info['atlas_transform'] = 'atlas_transform.txt'

        info = yaml.dump(_info)
        info_file = open(os.path.join(path, 'visor_brain.txt'), 'w')
        info_file.write(info)
        info_file.close()

    def load(self, file_path):
        path = os.path.dirname(file_path)
        with open(file_path, 'r') as info_file:
            _info = yaml.load(info_file)
        self.sphere = _info['sphere']
        self.slice_spheres = _info['slices_spheres']
        self._transforms = {}
        if 'version' in _info:
            self.version = _info['version']
        else:
            self.version = '0.2.0'
        for k, f in _info['slices'].items():
            sl = VISoRSample()
            sl.load(os.path.join(path, f))
            self.slices[k] = sl
        for k, f in _info['transforms'].items():
            self.transform_source[k] = os.path.join(path, f)
            self._transforms[k] = None
        if _info['sphere_map'] is not None:
            self.sphere_map = sitk.ReadImage(_info['sphere_map'])
            self.sphere_map = sitk.GetArrayFromImage(self.sphere_map)
        try:
            if _info['atlas_transform'] is not None:
                self.atlas_transform = sitk.ReadTransform(os.path.join(path, _info['atlas_transform']))
        except KeyError:
            pass

    def _load_transform(self, index):
        df = sitk.ReadImage(self.transform_source[index])
        df = sitk.Cast(df, sitk.sitkVectorFloat64)
        self._transforms[index] = sitk.DisplacementFieldTransform(df)

    def release_transform(self, index):
        if self._transforms[index] is not None:
            self._transforms[index] = None

    def transform(self, index: int):
        if self._transforms[index] is None:
            self._load_transform(index)
        return self._transforms[index]

    def transforms(self):
        for i in self._transforms:
            yield i, self.transform(i)

    def set_transform(self, index: int, transform: sitk.Transform):
        self._transforms[index] = transform

    def get_slice_position(self, brain_pos, index=None, strict=False):
        if index is None:
            if strict is True:
                index = next(iter(self.slices.keys()))
                for k, f in self.slice_spheres.items():
                    if f[0][0] <= brain_pos[0] < f[1][0] \
                            and f[0][1] <= brain_pos[1] < f[1][1] \
                            and f[0][2] <= brain_pos[2] < f[1][2]:
                        index = k
                        break
            else:
                index = next(iter(self.slices.keys()))
                for k, f in self.slice_spheres.items():
                    if f[0][2] <= brain_pos[2]:
                        index = k
                        if brain_pos[2] < f[1][2]:
                            break
        pos = self.transform(index).TransformPoint(brain_pos)
        return index, pos

    def get_column_position(self, brain_pos, slice_index=None, column_index=None, strict=False):
        slice_index, pos = self.get_slice_position(brain_pos, slice_index)
        column_index, pos = self.slices[slice_index].get_column_position(pos, column_index, strict)
        if column_index is None:
            return None, None, None
        return slice_index, column_index, pos

    def get_brain_position_from_slice(self, slice_index, slice_pos, accuracy=0.001, max_iteration=100):
        pos = [slice_pos[0], slice_pos[1], self.slice_spheres[slice_index][0][2]]
        for i in range(max_iteration):
            sp1 = self.get_slice_position(pos, slice_index)[1]
            sp2 = self.get_slice_position([pos[j] + 0.01 for j in range(3)], slice_index)[1]
            d = [sp2[j] - sp1[j] for j in range(3)]
            D = [sp1[j] - slice_pos[j] for j in range(3)]
            pos = [pos[j] - D[j] * d[j] / 0.013 for j in range(3)]
            if max(D) < accuracy and min(D) > -accuracy:
                break
        if i == max_iteration - 1:
            print(slice_index, D, slice_pos)
            warnings.warn('Not converge in max iterations. Max error {}'.format(max(max(D), -min(D))), RuntimeWarning)
        return pos

    def get_brain_position_from_column(self, slice_index, column_index, pos, accuracy=0.001, max_iteration=100):
        slice_pos = self.slices[slice_index].get_slice_position(column_index, pos)
        pos = self.get_brain_position_from_slice(slice_index, slice_pos, accuracy, max_iteration)
        return pos

    def calculate_sphere(self):
        sphere_list = [s for s in self.slice_spheres.values()]
        self.sphere = [np.min(sphere_list, 0)[0].tolist(), np.max(sphere_list, 0)[1].tolist()]

    def create_sphere_map(self, voxel_size=50):
        size = [3 * voxel_size,
                self.sphere[1][2] - self.sphere[0][2] + voxel_size,
                self.sphere[1][1] - self.sphere[0][1] + voxel_size,
                self.sphere[1][0] - self.sphere[0][0] + voxel_size]
        print(size)
        print(self.sphere)
        size = np.int32(np.array(size) / voxel_size)
        sphere_map = np.zeros(size, np.int32)
        s0i, s0j, s0k = self.sphere[0][2], self.sphere[0][1], self.sphere[0][0]
        for i in range(int(self.sphere[0][2]), int(self.sphere[1][2]), voxel_size):
            print(i)
            for j in range(int(self.sphere[0][1]), int(self.sphere[1][1]), voxel_size):
                for k in range(int(self.sphere[0][0]), int(self.sphere[1][0]), voxel_size):
                    i_, j_, k_ = int((i - s0i) / voxel_size), \
                                 int((j - s0j) / voxel_size), \
                                 int((k - s0k) / voxel_size)
                    slice_index, stack_index, pos = self.get_column_position([k, j, i], strict=True)
                    if slice_index is None:
                        continue
                    sphere_map[0, i_, j_, k_] = slice_index
                    sphere_map[1, i_, j_, k_] = stack_index
                    sphere_map[2, i_, j_, k_] = int(pos[2])
        self.sphere_map = sphere_map


def load_visor_brain(path):
    return VISoRBrain(path)

