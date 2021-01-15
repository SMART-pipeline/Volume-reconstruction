from VISoR_Brain.positioning.visor_sample import *
from VISoR_Brain.lib.gpu_resample import *
from VISoR_Brain.utils.column_data_set import ColumnDataset
from torch.utils.data import DataLoader
import torch

'''
def blend_average(a: sitk.Image, b: sitk.Image):
    b.SetOrigin(a.GetOrigin())
    out = sitk.Divide(a, 2) + sitk.Divide(b, 2)
    return out
'''
def blend_average(a, b):
    return np.multiply(a, 0.5) + np.multiply(b, 0.5)


def blend_right_side(a: sitk.Image, b: sitk.Image):
    return b


def blend_left_side(a: sitk.Image, b: sitk.Image):
    return a

'''
def blend_middle(a: sitk.Image, b: sitk.Image):
    n = int(a.GetSize()[1] / 2)
    return sitk.Tile([a[:, :n, :], b[:, n:, :]], [1, 2, 1])
'''
def blend_middle(a, b):
    n = int(a.shape[1] / 2)
    return np.concatenate((a[:, :n, :], b[:, n:, :]), 1)


def reconstruct(sample_data: VISoRSample, roi, pixel_size: float, column_index=None, source='raw', block_size=None,
                blend_method='middle'):
    blend_func = blend_right_side
    if blend_method == 'left_side':
        blend_func = blend_left_side
    elif blend_method == 'average':
        blend_func = blend_average
    elif blend_method == 'middle':
        blend_func = blend_middle

    # Reconstruct each blocks.
    def calc_raw_data_range(roi, column_index):
        roi = [roi[0].copy(), roi[1].copy()]

        tf = sample_data.transforms[column_index]

        plist = [roi[0], roi[1], [roi[0][0], roi[0][1], roi[1][2]], [roi[1][0], roi[1][1], roi[0][2]]]
        plist = np.array([tf.TransformPoint(p) for p in plist])
        roi_col = [column_index, np.min(plist - 1, 0).tolist(), np.max(plist + 2, 0).tolist()]
        if (not sample_data.column_images[column_index].have_sphere_overlap(roi_col)) \
                or int(roi_col[2][2]) < sample_data.column_images[column_index].sphere[0][2] + 1 \
                or int(roi_col[1][2] > sample_data.column_images[column_index].sphere[1][2]):
            return None
        return roi_col

    # Divide the given region into blocks.
    if block_size is None:
        block_size = [min(int((sample_data.sphere[1][2] - sample_data.sphere[0][2]) / 1.6 / pixel_size) * 8 * pixel_size, 512 * pixel_size),
                      1024 * pixel_size,
                      1024 * pixel_size]
    if isinstance(block_size, int):
        block_size = [block_size, block_size, block_size]
    block_size = [round(i / pixel_size) * pixel_size for i in block_size]

    block_generate_list = []
    tile_list = {}
    roi[1] = [np.ceil((roi[1][j] - roi[0][j]) / pixel_size) * pixel_size + roi[0][j] for j in range(3)]
    if column_index is None:
        column_range = [*range(len(sample_data.column_images))]
    else:
        column_range = [column_index]
    for i in column_range:
        col_y_range = [max(sample_data.column_spheres[i][0][1], roi[0][1]),
                       min(sample_data.column_spheres[i][1][1], roi[1][1])]
        if col_y_range[1] - col_y_range[0] <= 0:
            continue
        y_range = [np.ceil((col_y_range[0] - roi[0][1]) / pixel_size) * pixel_size + roi[0][1], None, None,
                   np.floor((col_y_range[1] - roi[0][1]) / pixel_size) * pixel_size + roi[0][1]]
        if i > column_range[0]:
            if y_range[0] < sample_data.column_spheres[i - 1][1][1] < y_range[3]:
                y_range[1] = np.floor((sample_data.column_spheres[i - 1][1][1] - roi[0][1]) / pixel_size) * pixel_size + roi[0][1]
        else:
            y_range[0] = roi[0][1]
        if i < column_range[-1]:
            if y_range[0] < sample_data.column_spheres[i + 1][0][1] < y_range[3]:
                y_range[2] = np.ceil((sample_data.column_spheres[i + 1][0][1] - roi[0][1]) / pixel_size) * pixel_size + roi[0][1]
        else:
            y_range[3] = roi[1][1]
        x_range = np.arange(roi[0][0], roi[1][0], block_size[0])
        if i % 2 != 0:
            np.arange(roi[1][0], roi[0][0], -block_size[0])
        for x in x_range:
            for z in np.arange(roi[0][2], roi[1][2], block_size[2]):
                block_roi = [[x, y_range[0], z], [x + block_size[0], y_range[3], z + block_size[2]]]
                block_roi[1] = np.minimum(block_roi[1], roi[1]).tolist()
                block_image_roi = [[int((block_roi[0][i] - roi[0][i]) / pixel_size) for i in range(3)],
                                   [int((block_roi[1][i] - roi[0][i]) / pixel_size) for i in range(3)]]
                if min([block_image_roi[1][j] - block_image_roi[0][j] for j in range(3)]) <= 0:
                    continue
                tiles = [block_roi, i, block_image_roi]
                y0 = y_range[0]
                for c in range(3):
                    y1 = y_range[c + 1]
                    if y1 is None:
                        continue
                    tile_roi = [[x, y0, z], [x + block_size[0], y1, z + block_size[2]]]
                    tile_roi[1] = np.minimum(tile_roi[1], block_roi[1]).tolist()
                    block_image_roi = [[int((tile_roi[0][i] - roi[0][i]) / pixel_size) for i in range(3)],
                                       [int((tile_roi[1][i] - roi[0][i]) / pixel_size) for i in range(3)]]
                    if min([block_image_roi[1][j] - block_image_roi[0][j] for j in range(3)]) <= 0:
                        continue
                    tiles.append(block_image_roi)
                    y0 = y1
                block_generate_list.append(tiles)
    image_roi = [int(round((roi[1][j] - roi[0][j]) / pixel_size)) for j in range(2, -1, -1)]

    ct = 0
    block_count = len(block_generate_list)
    if block_count < 1:
        return np.zeros(image_roi, np.uint16)
    prev_index = block_generate_list[0][1]
    raw_data_range_list = []
    for block_roi in block_generate_list:
        raw_data_range_list.append(calc_raw_data_range(block_roi[0], block_roi[1]))

    try:
        raw_data_loader = DataLoader(ColumnDataset(sample_data.raw_data.file, raw_data_range_list, source), num_workers=4)
    except AssertionError:
        return np.zeros(image_roi, np.uint16)

    def resample(roi, pixel_size, src, src_roi, column_index):
        roi = [roi[0].copy(), roi[1].copy()]
        roi_size = np.int32((np.array(roi[1]) - np.array(roi[0])) / pixel_size).tolist()

        if src_roi is None:
            return None

        tf = sample_data.transforms[column_index]
        tf = sitk.AffineTransform(tf)
        tf.Translate(roi[0], True)
        tf.Scale(pixel_size, True)
        src_scale = 1 / sample_data.raw_data.scales[source]
        tf.Scale([src_scale, src_scale, 1])
        tf.Translate([0, 0, min(np.ceil(-src_roi[1][2]), 0)])
        t = tf.GetParameters()
        t = np.float32([t[0], t[1], t[2], t[9],
                        t[3], t[4], t[5], t[10],
                        t[6], t[7], t[8], t[11]])
        dst_size = [int(np.ceil(roi_size[i] / 8) * 8) for i in range(3)]
        dst = np.zeros([dst_size[2], dst_size[1], dst_size[0]], np.uint16)
        dst = resample_affine(src, dst, t)
        #dst = dst[:roi_size[2], :roi_size[1], :roi_size[0]]
        #dst = sitk.GetImageFromArray(dst)
        #dst.SetOrigin(roi[0])
        #dst.SetSpacing([pixel_size, pixel_size, pixel_size])
        return dst

    # Generate blocks and copy to image
    out = np.zeros(image_roi, np.uint16)
    for i, raw_image in enumerate(raw_data_loader):
        block_roi = block_generate_list[i]
        print('Generating block {0}/{1}'.format(ct + 1, int(block_count)), *block_roi[2])
        if block_roi[1] != prev_index:
            prev_index = block_roi[1]
            sample_data.raw_data.release()
        block = resample(block_roi[0], pixel_size,
                         raw_image[0].numpy().view(np.uint16),
                         raw_data_range_list[i],
                         block_roi[1])
        ct += 1
        if block is None:
            continue
        for i in range(3, len(block_roi)):
            tile_roi = [[block_roi[i][0][j] - block_roi[2][0][j] for j in range(3)],
                        [block_roi[i][1][j] - block_roi[2][0][j] for j in range(3)]]
            tile = block[tile_roi[0][2]:tile_roi[1][2], tile_roi[0][1]:tile_roi[1][1], tile_roi[0][0]:tile_roi[1][0]]
            image_tile = out[block_roi[i][0][2]:block_roi[i][0][2] + tile.shape[0],
                         block_roi[i][0][1]:block_roi[i][0][1] + tile.shape[1],
                         block_roi[i][0][0]:block_roi[i][0][0] + tile.shape[2]]
            x, y, z = tuple(block_roi[i][0])
            if z not in tile_list:
                tile_list[z] = {}
            if y not in tile_list[z]:
                tile_list[z][y] = {}
            if x in tile_list[z][y]:
                tile = blend_func(tile_list[z][y][x], tile)
            np.copyto(image_tile, tile)
            tile_list[z][y][x] = image_tile

    '''        
    nz = len(tile_list)
    ny, nx = 0, 0
    for t in tile_list.values():
        ny = len(t)
        for tt in t.values():
            nx = len(tt)
    tile_list = [tile_list[z][y][x] for z in tile_list for y in tile_list[z] for x in tile_list[z][y]]
    out = sitk.Tile(tile_list, [nx, ny, nz])
    out.SetOrigin(roi[0])
    out.SetSpacing([pixel_size, pixel_size, pixel_size])
    '''
    return out
