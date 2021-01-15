from VISoR_Brain.positioning.visor_sample import *


def blend_average(a: sitk.Image, b: sitk.Image):
    b.SetOrigin(a.GetOrigin())
    out = sitk.Divide(a, 2) + sitk.Divide(b, 2)
    return out


def blend_one_side(a: sitk.Image, b: sitk.Image):
    return a


def reconstruct(sample_data: VISoRSample, roi, pixel_size: float, column_index=None, source='auto', block_size=None, **kwargs):

    # Reconstruct each blocks.
    def inverse_resample(roi, pixel_size, source, column_index):
        roi = [roi[0].copy(), roi[1].copy()]
        roi_size = np.int32((np.array(roi[1]) - np.array(roi[0])) / pixel_size).tolist()
        r = sample_data.raw_data

        tf = sample_data.transforms[column_index]
        t1 = sitk.AffineTransform(3)
        sr = r.column_spacing[column_index] / r.pixel_size * np.sin(r.angle)
        t1.Shear(1, 2, sr)
        t2 = sitk.AffineTransform(tf)
        t2.Shear(1, 2, -sr, False)

        plist = [roi[0], roi[1], [roi[0][0], roi[0][1], roi[1][2]], [roi[1][0], roi[1][1], roi[0][2]]]
        plist = np.array([tf.TransformPoint(p) for p in plist])
        roi_col = [np.min(plist - 1, 0).tolist(), np.max(plist + 2, 0).tolist()]
        print('Generating image of column {0}'.format(column_index))
        if (not sample_data.column_images[column_index].have_sphere_overlap(roi_col)) \
                or int(roi_col[1][2]) < sample_data.column_images[column_index].sphere[0][2] + 1 \
                or int(roi_col[0][2] > sample_data.column_images[column_index].sphere[1][2]):
            out = sitk.Image(roi_size, sitk.sitkUInt16)
            out.SetOrigin(roi[0])
            out.SetSpacing([pixel_size, pixel_size, pixel_size])
            return out

        src = sample_data.column_images[column_index].get_image(roi_col, source)
        print(src.GetOrigin())
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
        out = sitk.Resample(src, roi_size, t2, sitk.sitkLinear, roi[0], [pixel_size, pixel_size, pixel_size])
        return out

    # Divide the given region into blocks.
    if block_size is None:
        block_size = [min((sample_data.sphere[1][2] - sample_data.sphere[0][2]) * 8, 512 * pixel_size),
                      512 * pixel_size,
                      512 * pixel_size]
    if isinstance(block_size, int):
        block_size = [block_size, block_size, block_size]
    block_size = [round(i / pixel_size) * pixel_size for i in block_size]

    block_generate_list = []
    tile_list = {}
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

    ct = 0
    block_count = len(block_generate_list)
    prev_index = block_generate_list[0][1]
    for block_roi in block_generate_list:
        print('Generating block {0}/{1}'.format(ct + 1, int(block_count)), *block_roi[2])
        if block_roi[1] != prev_index:
            prev_index = block_roi[1]
            sample_data.raw_data.release()
        block = inverse_resample(block_roi[0], pixel_size, source, block_roi[1])
        ct += 1
        for i in range(3, len(block_roi)):
            tile_roi = [[block_roi[i][0][j] - block_roi[2][0][j] for j in range(3)],
                        [block_roi[i][1][j] - block_roi[2][0][j] for j in range(3)]]
            tile = block[tile_roi[0][0]:tile_roi[1][0], tile_roi[0][1]:tile_roi[1][1], tile_roi[0][2]:tile_roi[1][2]]
            x, y, z = tuple(block_roi[i][0])
            if z not in tile_list:
                tile_list[z] = {}
            if y not in tile_list[z]:
                tile_list[z][y] = {}
            if x not in tile_list[z][y]:
                tile_list[z][y][x] = tile
            else:
                tile_list[z][y][x] = blend_one_side(tile_list[z][y][x], tile)
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
    return out
