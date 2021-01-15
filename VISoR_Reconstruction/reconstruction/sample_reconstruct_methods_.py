from positioning.visor_sample import *

# Deprecated

ELASTIX_TEMP = tempfile.TemporaryDirectory()

def simple_align(sample_data: VISoRSample):
    for i in range(len(sample_data.raw_data.columns)):
        affine_t = [0, 1 / sample_data.raw_data.pixel_size, 0,
                    0, 0, 1 / sample_data.raw_data.pixel_size / np.cos(sample_data.raw_data.angle),
                    1 / sample_data.raw_data.column_spacing[i], 0, 1 / sample_data.raw_data.column_spacing[i] / np.tan(sample_data.raw_data.angle)]

        p0 = sample_data.raw_data.column_pos0[i]
        af = sitk.AffineTransform(3)
        af.SetMatrix(affine_t)

        if sample_data.raw_data.column_spacing[i] < 0:
            p0 = (sample_data.raw_data.column_pos1[i][0], p0[1], p0[2])
        tl = np.subtract(sample_data.raw_data.pos0, p0).tolist()
        tl = af.TransformPoint(tl)
        af.Translate(tl)
        sample_data.transforms.append(af)



def projection_elastix_align(sample_data: VISoRSample):
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(sitk.ReadParameterFile('../parameters/tp_align_projection.txt'))
    elastix.SetOutputDirectory(ELASTIX_TEMP.name)
    #elastix.SetLogToConsole(False)

    for i in range(len(sample_data.raw_data.columns)):
        affine_t = [0, 1 / sample_data.raw_data.pixel_size, 0,
                    0, 0, 1 / sample_data.raw_data.pixel_size / np.cos(sample_data.raw_data.angle),
                    1 / sample_data.raw_data.column_spacing[i], 0, 1 / sample_data.raw_data.column_spacing[i] / np.tan(sample_data.raw_data.angle)]

        p0 = sample_data.raw_data.column_pos0[i]
        af = sitk.AffineTransform(3)
        af.SetMatrix(affine_t)

        if sample_data.raw_data.column_spacing[i] < 0:
            p0 = (sample_data.raw_data.column_pos1[i][0], p0[1], p0[2])
        tl = np.subtract(sample_data.raw_data.pos0, p0).tolist()
        tl = af.TransformPoint(tl)
        af.Translate(tl)
        sample_data.transforms.append(af)

        if i == 0:
            continue
        print('Aligning stack {0} to stack {1}'.format(i, i - 1))
        overlap_roi = [np.subtract(sample_data.raw_data.column_pos0[i], sample_data.raw_data.pos0),
                       np.subtract(sample_data.raw_data.column_pos1[i - 1], sample_data.raw_data.pos0)]
        overlap_roi[0][2] = -300
        prev_overlap = sitk.Threshold(sample_data.get_sample_image(overlap_roi, 1, i - 1), 100, 65535, 100) - 100
        overlap = sitk.Threshold(sample_data.get_sample_image(overlap_roi, 1, i), 100, 65535, 100) - 100
        prev_z_proj = sitk.MaximumProjection(prev_overlap, 2)[:,:,0]
        z_proj = sitk.MaximumProjection(overlap, 2)[:,:,0]
        sitk.WriteImage(z_proj, 'D:/chaoyu/Test/' + str(i) + 'z.tif')
        sitk.WriteImage(prev_z_proj, 'D:/chaoyu/Test/' + str(i) + 'z_.tif')
        prev_y_proj = sitk.MaximumProjection(prev_overlap, 1)[:,0,:]
        y_proj = sitk.MaximumProjection(overlap, 1)[:,0,:]
        sitk.WriteImage(y_proj, 'D:/chaoyu/Test/' + str(i) + 'y.tif')
        sitk.WriteImage(prev_y_proj, 'D:/chaoyu/Test/' + str(i) + 'y_.tif')
        elastix.SetFixedImage(prev_z_proj)
        elastix.SetMovingImage(z_proj)
        print('Calculating transform')
        try:
            elastix.Execute()
        except:
            print('Failed')
            continue
        tp = elastix.GetTransformParameterMap()[0]['TransformParameters']
        tp = [float(tp[0]), float(tp[1])]
        elastix.SetFixedImage(prev_y_proj)
        elastix.SetMovingImage(y_proj)
        print('Calculating transform')
        try:
            elastix.Execute()
        except:
            print('Failed')
            continue
        tp_z = elastix.GetTransformParameterMap()[0]['TransformParameters']
        tp.append(float(tp_z[1]))
        af.Translate(tp, True)
        sample_data.transforms[i] = af


