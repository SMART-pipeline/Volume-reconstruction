from VISoR_Brain.positioning.visor_sample import *
from VISoR_Reconstruction.misc import PARAMETER_DIR

ELASTIX_TEMP = tempfile.TemporaryDirectory()

def reconstruct(sample_data: VISoRSample):
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(sitk.ReadParameterFile(
        os.path.join(PARAMETER_DIR, 'tp_align_columns_2.txt')))
    elastix.SetOutputDirectory(ELASTIX_TEMP.name)
    elastix.SetLogToConsole(True)

    r = sample_data.raw_data
    raw_scale = sample_data.raw_data.scales['raw']
    thumb_scale = sample_data.raw_data.scales['thumbnail']
    new_transforms = {k: sitk.AffineTransform(v) for k, v in sample_data.transforms.items()}
    p0 = [r.column_pos0[0][0] - r.pos0[0], 0, 0]
    new_transforms[0].Translate(p0, True)
    cum_tp = [0, 0, 0]
    for i in range(1, len(sample_data.column_images)):
        print('Aligning column {0} to column {1}'.format(i, i - 1))
        tp = []
        threshold = 300
        x1 = sample_data.get_column_position(sample_data.column_spheres[i][0], i - 1)[1][0]
        x2 = sample_data.get_column_position(sample_data.column_spheres[i - 1][1], i)[1][0]
        z1 = sample_data.get_column_position(sample_data.column_spheres[i - 1][0], i - 1)[1][2]
        z2 = sample_data.get_column_position(sample_data.column_spheres[i][0], i - 1)[1][2]
        offset = x1 - int(x1 / raw_scale) * raw_scale #- x2 + int(x2 / thumb_scale) * thumb_scale
        z_offset = int(round(z2 - z1))
        while threshold > 110 and len(tp) < 1:
            ct = 100 + min(z_offset, 0)
            while ct < (len(r.columns[i]) - 100 - min(-z_offset, 0)) and len(tp) < 1:
                image1 = r.load(i - 1, ct, source_type='thumbnail')[int(x1 / thumb_scale):, :, :]
                image2 = r.load(i, ct - z_offset, source_type='thumbnail')[:int(x2 / thumb_scale), :, :]
                frame1 = sitk.GetArrayFromImage(sitk.BinaryThreshold(image1[:,:,0], 0, threshold))
                frame2 = sitk.GetArrayFromImage(sitk.BinaryThreshold(image2[:,:,0], 0, threshold))
                if np.average(frame1) > 0.8 or np.average(frame2) > 0.8:
                    ct += 100
                    continue
                image1 = r.load(i - 1, (ct - 100, ct + 100), source_type='raw')[int(x1 / raw_scale):, :, :]
                image2 = r.load(i, (ct - 100 - z_offset, ct + 100), source_type='raw')[:int(x2 / raw_scale), :, :]
                #sitk.WriteImage(image1, 'D:/Users/chaoyu/test/1.mha')

                def pre_process(image: sitk.Image):
                    image = sitk.Clamp((sitk.Log(sitk.Cast(image, sitk.sitkFloat32)) - 4.6) * 39.4,
                                       sitk.sitkFloat32, 0, 255)
                    image.SetSpacing([raw_scale, raw_scale, 1])
                    image.SetOrigin([0, 0, 0])
                    return image

                image1 = pre_process(image1)
                image2 = pre_process(image2)
                ct += 200

                elastix.SetFixedImage(image1)
                elastix.SetMovingImage(image2)
                print('Calculating transform')

                try:
                    result = elastix.Execute()
                    #sitk.WriteImage(result, 'D:/Users/chaoyu/test/2.mha')
                except Exception as e:
                    print(e, e.__traceback__)
                    print('Failed')
                    continue
                else:
                    tp_ = elastix.GetTransformParameterMap()[0]['TransformParameters']
                    tp_ = [-float(i) for i in tp_]
                    tp.append(tp_)
            threshold = 100 + 0.5 * (threshold - 100)
        if len(tp) == 0:
            print('Failed')
            continue
        tp = np.median(tp, 0).tolist()

        p0 = [r.column_pos0[i][0] - r.pos0[0], 0, 0]
        tp = [-tp[0] - offset, -tp[1], -tp[2] - z_offset]
        cum_tp = np.add(cum_tp, tp).tolist()
        new_transforms[i].Translate(p0, True)
        new_transforms[i].Translate(cum_tp, False)
        r.release()
    sample_data.transforms = new_transforms
