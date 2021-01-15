import SimpleITK as sitk
from VISoR_Brain.utils.elastix_files import *
from VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.common import fill_outside
from VISoR_Reconstruction.misc import PARAMETER_DIR


def align_surfaces(prev_surface, next_surface, ref_img: sitk.Image = None, prev_points=None, next_points=None,
                   outside_brightness=2, nonrigid=True, ref_size=None, ref_scale=1, use_rigidity_mask=False, **kwargs):
    prev_df1, next_df1 = None, None
    if prev_surface is not None:
        prev_surface = prev_surface[:, :, 0]
    if next_surface is not None:
        next_surface = next_surface[:, :, 0]
    if ref_img is not None:
        ref_img = ref_img[:, :, 0]
    s1, s2 = prev_surface, next_surface
    if ref_img is None:
        ref_img = prev_surface
        if prev_surface is None:
            ref_img = next_surface
        ref_scale = 1
    if ref_size is not None:
        origin = [(ref_size[i] - ref_img.GetSize()[i] * ref_scale) / 2 for i in range(2)]
        ref_img.SetSpacing([ref_scale for i in range(2)])
        ref_img.SetOrigin(origin)
        ref_img = sitk.Resample(ref_img, ref_size, sitk.Transform(), sitk.sitkLinear, [0, 0], [1, 1])
    if prev_surface is not None:
        prev_surface = fill_outside(prev_surface, outside_brightness)
        prev_surface, prev_transform1 = get_align_transform(ref_img, prev_surface,
                                                           [os.path.join(PARAMETER_DIR, 'tp_align_surface_rigid.txt')])
        prev_df1 = sitk.TransformToDisplacementField(prev_transform1,
                                                    sitk.sitkVectorFloat64,
                                                    ref_img.GetSize(),
                                                    ref_img.GetOrigin(),
                                                    ref_img.GetSpacing(),
                                                    ref_img.GetDirection())
    else:
        next_surface = fill_outside(next_surface, outside_brightness)
        next_surface, next_transform1 = get_align_transform(ref_img, next_surface,
                                                           [os.path.join(PARAMETER_DIR,
                                                                         'tp_align_surface_rigid.txt')])
        next_df1 = sitk.TransformToDisplacementField(next_transform1,
                                                    sitk.sitkVectorFloat64,
                                                    ref_img.GetSize(),
                                                    ref_img.GetOrigin(),
                                                    ref_img.GetSpacing(),
                                                    ref_img.GetDirection())
    if prev_surface is None or next_surface is None:
        return prev_df1, next_df1
    if not nonrigid:
        return prev_df1, prev_df1

    rigidity_mask = None
    if use_rigidity_mask == True:
        rigidity_mask = sitk.BinaryThreshold(next_surface, outside_brightness + 1)
        rigidity_mask = sitk.BinaryMorphologicalOpening(rigidity_mask)
    prev_surface = fill_outside(prev_surface, outside_brightness)
    next_surface = fill_outside(next_surface, outside_brightness)

    if prev_points is not None and next_points is not None:
        def get_transform_points(points, transform, file, s_):
            tf = transform.GetInverse()
            points = read_transformix_input_points(points)
            points = [tf.TransformPoint(p[:2]) for p in points]
            write_transformix_input_points(file, points, 2)

        with tempfile.TemporaryDirectory() as ELASTIX_TEMP:
            get_transform_points(prev_points, prev_transform1, os.path.join(ELASTIX_TEMP, 'prev.txt'), s1)
            get_transform_points(next_points, sitk.Transform(2, sitk.sitkIdentity), os.path.join(ELASTIX_TEMP, 'next.txt'), s2)
            _, transform2 = get_align_transform(prev_surface, next_surface,
                                                [os.path.join(PARAMETER_DIR, 'tp_align_surface_rigid_manual.txt'),
                                                 os.path.join(PARAMETER_DIR, 'tp_align_surface_bspline_manual.txt')],
                                                fixed_points=os.path.join(ELASTIX_TEMP, 'prev.txt'),
                                                moving_points=os.path.join(ELASTIX_TEMP, 'next.txt'),
                                                rigidity_mask=rigidity_mask)
    else:
        _, transform2 = get_align_transform(prev_surface, next_surface,
                                            [os.path.join(PARAMETER_DIR,'tp_align_surface_rigid.txt'),
                                             os.path.join(PARAMETER_DIR,'tp_align_surface_bspline3.txt')],
                                            rigidity_mask=rigidity_mask)
    #sitk.WriteImage(prev_surface, 'F:/chaoyu/test/1_.mha')
    #sitk.WriteImage(_, 'F:/chaoyu/test/2_.mha')
    prev_df = prev_df1
    next_df = sitk.TransformToDisplacementField(transform2,
                                                sitk.sitkVectorFloat64,
                                                ref_img.GetSize(),
                                                ref_img.GetOrigin(),
                                                ref_img.GetSpacing(),
                                                ref_img.GetDirection())
    return prev_df, next_df
