import SimpleITK as sitk
import tempfile, os
from VISoR_Reconstruction.misc import PARAMETER_DIR


def read_transformix_output_points(srcfile):
    points = []
    file = open(srcfile)
    while 1:
        l = file.readline()
        if len(l) < 10:
            break
        x = float(l.split(';')[4].split(' ')[4])
        y = float(l.split(';')[4].split(' ')[5])
        if not l.split(';')[4].split(' ')[6].split('.')[0].split('-')[-1].isdigit():
            points.append((x, y))
        else:
            z = float(l.split(';')[4].split(' ')[6])
            points.append((x, y, z))
    return points


def write_transformix_input_points(file, points, ndim):
    df = open(file, "w")
    df.write("point\n")
    df.write(str(len(points)) + "\n")
    for c in points:
        df.write(str(c[0]))
        for i in range(1, ndim):
            df.write(" {0}".format(str(c[i])))
        df.write("\n")
    df.close()


def read_transformix_input_points(file):
    f = open(file)
    f.readline()
    ct = int(f.readline())
    points = []
    for i in range(ct):
        line = f.readline()
        print(line)
        if len(line) < 3:
            break
        line = line.split(' ')
        points.append([float(line[0]), float(line[1]), float(line[2])])
    return points


def read_elastix_parameter(file):
    f = open(file)
    param = {}
    while 1:
        line = f.readline()
        if len(line) == 0:
            break
        p1, p2 = line.find('('), line.find(')')
        if p1 == -1:
            continue
        l = line[p1 + 1: p2].split(' ')
        k = l[0]
        v = []
        for i in range(1, len(l)):
            p = l[i]
            if len(p) == 0:
                continue
            if p[0] == '\"':
                v.append(p[1:-1])
            else:
                v.append(float(p))
        param[k] = v
    return param


def get_sitk_transform(elastix_param):
    tf_type = elastix_param['Transform'][0]
    image_spec = {'Size': [int(i) for i in elastix_param['Size']],
                  'Origin': [float(i) for i in elastix_param['Origin']],
                  'Spacing': [float(i) for i in elastix_param['Spacing']],
                  'Direction': [float(i) for i in elastix_param['Direction']]}
    ndim = int(elastix_param['FixedImageDimension'][0])

    if tf_type == 'EulerTransform':
        param = [float(i) for i in elastix_param['TransformParameters']]
        center = [float(i) for i in elastix_param['CenterOfRotationPoint']]
        if ndim == 2:
            tf = sitk.Euler2DTransform()
        elif ndim == 3:
            tf = sitk.Euler3DTransform()
        else:
            raise NotImplementedError('cannot resolve transformation {},{}'.format(tf_type, ndim))
        tf.SetCenter(center)
        tf.SetParameters(param)

    elif tf_type == 'AffineTransform':
        param = [float(i) for i in elastix_param['TransformParameters']]
        tf = sitk.AffineTransform(ndim)
        tf.SetParameters(param)

    elif tf_type == 'BSplineTransform':
        order = int(elastix_param['BSplineTransformSplineOrder'][0])
        param = [float(i) for i in elastix_param['TransformParameters']]
        tf = sitk.BSplineTransform(ndim, order)
        mesh_size =[int(i) - 3 for i in elastix_param['GridSize']]
        physical_dim = [float(elastix_param['GridSpacing'][i]) * mesh_size[i] for i in range(ndim)]
        origin = [float(elastix_param['GridOrigin'][i]) + float(elastix_param['GridSpacing'][i]) for i in range(ndim)]
        direction = [float(i) for i in elastix_param['GridDirection']]
        tf.SetTransformDomainMeshSize(mesh_size)
        tf.SetTransformDomainOrigin(origin)
        tf.SetTransformDomainPhysicalDimensions(physical_dim)
        tf.SetTransformDomainDirection(direction)
        tf.SetParameters(param)

    else:
        raise NotImplementedError('cannot resolve transformation {},{}'.format(tf_type, ndim))

    return tf, image_spec


def get_sitk_transform_from_file(file):
    param = read_elastix_parameter(file)
    prev_file = param['InitialTransformParametersFileName'][0]
    tf, image_spec = get_sitk_transform(param)
    t = sitk.Transform(tf.GetDimension(), sitk.sitkComposite)
    t.AddTransform(tf)
    if not prev_file == "NoInitialTransform":
        t.AddTransform(get_sitk_transform_from_file(prev_file)[0])
    return t, image_spec


def get_align_transform(fixed, moving, parameter_files, fixed_mask=None, moving_mask=None,
                        fixed_points=None, moving_points=None, rigidity_mask=None, inverse_transform=False,
                        initial_transform=None, multichannel=False):
    with tempfile.TemporaryDirectory() as ELASTIX_TEMP:
        elastix = sitk.ElastixImageFilter()
        elastix.SetOutputDirectory(ELASTIX_TEMP)
        params = sitk.VectorOfParameterMap()
        for p in parameter_files:
            param = sitk.ReadParameterFile(p)
            size = 1
            for s in moving.GetSize():
                size *= s
            if len(moving.GetSize()) == 2:
                param['NumberOfSpatialSamples'] = [str(int(max(moving.GetSize()[0] * moving.GetSize()[1] / 2048 * pow(4, i), 2048))) for i in range(4)]
            if rigidity_mask is not None:
                mask_path = os.path.join(ELASTIX_TEMP, 'rigidity_mask.mha')
                sitk.WriteImage(rigidity_mask, mask_path)
                param['MovingRigidityImageName'] = [mask_path]
            if multichannel:
                if param['Registration'][0] == 'MultiMetricMultiResolutionRegistration':
                    m = [*param['Metric']]
                    for i in range(1, fixed.GetSize()[2]):
                        m = [param['Metric'][0], *m]
                    param['Metric'] = m
                m = {'FixedImagePyramid': [], 'MovingImagePyramid': [], 'Interpolator': [], 'ImageSampler': []}
                for i in range(0, len(param['Metric'])):
                    for k in m:
                        m[k] = [*m[k], param[k][0]]
                for k in m:
                    param[k] = m[k]
            params.append(param)
            #elastix.WriteParameterFile(param, 'F:/chaoyu/test/f.txt')
        elastix.SetParameterMap(params)
        if multichannel:
            for c in range(fixed.GetSize()[2]):
                elastix.AddFixedImage(fixed[:, :, c])
                elastix.AddMovingImage(moving[:, :, c])
            for i in range(len(param['Metric']) - fixed.GetSize()[2]):
                elastix.AddFixedImage(fixed[:, :, 0])
                elastix.AddMovingImage(moving[:, :, 0])
        else:
            elastix.SetFixedImage(fixed)
            elastix.SetMovingImage(moving)
        if fixed_mask is not None:
            elastix.SetFixedMask(fixed_mask)
        if moving_mask is not None:
            elastix.SetMovingMask(moving_mask)
        if fixed_points is not None and moving_points is not None:
            elastix.SetFixedPointSetFileName(fixed_points)
            elastix.SetMovingPointSetFileName(moving_points)
        if initial_transform is not None:
            elastix.SetInitialTransformParameterFileName(initial_transform)
        s = elastix.Execute()
        tf_par = elastix.GetTransformParameterMap()
        #transformix = sitk.TransformixImageFilter()
        #transformix.SetOutputDirectory(ELASTIX_TEMP.name)
        #transformix.ComputeDeformationFieldOn()
        #transformix.SetTransformParameterMap(tf_par)
        #transformix.Execute()
        transform = None
        transforms = []
        for p in tf_par:
            tf, im = get_sitk_transform(p)
            if transform is None:
                transform = sitk.Transform(tf.GetDimension(), sitk.sitkComposite)
            transforms.append(tf)
        transforms.reverse()
        for t in transforms:
            transform.AddTransform(t)
        #df = sitk.ReadImage(os.path.join(ELASTIX_TEMP.name, 'deformationField.mhd'))
        #df = sitk.Compose(sitk.VectorIndexSelectionCast(df, 0),
        #                  sitk.VectorIndexSelectionCast(df, 1))
        #df = sitk.Cast(df, sitk.sitkVectorFloat64)
        if inverse_transform:
            ct = 0
            for p in tf_par:
                file = os.path.join(ELASTIX_TEMP, 'TransformParameters.{}.txt'.format(ct))
                if ct > 0:
                    p['InitialTransformParametersFileName'] = [os.path.join(ELASTIX_TEMP, 'TransformParameters.{}.txt'.format(ct - 1)).replace('\\', '/')]
                #p['Size'] = [str(k) for k in moving.GetSize()]
                elastix.WriteParameterFile(p, file)
                ct += 1
            out, inv = get_align_transform(moving, fixed, [os.path.join(PARAMETER_DIR, 'tp_inverse_bspline.txt')],
                                           initial_transform=file)
            return s, transform, inv
    return s, transform
