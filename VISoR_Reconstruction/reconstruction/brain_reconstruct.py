import cv2
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from VISoR_Brain.positioning.visor_brain import *
from VISoR_Brain.format.raw_data import RawData
from .sample_reconstruct import reconstruct_sample, reconstruct_image
import gc, tifffile
from VISoR_Reconstruction.misc import ROOT_DIR


def _get_all_methods():
    global _all_methods
    methods_root = os.path.join(ROOT_DIR, 'reconstruction', 'brain_reconstruct_methods')
    dirs = [d for d in os.listdir(methods_root)
            if os.path.isdir(os.path.join(methods_root, d))]
    _all_methods = {}
    for d in dirs:
        methods = {}
        files = os.listdir(os.path.join(methods_root, d))
        for f in files:
            if f.split('.')[-1] == 'py':
                try:
                    method = __import__('VISoR_Reconstruction.reconstruction.brain_reconstruct_methods.{}.{}'.format(d, f[:-3]),
                                        globals(), locals(), [d]).__getattribute__(d)
                except AttributeError:
                    continue
                methods[f[:-3]] = method
        if len(methods) > 0:
            _all_methods[d] = methods
    return _all_methods


_all_methods = _get_all_methods()


def fill_blank(img):
    state = sitk.Image([img.GetSize()[0], img.GetSize()[1]], sitk.sitkUInt16) * 0
    out_ = []
    for i in range(img.GetSize()[2]):
        sl = sitk.Extract(img, [img.GetSize()[0], img.GetSize()[1], 0], [0, 0, i])
        sl.SetOrigin([0., 0.])
        sl.SetSpacing([1, 1])
        zero_area = sitk.LessEqual(sl, 90)
        zero_area = sitk.Cast(zero_area, sitk.sitkUInt16)
        state = state * zero_area + sl * sitk.Not(zero_area)
        out_.append(state)
    out = []
    state = sitk.Image([img.GetSize()[0], img.GetSize()[1]], sitk.sitkUInt16) * 0
    for i in range(img.GetSize()[2] - 1, -1, -1):
        sl = out_[i]
        sl.SetOrigin([0., 0.])
        sl.SetSpacing([1, 1])
        zero_area = sitk.LessEqual(sl, 90)
        zero_area = sitk.Cast(zero_area, sitk.sitkUInt16)
        state = state * zero_area + sl * sitk.Not(zero_area)
        out.append(state)
    out.reverse()
    return sitk.JoinSeries(out)


def calc_surface_height_map(img: sitk.Image, slice_thickness=300, internal_pixel_size=4, internal_downsample=(2, 2, 1)):
    k_surface_band = cv2.getGaussianKernel(3, -1)
    k_surface_band = np.matmul(k_surface_band, np.transpose(k_surface_band))
    k_surface_band[1, 1] = k_surface_band[1, 1] - np.sum(k_surface_band)
    k_surface_band = np.array([[k_surface_band]])
    k_surface_band = torch.FloatTensor(k_surface_band)
    k_surface_band = Variable(k_surface_band)
    k_surface_band_c = k_surface_band.cuda()

    k_grad = np.float32([[[0.5, 1, 2, 4, 8, 0, -8, -4, -2, -1, 0.5]]])
    k_grad = np.transpose(k_grad, [2, 1, 0])
    k_grad = sitk.GetImageFromArray(k_grad)

    k_grad2 = np.float32([[[1, 0, -1]]])
    k_grad2 = np.transpose(k_grad2, [2, 1, 0])
    k_grad2 = sitk.GetImageFromArray(k_grad2)

    thickness = int(slice_thickness / internal_pixel_size / internal_downsample[2])
    surface_pixel_size = float(internal_pixel_size / np.sqrt(internal_downsample[0] * internal_downsample[1]))
    img.SetSpacing([1, 1, 1])
    img.SetOrigin([0, 0, 0])
    size_ = img.GetSize()
    tf = sitk.AffineTransform(3)
    i_ = list(internal_downsample)
    tf.Scale(i_)
    proc_size = [int(size_[i] / i_[i]) for i in range(3)]
    img = sitk.Resample(img, proc_size, tf)
    img = fill_blank(img)
    img = sitk.Cast(img, sitk.sitkFloat32)
    img = (sitk.Log(img) - 4.6) * 39.4
    img = sitk.Clamp(img, sitk.sitkFloat32, 0, 255)

    def get_edge_grad(img_: sitk.Image, ul):
        grad_m = sitk.Convolution(img_, k_grad)
        if ul == 1:
            grad_m = sitk.Clamp(grad_m, sitk.sitkFloat32, 0, 65535)
        else:
            grad_m = sitk.Clamp(grad_m, sitk.sitkFloat32, -65535, 0)
        grad_m = ul * sitk.Convolution(grad_m, k_grad2)
        grad_m = sitk.GetArrayFromImage(grad_m)
        grad_m = torch.FloatTensor(grad_m)
        grad_m = grad_m.cuda()
        return grad_m

    u_grad_m = get_edge_grad(img, 1)
    l_grad_m = get_edge_grad(img, -1)

    shape = u_grad_m.shape
    u = (torch.rand(1, 1, shape[1], shape[2]) * (shape[0]) / 2 * 0)
    l = (shape[0] + 0 * torch.rand(1, 1, shape[1], shape[2]) * (shape[0]) / 2)

    lr = 0.001
    momentum = 0.9
    lr_decay = 0.0001
    u_grad = torch.zeros(1, 1, shape[1], shape[2])
    l_grad = torch.zeros(1, 1, shape[1], shape[2])
    k = k_surface_band
    u = u.cuda()
    l = l.cuda()
    u_grad = u_grad.cuda()
    l_grad = l_grad.cuda()
    k = k_surface_band_c
    grid = torch.meshgrid(torch.Tensor([0.0]).float().cuda(),
                          torch.linspace(-1, 1, shape[1]).float().cuda(),
                          torch.linspace(-1, 1, shape[2]).float().cuda())
    grid = torch.stack([grid[2 - i] for i in range(3)], 3)[None,]

    u_grad_m = u_grad_m[None, None,]
    l_grad_m = l_grad_m[None, None,]
    gu, gl = None, None
    for i in range(8000):
        def calc_grad(s, grad, grad_m, ul):
            g = torch.clamp(s[0], 0, shape[0] - 1)
            u_plane_bend = F.pad(s, (1, 1, 1, 1), mode='reflect')
            u_plane_bend = F.conv2d(u_plane_bend, k, padding=0)[0].data
            grid[:, :, :, :, 2].copy_(g * (2 / shape[0]) - 1)
            u_edge = F.grid_sample(grad_m, grid)[0]
            grad = u_edge + \
                   (1200 * surface_pixel_size) * u_plane_bend + \
                   0.004 * thickness * ul * torch.clamp(l - u - thickness, -1000, 10) + \
                   momentum * grad
            return grad, g

        u_grad, gu = calc_grad(u, u_grad, u_grad_m, 1)
        l_grad, gl = calc_grad(l, l_grad, l_grad_m, -1)
        u += lr * u_grad
        l += lr * l_grad
        u = torch.clamp(u, 0, shape[0] - 1)
        l = torch.clamp(l, 0, shape[0] - 1)
        lr *= (1 - lr_decay)

    umap = np.float32((u * internal_downsample[2] + 0.5).cpu().numpy()[0][0])
    lmap = np.float32((l * internal_downsample[2] - 1.5).cpu().numpy()[0][0])
    umap = cv2.resize(umap, (size_[0], size_[1]), interpolation=cv2.INTER_CUBIC)
    lmap = cv2.resize(lmap, (size_[0], size_[1]), interpolation=cv2.INTER_CUBIC)
    umap = sitk.GetImageFromArray(umap)
    lmap = sitk.GetImageFromArray(lmap)

    return umap, lmap


def extract_surface(img: sitk.Image, umap: sitk.Image, lmap: sitk.Image):
    img.SetSpacing([1, 1, 1])
    img.SetOrigin([0, 0, 0])
    umap_s = umap + 1
    lmap_s = lmap - 1
    zeros = sitk.Image(umap.GetSize(), umap.GetPixelIDValue())
    df = sitk.JoinSeries(sitk.Compose(zeros, zeros, umap_s), sitk.Compose(zeros, zeros, lmap_s))
    df = sitk.Cast(df, sitk.sitkVectorFloat64)
    ref = sitk.Image(df)
    tr = sitk.DisplacementFieldTransform(3)
    tr.SetDisplacementField(df)
    surfaces = sitk.Resample(img, ref, tr)
    surfaces = sitk.Cast(surfaces, sitk.sitkFloat32)
    surfaces = sitk.Clamp((sitk.Log(sitk.Cast(surfaces, sitk.sitkFloat32)) - 4.6) * 39.4, sitk.sitkUInt8, 0, 255)
    #u_surface = torch.gather(src_img, 0, torch.Tensor(np.array([umap_s])).long())[0].cpu().numpy()
    #l_surface = torch.gather(src_img, 0, torch.Tensor(np.array([lmap_s])).long())[0].cpu().numpy()
    #u_surface = np.clip((np.log(u_surface) - 4.6) * 39.4, 0, 255)
    #l_surface = np.clip((np.log(l_surface) - 4.6) * 39.4, 0, 255)

    #u_surface = sitk.GetImageFromArray(np.uint8(u_surface))
    #l_surface = sitk.GetImageFromArray(np.uint8(l_surface))

    return surfaces[:, :, 0], surfaces[:, :, 1]


def generate_projection(image: sitk.Image):
    return sitk.MaximumProjection(image, 2)


def calc_surface_height_map_(img: sitk.Image, slice_thickness=300, internal_pixel_size=4, internal_downsample=(2, 2, 1)):
    umap, lmap = calc_surface_height_map(img, slice_thickness, internal_pixel_size, internal_downsample)
    u_surface, l_surface = extract_surface(img ,umap, lmap)
    return umap, lmap, u_surface, l_surface


def align_surfaces(method='elastix', **kwargs):
    if 'prev_surface' in kwargs:
        prev_surface = [kwargs['prev_surface']]
        next_surface = [kwargs['next_surface']]
        if kwargs['ref_img'] is not None:
            ref_img = [kwargs['ref_img']]
    else:
        prev_surface, next_surface, ref_img = [], [], []
        for i in range(1000):
            b = True
            if 'prev_surface_{}'.format(i) in kwargs:
                prev_surface.append(kwargs['prev_surface_{}'.format(i)])
                b = False
            if 'next_surface_{}'.format(i) in kwargs:
                next_surface.append(kwargs['next_surface_{}'.format(i)])
                b = False
            if 'ref_img_{}'.format(i) in kwargs:
                ref_img.append(kwargs['ref_img_{}'.format(i)])
                b = False
            if b:
                break
    def proc(img):
        print(img)
        if len(img) > 0 and img[0] is not None:
            return sitk.JoinSeries(img)
        else:
            return None
    kwargs['prev_surface'] = proc(prev_surface)
    kwargs['next_surface'] = proc(next_surface)
    if 'ref_img_0' in kwargs or kwargs['ref_img'] is not None:
        ref_img = proc(ref_img)
        kwargs['ref_img'] = ref_img
    return _all_methods['align_surfaces'][method](**kwargs)


def combine_transforms(tf1, tf2):
    size = tf2.GetSize()
    tf1 = sitk.Cast(tf1, sitk.sitkVectorFloat64)
    tf2 = sitk.Cast(tf2, sitk.sitkVectorFloat64)
    df1 = sitk.DisplacementFieldTransform(sitk.Image(tf1))
    df2 = sitk.DisplacementFieldTransform(sitk.Image(tf2))
    tr = sitk.Transform(df1.GetDimension(), sitk.sitkComposite)
    tr.AddTransform(df1)
    tr.AddTransform(df2)
    df = sitk.TransformToDisplacementField(tr, sitk.sitkVectorFloat32, size)
    return df


def resample(image:torch.Tensor, displacement, image_is_displacement=False):
    shape = displacement[0].shape

    grid = torch.meshgrid(torch.linspace(-1, 2 * shape[0] / image.shape[0] - 1, shape[0]).float().cuda(),
                          torch.linspace(-1, 2 * shape[1] / image.shape[1] - 1, shape[1]).float().cuda())

    df = torch.stack([torch.add(grid[1 - i], 2 / image.shape[1 - i], displacement[i]) for i in range(2)], 2)[None,]

    image = image[None, None,]

    if not image_is_displacement:
        out = F.grid_sample(image, df, padding_mode='zeros')
    else:
        out = F.grid_sample(image, df, padding_mode='border')

        dx = torch.mean((image[:, :, :, -1] - image[:, :, :, 0]), 2) / 2
        dy = torch.mean((image[:, :, -1, :] - image[:, :, 0, :]), 2) / 2
        pad_offset = F.softshrink(df, 1) * torch.stack((dx, dy), 2)[:, :, None, None, :]
        pad_offset = torch.sum(pad_offset, 4)
        out += pad_offset
    return out[0][0]


def downsample(img, scale, cuda=True):
    img = torch.unsqueeze(torch.unsqueeze(img, 0), 0)
    img = F.interpolate(img, None, 1 / scale, 'bilinear', align_corners=True)[0, 0, 0:, 0:]
    return img


def upsample(img, scale, size):
    img = torch.unsqueeze(torch.unsqueeze(img, 0), 0)
    img = F.interpolate(img, size, None, 'bilinear', align_corners=True)[0, 0, 0:, 0:]
    return img


def h_p_rigidity_grad(u, du):
    ux, uy = u[0], u[1]
    lhx = ux[:, 1:] - ux[:, :-1]
    lhy = uy[:, 1:] - uy[:, :-1]
    lvx = ux[1:, :] - ux[:-1, :]
    lvy = uy[1:, :] - uy[:-1, :]
    lhx = F.pad(lhx, (1, 1), value=1) + 1
    lhy = F.pad(lhy, (1, 1), value=0)
    lvx = F.pad(lvx, (0, 0, 1, 1), value=0)
    lvy = F.pad(lvy, (0, 0, 1, 1), value=1) + 1
    lhx1, lhx2 = lhx[:, :-1], lhx[:, 1:]
    lhy1, lhy2 = lhy[:, :-1], lhy[:, 1:]
    lvx1, lvx2 = lvx[:-1, :], lvx[1:, :]
    lvy1, lvy2 = lvy[:-1, :], lvy[1:, :]
    lh = torch.sqrt(lhx * lhx + lhy * lhy)
    lv = torch.sqrt(lvx * lvx + lvy * lvy)
    lh1, lh2 = lh[:, :-1], lh[:, 1:]
    lv1, lv2 = lv[:-1, :], lv[1:, :]
    kh = torch.ones_like(ux[:, 1:])
    kh = F.pad(kh, (1, 1), value=0)
    kv = torch.ones_like(ux[1:, :])
    kv = F.pad(kv, (0, 0, 1, 1), value=0)
    kh1, kh2 = kh[:, :-1], kh[:, 1:]
    kv1, kv2 = kv[:-1, :], kv[1:, :]

    grad_x = - lhx1 * (1 - 1 / lh1) * kh1 \
             + lhx2 * (1 - 1 / lh2) * kh2 \
             - lvx1 * (1 - 1 / lv1) * kv1 \
             + lvx2 * (1 - 1 / lv2) * kv2
    grad_y = - lhy1 * (1 - 1 / lh1) * kh1 \
             + lhy2 * (1 - 1 / lh2) * kh2 \
             - lvy1 * (1 - 1 / lv1) * kv1 \
             + lvy2 * (1 - 1 / lv2) * kv2
    return grad_x, grad_y


def v_p_rigidity_grad(u0, u1):
    grad_x = u0[0] - u1[0]
    grad_y = u0[1] - u1[1]
    return grad_x, grad_y


def optimize_transforms(u, l, i_pyr):
    print('Optimizing all transformations...')

    def U(i, dim):
        return torch.Tensor(sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(i, dim))).float()
    uu_ = {}
    for i in u:
        uu_[i] = [U(u[i], 0), U(u[i], 1)]
        m = u[i]
        u[i] = None
        del m
    gc.collect()
    lu_ = {}
    for i in l:
        lu_[i] = [U(l[i], 0), U(l[i], 1)]
        m = l[i]
        l[i] = None
        del m

    if i_pyr is None:
        uu0 = uu_
        lu0 = lu_
    else:
        uu0 = {i: [downsample(uu_[i][0], i_pyr, False), downsample(uu_[i][1], i_pyr, False)] for i in u}
        lu0 = {i: [downsample(lu_[i][0], i_pyr, False), downsample(lu_[i][1], i_pyr, False)] for i in u}

    for i in uu0:
        for j in range(2):
            #uu0[i][j] = uu0[i][j].pin_memory()
            #lu0[i][j] = lu0[i][j].pin_memory()
            uu0[i][j] = uu0[i][j].cuda()
            lu0[i][j] = lu0[i][j].cuda()

    def Du(i):
        t = torch.zeros(i.size())
        #t = t.pin_memory()
        t = t.cuda()
        return t
    uDu = {i: [Du(uu0[i][0]), Du(uu0[i][1])] for i in uu0}
    lDu = {}
    for i in uDu:
        if i + 1 in uDu:
            lDu[i] = uDu[i + 1]
        else:
            lDu[i] = [Du(uu0[i][0]), Du(uu0[i][1])]

    ct = 0
    ua = {}
    la = {}
    #i_pyr_ = i_pyr
    #if i_pyr is None:
    #    i_pyr_ = 1
    for pyr in [1]:
        for i in range(100 * len(uu0) + 200):
            ua = {}
            la = {}
            metric_sum = 0
            for j in uu0:
                #uDu_ = [uDu[j][k].cuda(non_blocking=True) for k in range(2)]
                #lDu_ = [lDu[j][k].cuda(non_blocking=True) for k in range(2)]
                #uu = [resample(uu0[j][k].cuda(non_blocking=True), uDu_) + uDu_[k] for k in range(2)]
                #lu = [resample(lu0[j][k].cuda(non_blocking=True), lDu_) + lDu_[k] for k in range(2)]
                uu = [resample(uu0[j][k], uDu[j], True) + uDu[j][k] * i_pyr for k in range(2)]
                lu = [resample(lu0[j][k], lDu[j], True) + lDu[j][k] * i_pyr for k in range(2)]

                if pyr == 1:
                    uup = uu
                    lup = lu
                else:
                    uup = [downsample(uu[k], pyr) for k in range(2)]
                    lup = [downsample(lu[k], pyr) for k in range(2)]

                uhp = h_p_rigidity_grad(uup, uDu[j])
                lhp = h_p_rigidity_grad(lup, lDu[j])
                vp = v_p_rigidity_grad(uup, lup)
                ua[j] = [0.001 * uhp[k] - 0.002 * vp[k] for k in range(2)]
                la[j] = [0.001 * lhp[k] + 0.002 * vp[k] for k in range(2)]

                def limit_step(dx, dy, limit=2):
                    s = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
                    s = torch.clamp(s, limit) / limit
                    dx /= s
                    dy /= s
                    return dx, dy
                ua[j] = limit_step(*ua[j])
                la[j] = limit_step(*la[j])

                metric_sum += torch.mean(torch.abs(ua[j][0])) / len(uu0)
                metric_sum += torch.mean(torch.abs(ua[j][1])) / len(uu0)
                metric_sum += torch.mean(torch.abs(la[j][0])) / len(uu0)
                metric_sum += torch.mean(torch.abs(la[j][1])) / len(uu0)

            for j in uu0:

                if pyr != 1:
                    ua_ = [upsample(ua[j][k], pyr, uu0[j][k].size()) for k in range(2)]
                    la_ = [upsample(la[j][k], pyr, lu0[j][k].size()) for k in range(2)]
                else:
                    ua_ = ua[j]
                    la_ = la[j]

                for k in range(2):
                    #uDu[j][k].copy_(uDu[j][k] + ua_[k].cpu())
                    #lDu[j][k].copy_(lDu[j][k] + la_[k].cpu())
                    uDu[j][k] += ua_[k]
                    lDu[j][k] += la_[k]
            ct += 1
            print(ct, metric_sum)

    uout, lout = {}, {}
    del uu0
    del lu0
    for j in uDu:
        uDu_ = [uDu[j][k].cuda() * i_pyr for k in range(2)]
        lDu_ = [lDu[j][k].cuda() * i_pyr for k in range(2)]
        if i_pyr is not None:
            uDu_ = [upsample(uDu_[k], i_pyr, uu_[j][k].size()) for k in range(2)]
            lDu_ = [upsample(lDu_[k], i_pyr, uu_[j][k].size()) for k in range(2)]
        uu = [resample(uu_[j][k].cuda(), uDu_, True) + uDu_[k] for k in range(2)]
        lu = [resample(lu_[j][k].cuda(), lDu_, True) + lDu_[k] for k in range(2)]
        uu_[j] = None
        lu_[j] = None
        uu = sitk.Compose([sitk.GetImageFromArray(uu[k].cpu().numpy()) for k in range(2)])
        lu = sitk.Compose([sitk.GetImageFromArray(lu[k].cpu().numpy()) for k in range(2)])
        uDu[j] = uu
        lDu[j] = lu
    return uDu, lDu


def combine_transforms_xy_z(tf_xy, tf_z):
    tf_xy = sitk.Compose(sitk.VectorIndexSelectionCast(tf_xy, 0),
                         sitk.VectorIndexSelectionCast(tf_xy, 1),
                         sitk.Image(tf_xy.GetSize(), sitk.sitkFloat32))
    tf_z = sitk.Compose(sitk.Image(tf_z.GetSize(), sitk.sitkFloat32),
                        sitk.Image(tf_z.GetSize(), sitk.sitkFloat32),
                        sitk.Cast(tf_z, sitk.sitkFloat32))
    tf_xy = sitk.JoinSeries([tf_xy])
    tf_z = sitk.JoinSeries([tf_z])
    return combine_transforms(tf_z, tf_xy)


def process_transforms_(transforms, pyr=None, nonrigid=True):
    u_xy = {}
    l_xy = {}
    u_z = {}
    l_z = {}
    for k in transforms:
        n = int(k.split(',')[0])
        a = k.split(',')[1]
        b = k.split(',')[2]
        if a == 'xy':
            if b == 'u':
                u_xy[n] = sitk.Cast(transforms[k], sitk.sitkVectorFloat32)
            elif b == 'l':
                l_xy[n] = sitk.Cast(transforms[k], sitk.sitkVectorFloat32)
        elif a == 'z':
            if b == 'u':
                u_z[n] = transforms[k]
            elif b == 'l':
                l_z[n] = transforms[k]
        else:
            continue
        transforms[k] = None
    if nonrigid:
        u_xy, l_xy = optimize_transforms(u_xy, l_xy, pyr)
    else:
        u_xy = l_xy
    out = []
    for k in u_xy:
        out.append(combine_transforms_xy_z(u_xy[k], u_z[k]))
        out.append(combine_transforms_xy_z(l_xy[k], l_z[k]))
        u_xy[k], u_z[k], l_xy[k], l_z[k] = None, None, None, None
    return out


def process_transforms(pyr=4, nonrigid=True, **transforms):
    out = process_transforms_(transforms, pyr, nonrigid)
    return tuple(out)


def create_brain_(input_, internal_pixel_size, slice_thickness, output_path=None):
    brain = VISoRBrain()
    slices = {}
    ud = {}
    ld = {}
    for k in input_:
        i = int(k.split(',')[0])
        a = k.split(',')[1]
        if a == 'sl':
            slices[i] = input_[k]
        elif a == 'u':
            ud[i] = input_[k]
        elif a == 'l':
            ld[i] = input_[k]
        input_[k] = None
    for i in ud:
        sl = slices[i]
        u = ud[i]
        l = ld[i]
        u = sitk.Compose(sitk.VectorIndexSelectionCast(u, 0) * internal_pixel_size + sl.sphere[0][0],
                         sitk.VectorIndexSelectionCast(u, 1) * internal_pixel_size + sl.sphere[0][1],
                         sitk.VectorIndexSelectionCast(u, 2) * internal_pixel_size + (sl.sphere[0][2] - (i - 1) * slice_thickness))
        l = sitk.Compose(sitk.VectorIndexSelectionCast(l, 0) * internal_pixel_size + sl.sphere[0][0],
                         sitk.VectorIndexSelectionCast(l, 1) * internal_pixel_size + sl.sphere[0][1],
                         sitk.VectorIndexSelectionCast(l, 2) * internal_pixel_size + (sl.sphere[0][2] - i * slice_thickness))
        df = sitk.JoinSeries([u[:,:,0], l[:,:,0]])
        df.SetOrigin([0, 0, (i - 1) * slice_thickness])
        df.SetSpacing([internal_pixel_size, internal_pixel_size, slice_thickness])
        size = df.GetSize()
        df = sitk.Cast(df, sitk.sitkVectorFloat64)
        df = sitk.DisplacementFieldTransform(df)
        brain.slices[i] = sl
        brain.set_transform(i, df)
        brain.slice_spheres[i] = [[0, 0, (i - 1) * slice_thickness],
                                  [size[0] * internal_pixel_size, size[1] * internal_pixel_size, i * slice_thickness]]
        if output_path is not None:
            brain.save(output_path)
            brain.release_transform(i)
        ud[i] = None
        ld[i] = None
    brain.calculate_sphere()
    return brain


def create_brain(internal_pixel_size, slice_thickness, **input_):
    return create_brain_(input_, internal_pixel_size, slice_thickness)


def generate_brain_image(brain: VISoRBrain, img, slice_index, input_pixel_size, output_pixel_size, name_format, n_start,
                         roi=None, slice_origin=None, bit_downsample=True):
    if slice_origin is None:
        slice_origin = brain.slices[slice_index].sphere[0]
    img.SetOrigin(slice_origin)
    img.SetSpacing([input_pixel_size, input_pixel_size, input_pixel_size])
    if roi is None:
        roi = brain.slice_spheres[slice_index]
    size = [int((roi[1][j] - roi[0][j]) / output_pixel_size)
            for j in range(3)]
    print(size)
    res = sitk.Resample(img, size, brain.transform(slice_index), sitk.sitkLinear, roi[0],
                        [output_pixel_size, output_pixel_size, output_pixel_size])
    res.SetSpacing([j / 1000 for j in res.GetSpacing()])
    paths = [name_format.format(n_start + j) for j in range(size[2])]
    if not os.path.exists(os.path.dirname(paths[0])):
        os.makedirs(os.path.dirname(paths[0]))
    for i in range(size[2]):
        m = sitk.GetArrayFromImage(res[:, :, i])
        if bit_downsample:
            m = np.left_shift(np.right_shift((m + 8), 4), 4)
        tifffile.imwrite(paths[i], m, compress=1)
    file_list = paths.__str__()[2:-2].replace('\', \'', '\n')
    return file_list


def generate_brain_projection(input_image_list, thickness, output_path):
    input_image_list = input_image_list.split('\n')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(0, len(input_image_list), thickness):
        image = sitk.ReadImage(input_image_list[i: min(i + thickness, len(input_image_list))])
        image = sitk.MaximumProjection(image, 2)
        tifffile.imwrite(os.path.join(output_path, os.path.basename(input_image_list[i])), sitk.GetArrayViewFromImage(image), compress=1)
    return ''


def _tile_images(name_format, width, height, n_start, n_end, block_size, image_lists):
    image_lists = {(int(k.split(',')[0]), int(k.split(',')[1])): image_lists[k].split('\n') for k in image_lists}
    images = {name_format.format(i): {} for i in range(n_start, n_end)}
    output_path = os.path.dirname(name_format.format(0))
    for k in image_lists:
        for f in image_lists[k]:
            f_ = os.path.join(output_path, os.path.basename(f))
            if f_ in images:
                images[f_][k] = f
    output_files = []
    if not os.path.exists(os.path.dirname(name_format.format(0))):
        os.makedirs(os.path.dirname(name_format.format(0)))
    for f_ in images:
        tiles = {k: tifffile.imread(v) for k, v in images[f_].items()}
        img = np.zeros((height, width), np.uint16)
        for k, tile in tiles.items():
            np.copyto(img[k[1] * block_size: (k[1] + 1) * block_size,
                      k[0] * block_size: (k[0] + 1) * block_size], tile)
        tifffile.imwrite(f_, img, compress=1)
        output_files.append(f_)
    file_list = output_files.__str__()[2:-2].replace('\', \'', '\n')
    return file_list


def tile_images(name_format, width, height, n_start, n_end, block_size, **image_lists):
    return _tile_images(name_format, width, height, n_start, n_end, block_size, image_lists)

