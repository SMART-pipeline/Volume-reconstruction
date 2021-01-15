from VISoR_Brain.positioning.visor_sample import *

def reconstruct(sample_data: VISoRSample):
    r = sample_data.raw_data
    for i in range(len(r.columns)):
        affine_t = [0, 1 / r.pixel_size, 0,
                    0, 0, 1 / r.pixel_size / np.cos(r.angle),
                    1 / r.column_spacing[i], 0, 1 / r.column_spacing[i] / np.tan(r.angle)]

        p0 = r.column_pos0[i]
        af = sitk.AffineTransform(3)
        af.SetMatrix(affine_t)

        if r.column_spacing[i] < 0:
            p0 = (r.column_pos1[i][0], p0[1], p0[2])
        tl = np.subtract(r.pos0, p0).tolist()
        tl = af.TransformPoint(tl)
        af.Translate(tl)
        sample_data.transforms[i] = af