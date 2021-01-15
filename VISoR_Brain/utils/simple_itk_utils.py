import SimpleITK as sitk
import math


def downsample_image(image: sitk.Image, input_pixel_size, output_pixel_size):
    image.SetSpacing([input_pixel_size for i in range(3)])
    size = [int(math.ceil(i / output_pixel_size * input_pixel_size)) for i in image.GetSize()]
    return sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, [0, 0, 0],
                         [output_pixel_size for i in range(3)])

