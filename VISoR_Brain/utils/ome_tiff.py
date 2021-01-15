import numpy
import typing
import SimpleITK as sitk
import tifffile


# Copy from https://github.com/ilastik/lazyflow
def generate_ome_xml_description(axes, shape, dtype, filename=''):
    """
    Generate an OME XML description of the data we're exporting,
    suitable for the image_description tag of the first page.

    axes and shape should be provided in C-order (will be reversed in the XML)
    """
    import uuid
    import xml.etree.ElementTree as ET

    # Normalize the inputs
    axes = "".join(axes)
    shape = tuple(shape)
    #if not isinstance(dtype, type):
    #    dtype = dtype().type

    ome = ET.Element('OME')
    uuid_str = "urn:uuid:" + str(uuid.uuid1())
    ome.set('UUID', uuid_str)
    ome.set('xmlns', 'http://www.openmicroscopy.org/Schemas/OME/2015-01')
    ome.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
    ome.set('xsi:schemaLocation',
            "http://www.openmicroscopy.org/Schemas/OME/2015-01 "
            "http://www.openmicroscopy.org/Schemas/OME/2015-01/ome.xsd")

    image = ET.SubElement(ome, 'Image')
    image.set('ID', 'Image:0')
    image.set('Name', 'exported-data')

    pixels = ET.SubElement(image, 'Pixels')
    pixels.set('BigEndian', 'true')
    pixels.set('ID', 'Pixels:0')

    fortran_axes = "".join(reversed(axes)).upper()
    pixels.set('DimensionOrder', fortran_axes)

    for axis, dim in zip(axes.upper(), shape):
        pixels.set('Size' + axis, str(dim))

    types = {sitk.sitkUInt8: 'uint8',
             sitk.sitkUInt16: 'uint16',
             sitk.sitkUInt32: 'uint32',
             sitk.sitkInt8: 'int8',
             sitk.sitkInt16: 'int16',
             sitk.sitkInt32: 'int32',
             sitk.sitkFloat32: 'float',
             sitk.sitkFloat64: 'double',
             sitk.sitkComplexFloat32: 'complex',
             sitk.sitkComplexFloat64: 'double-complex',
             numpy.dtype(numpy.uint8): 'uint8',
             numpy.dtype(numpy.uint16): 'uint16',
             numpy.dtype(numpy.uint32): 'uint32',
             numpy.dtype(numpy.int8): 'int8',
             numpy.dtype(numpy.int16): 'int16',
             numpy.dtype(numpy.int32): 'int32',
             numpy.dtype(numpy.float32): 'float',
             numpy.dtype(numpy.float64): 'double'}


    pixels.set('Type', types[dtype])

    ## Omit channel info for now
    # channel0 = ET.SubElement(pixels, "Channel")
    # channel0.set("ID", "Channel0:0")
    # channel0.set("SamplesPerPixel", "1")
    # channel0.append(ET.Element("LightPath"))

    assert axes[-2:] == "yx"
    for page_index, page_ndindex in enumerate(numpy.ndindex(*shape[:-2])):
        tiffdata = ET.SubElement(pixels, "TiffData")
        for axis, index in zip(axes[:-2].upper(), page_ndindex):
            tiffdata.set("First" + axis, str(index))
        tiffdata.set("PlaneCount", "1")
        tiffdata.set("IFD", str(page_index))
        uuid_tag = ET.SubElement(tiffdata, "UUID")
        uuid_tag.text = uuid_str
        uuid_tag.set('FileName', filename)

    from textwrap import dedent
    import sys
    import io
    xml_stream = io.BytesIO()
    # if sys.version_info.major == 2:
    #    xml_stream = io.BytesIO()
    # else:
    #    xml_stream = io.StringIO()

    comment = ET.Comment(
        ' Warning: this comment is an OME-XML metadata block, which contains crucial '
        'dimensional parameters and other important metadata. Please edit cautiously '
        '(if at all), and back up the original data before doing so. For more information, '
        'see the OME-TIFF web site: http://www.openmicroscopy.org/site/support/ome-model/ome-tiff/. ')
    ET.ElementTree(comment).write(xml_stream, encoding='utf-8', xml_declaration=True)

    tree = ET.ElementTree(ome)
    tree.write(xml_stream, encoding='utf-8', xml_declaration=False)

    return xml_stream.getvalue()


def write_ome_tiff(image, file_name: str, bit_downsample=True):
    if isinstance(image, sitk.Image):
        size = list(image.GetSize())
        size = (1, 1, size[2], size[1], size[0])
        dtype = image.GetPixelIDValue()
    else:
        size = image.shape
        size = (1, 1, size[0], size[1], size[2])
        dtype = image.dtype
    xml_description = generate_ome_xml_description('tczyx', size, dtype, file_name)
    total_size = size[0] * size[1] * size[2] * size[3] * size[4]
    print(total_size)
    if total_size > 4000000000:
        bigtiff = True
    else:
        bigtiff = False

    # Write the first slice with tifffile, which allows us to write the tags.
    with tifffile.TiffWriter(file_name, bigtiff=bigtiff) as writer:

        def get_page(i):
            if isinstance(image, sitk.Image):
                page = sitk.GetArrayFromImage(image[:,:,i])
            else:
                page = image[i]
            if bit_downsample:
                page = numpy.left_shift(numpy.right_shift((page + 8), 4), 4)
            return page

        writer.save(get_page(0), description=xml_description.decode('ascii'), compress=1)
        for i in range(1, size[2]):
            writer.save(get_page(i), compress=1)

