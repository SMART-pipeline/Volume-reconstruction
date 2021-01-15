from .executor import *
import json, math
from VISoR_Brain.misc import *
from VISoR_Brain.format.visor_data import VISoRData
from VISoR_Reconstruction.misc import VERSION


def create_task(task_type, suffix, parameters, input_targets, output_targets):
    task = {'name': '{}_{}'.format(task_type, suffix),
            'type': task_type, 'input_targets': input_targets, 'output_targets': output_targets,
            'parameters': parameters}
    return task


def create_target(name, target_type, path, category=None, metadata=None):
    if category is None:
        category = ['Temp', name]
    if metadata is None:
        metadata = {}
    target = {'name': name, 'type': target_type,
              'path': path, 'category': category, 'metadata': metadata}
    return target


default_param = {
    "annotation_path": None,
    "brain_projection_thickness": 30,
    "channel_align_method": 'channel_elastix_align',
    "generate_brain_projection": False,
    "generate_projection": True,
    "ignore_channel": None,
    "ignore_slice": None,
    "internal_pixel_size": 4.0,
    "nonrigid": True,
    "output_path": None,
    "output_pixel_size": 4.0,
    "outside_brightness": 112,
    "reconstruct_brain": True,
    "reference_channel": None,
    "roi_height": 10000,
    "roi_width": 14000,
    "separate_brain_image": False,
    "slice_source": "thumbnail",
    "slice_template": "null",
    "slice_thickness": 300,
    "slice_stitch_method": 'elastix',
    "slice_stitch_channels": "all",
    "use_annotation": True,
    "use_rigidity_mask": False
}


preset_param = {'mouse_fine': {'internal_pixel_size': 4, 'output_pixel_size': 2, 'slice_source': 'raw'},
                'mouse_fast': {},
                'macaque_fast': {'slice_thickness': 300, 'internal_pixel_size': 10, 'output_pixel_size': 5,
                                 'slice_template': 'general'},
                'mouse_spinal_cord_fine': {'output_pixel_size': 2, 'slice_source': 'raw',
                                           'slice_template': 'mouse_spinal_cord'}
                }


def get_default_parameter(preset='mouse_fast'):
    return {**default_param, **preset_param[preset]}


def gen_brain_reconstruction_metadata(param):
    pass


def gen_brain_reconstruction_pipeline(dataset: VISoRData, preset='mouse_fast', **param):
    param = {**default_param, **preset_param[preset], **param}
    name = 'Reconstruction'
    tasks = {}
    source_d = {dataset.name: {}}
    ignore_slice = {}
    if param['ignore_slice'] is not None:
        ignore_slice = {int(s) for s in param['ignore_slice'].split(',')}

    def filter_channels(channels, channel_list_str, exclusive=True):
        filtered = {}
        if channel_list_str is not None:
            filtered = {s for s in channel_list_str.split(',')}
        out = []
        for c__ in channels:
            if (dataset.channels[c__]['ChannelName'] in filtered
                or dataset.channels[c__]['LaserWavelength'] in filtered) \
                    == exclusive:
                continue
            out.append(c__)
        return out

    for c in filter_channels(dataset.acquisition_results, param['ignore_channel']):
        for i in dataset.acquisition_results[c]:
            i_ = int(i)
            if i_ in ignore_slice:
                continue
            if c not in source_d[dataset.name]:
                source_d[dataset.name][c] = {}
            source_d[dataset.name][c][i_] = dataset.acquisition_results[c][i]

    ref_channel = list(source_d[dataset.name].keys())[0]
    if param['reference_channel'] is not None:
        ref_channel = filter_channels(source_d[dataset.name], param['reference_channel'], False)[0]

    if param['output_path'] is None:
        param['output_path'] = dataset.path
    if param['internal_pixel_size'] is None:
        param['internal_pixel_size'] = 4
    rec_path = os.path.join(param['output_path'], name)
    annotation_path = os.path.join(param['output_path'], 'Annotation')
    if param['annotation_path'] is not None:
        annotation_path = param['annotation_path']

    slice_stitch_channels = [ref_channel]
    if param['slice_stitch_channels'] == 'all':
        slice_stitch_channels = [c for c in source_d[dataset.name]]
    elif param['slice_stitch_channels'] is not None:
        slice_stitch_channels = filter_channels(source_d[dataset.name], param['slice_stitch_channels'], True)

    all_targets = {}

    def _create_target(*args, **kwargs):
        t = create_target(*args, **kwargs)
        all_targets[t['name']] = t
        return t
    _create_target('null', 'null', None)

    def _create_task(*args):
        t = create_task(*args)
        tasks[t['name']] = t
        return t

    output_pixel_size = param['output_pixel_size']
    internal_pixel_size = param['internal_pixel_size']
    roi_width = param['roi_width']
    roi_height = param['roi_height']

    raw_data_info = dataset.to_dict(param['output_path'])

    for b in source_d:
        channels = [k for k in source_d[b] if k != ref_channel]
        channels.sort(key=lambda c: float(c))
        channels.insert(0, ref_channel)

        # Remove raw data without correspondence reference channel
        c_ = []
        for c in channels:
            poplist = []
            for i in source_d[b][c]:
                if i not in source_d[b][ref_channel]:
                    poplist.append(i)
            for i in poplist:
                source_d[b][c].popitem(i)
            if len(source_d[b][c]) == 0:
                c_.append(c)
        for c in c_:
            source_d[b].popitem(c)

        for c in channels:
            channel_name = dataset.channels[c]['ChannelName']
            for i, r in source_d[b][c].items():
                slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                print(slice_name)

                # Stitch & channel align
                t_rawdata = _create_target('raw_data_{}'.format(slice_name), 'raw_data', r)
                parameters = {}
                input_targets = {'rawdata': t_rawdata}
                if c == ref_channel:
                    parameters['methods'] = {'stitch': 'elastix_align2'}
                else:
                    #ref_raw_data = source_d[b][ref_channel][i]
                    ref_slice_name = '{}_{:03d}_{}'.format(b, i, dataset.channels[ref_channel]['ChannelName'])
                    parameters['methods'] = {"align_channels": param["channel_align_method"]}
                    t_reference_slice = all_targets['slice_{}'.format(ref_slice_name)]
                    input_targets['reference_slice'] = t_reference_slice
                t_slice = _create_target('slice_{}'.format(slice_name), 'reconstructed_slice',
                                         os.path.join(rec_path, 'SliceTransform', slice_name + '.txt'),
                                         ['SliceTransform', 'SliceTransform'],
                                         {"SliceID": i, "ChannelName": channel_name})
                _create_task('reconstruct_sample', slice_name, parameters, input_targets, [t_slice])

            for i, r in source_d[b][c].items():
                slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)

                # Slice image reconstruction
                t_slice_image = _create_target('slice_image_{}'.format(slice_name), 'ome_tiff',
                                               os.path.join(rec_path, 'SliceImage/' + str(output_pixel_size),
                                                            slice_name + '.tif'),
                                               ['SliceImage', 'SliceImage'],
                                               {"SliceID": i, "ChannelName": channel_name, "PixelSize": output_pixel_size})
                _create_task('reconstruct_image', slice_name,
                             {'pixel_size': output_pixel_size, 'source': param['slice_source'], 'method': 'gpu_resample'},
                             {'sample_data': all_targets['slice_{}'.format(slice_name)]}, [t_slice_image])

            for i, r in source_d[b][c].items():
                if not param['generate_projection']:
                    break
                slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                # Generate projection
                t_slice_projection = _create_target('slice_projection_{}'.format(slice_name), 'image',
                                                    os.path.join(rec_path, 'Projection/' + str(output_pixel_size),
                                                                 slice_name + '.tif'),
                                                    ['Projection', 'Projection'],
                                                    {"SliceID": i, "ChannelName": channel_name, "PixelSize": output_pixel_size})
                _create_task('generate_projection', slice_name, {},
                             {'image': all_targets['slice_image_{}'.format(slice_name)]}, [t_slice_projection])

            for i, r in source_d[b][c].items():
                slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                # Downsample image
                t_downsampled = _create_target('downsampled_image_{}'.format(slice_name), 'ome_tiff',
                                              os.path.join(rec_path, 'SliceImage/' + str(internal_pixel_size),
                                                           slice_name + '.tif'),
                                               ['SliceImage', 'SliceImage'],
                                               {"SliceID": i, "ChannelName": channel_name, "PixelSize": internal_pixel_size})
                if not internal_pixel_size == output_pixel_size:
                    _create_task('downsample_image', slice_name,
                                 {'output_pixel_size': internal_pixel_size,
                                  'input_pixel_size': output_pixel_size},
                                 {'image': all_targets['slice_image_{}'.format(slice_name)]}, [t_downsampled])

            #'''
            for i, r in source_d[b][c].items():
                if not param['generate_projection']:
                    break
                if internal_pixel_size == output_pixel_size:
                    break
                slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                # Generate downsampled projection
                t_slice_projection = _create_target('downsampled_projection_{}'.format(slice_name), 'image',
                                                    os.path.join(rec_path, 'Projection/' + str(internal_pixel_size),
                                                                 slice_name + '.tif'),
                                                    ['Projection', 'Projection'],
                                                    {"SliceID": i, "ChannelName": channel_name, "PixelSize": internal_pixel_size})
                _create_task('generate_projection', 'downsampled_' + slice_name, {},
                             {'image': all_targets['downsampled_image_{}'.format(slice_name)]}, [t_slice_projection])
            #'''

        if not param['reconstruct_brain']:
            continue

        for i, r in source_d[b][ref_channel].items():
            channel_name = dataset.channels[ref_channel]['ChannelName']
            slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
            # Find slice surface
            t_uz = _create_target('uz_{}'.format(slice_name), 'image',
                                 os.path.join(rec_path, 'Temp', slice_name + '_uz.mha'))
            t_lz = _create_target('lz_{}'.format(slice_name), 'image',
                                 os.path.join(rec_path, 'Temp', slice_name + '_lz.mha'))
            _create_task('calc_surface_height_map', slice_name,
                         {'slice_thickness': param['slice_thickness'],
                          'internal_pixel_size': internal_pixel_size},
                         {'img': all_targets['downsampled_image_{}'.format(slice_name)]}, [t_uz, t_lz])

        for i, r in source_d[b][ref_channel].items():
            # Extract surface image
            for c in slice_stitch_channels:
                channel_name = dataset.channels[c]['ChannelName']
                slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                print(slice_name)
                t_us = _create_target('us_{}'.format(slice_name), 'image',
                                     os.path.join(rec_path, 'Temp', slice_name + '_us.mha'))
                t_ls = _create_target('ls_{}'.format(slice_name), 'image',
                                     os.path.join(rec_path, 'Temp', slice_name + '_ls.mha'))
                _create_task('extract_surface', slice_name, {},
                             {'img': all_targets['downsampled_image_{}'.format(slice_name)], 'umap': t_uz, 'lmap': t_lz},
                             [t_us, t_ls])

        # Align surfaces
        for i, r in source_d[b][ref_channel].items():
            input_targets = {}
            channel_name_ = dataset.channels[ref_channel]['ChannelName']
            slice_name_ = '{}_{:03d}_{}'.format(b, i, channel_name_)
            parameters = {'nonrigid': param['nonrigid'],
                          'outside_brightness': int(round((math.log(param['outside_brightness']) - 4.6) * 39.4)),
                          'use_rigidity_mask': param['use_rigidity_mask'],
                          'method': param['slice_stitch_method']}
            t_uxy = _create_target('uxy_{}'.format(slice_name_), 'image',
                                   os.path.join(rec_path, 'Temp', slice_name_ + '_uxy.mha'))
            if i - 1 in source_d[b][ref_channel]:
                prev_slice_name = '{}_{:03d}_{}'.format(b, i - 1, channel_name_)
                t_lxy = _create_target('lxy_{}'.format(prev_slice_name), 'image',
                                       os.path.join(rec_path, 'Temp', prev_slice_name + '_lxy.mha'))
            else:
                t_lxy = all_targets['null']
            for ct in range(len(slice_stitch_channels)):
                channel_name = dataset.channels[slice_stitch_channels[ct]]['ChannelName']
                slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                input_targets['next_surface_{}'.format(ct)] = all_targets['us_{}'.format(slice_name)]
                if i - 1 in source_d[b][ref_channel]:
                    prev_slice_name = '{}_{:03d}_{}'.format(b, i - 1, channel_name)
                    input_targets['prev_surface_{}'.format(ct)] = all_targets['ls_{}'.format(prev_slice_name)]
                else:
                    parameters['prev_surface_{}'.format(ct)] = None

                # Assign reference image
                if param['slice_template'] == 'mouse_brain':
                    n = i
                    if i > 46:
                        n = 46
                    if i < 1:
                        n = 1
                    t_ref = _create_target('reference_{}_{}'.format(n, ct), 'image',
                                           os.path.join(ROOT_DIR, 'data/slice_template/{:02d}.tif'.format(n)))
                    ref_scale = 4 / internal_pixel_size
                elif param['slice_template'] == 'macaque_brain':
                    n = i
                    if i > 250:
                        n = 250
                    if i < 1:
                        n = 1
                    t_ref = _create_target('reference_{}_{}'.format(n, ct), 'image',
                                           os.path.join(ROOT_DIR, 'data/macaque_brain_template/{:03d}.tif'.format(n)))
                    ref_scale = 250 / internal_pixel_size
                elif param['slice_template'] == 'mouse_spinal_cord':
                    n = i // 12
                    if i > 9:
                        n = 9
                    if i < 0:
                        n = 0
                    t_ref = _create_target('reference_{}_{}'.format(n, ct), 'image',
                                           os.path.join(ROOT_DIR, 'data/spinal_cord_template/{}.tif'.format(n)))
                    ref_scale = 4 / internal_pixel_size
                else:
                    t_ref = _create_target('reference_{}_{}'.format(i, ct), 'null',
                                           os.path.join(ROOT_DIR, 'data/general_template.png'))
                    ref_scale = 10 / internal_pixel_size
                parameters['ref_size'] = [int(roi_width / internal_pixel_size), int(roi_height / internal_pixel_size)]
                parameters['ref_scale'] = ref_scale
                input_targets['ref_img_{}'.format(ct)] = t_ref

                # Surface anchor points
                if param['use_annotation']:
                    lp_file = os.path.join(annotation_path, 'SurfaceRegistration', '{}_lp.txt'.format(i - 1))
                    if os.path.isfile(lp_file):
                        parameters['prev_points'] = lp_file
                    up_file = os.path.join(annotation_path, 'SurfaceRegistration', '{}_up.txt'.format(i))
                    if os.path.isfile(lp_file):
                        parameters['next_points'] = up_file

            _create_task('align_surfaces', 'u_' + slice_name_, parameters, input_targets, [t_lxy, t_uxy])

            # Last surface
            if i + 1 not in source_d[b][ref_channel]:
                for ct in range(len(slice_stitch_channels)):
                    channel_name = dataset.channels[slice_stitch_channels[ct]]['ChannelName']
                    slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                    t_lxy = _create_target('lxy_{}'.format(slice_name), 'image',
                                           os.path.join(rec_path, 'Temp', slice_name + '_lxy.mha'))
                    parameters = {k: v for k, v in parameters.items() if k != 'prev_surface'}
                    parameters['next_surface'] = None
                    _create_task('align_surfaces', 'l_' + slice_name, parameters,
                                 {'prev_surface': all_targets['ls_{}'.format(slice_name)], 'ref_img': t_ref},
                                 [t_lxy, all_targets['null']])

        input_targets_1, input_targets_2 = {}, {}
        output_targets_1 = []
        # Optimize transformation
        for i, r in source_d[b][ref_channel].items():
            channel_name = dataset.channels[ref_channel]['ChannelName']
            slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
            input_targets_1['{0},xy,u'.format(i)] = all_targets['uxy_{}'.format(slice_name)]
            input_targets_1['{0},xy,l'.format(i)] = all_targets['lxy_{}'.format(slice_name)]
            input_targets_1['{0},z,u'.format(i)] = all_targets['uz_{}'.format(slice_name)]
            input_targets_1['{0},z,l'.format(i)] = all_targets['lz_{}'.format(slice_name)]
            t_udf = _create_target('udf_{}'.format(slice_name), 'image',
                                   os.path.join(rec_path, 'Temp', slice_name + '_udf.mha'))
            t_ldf = _create_target('ldf_{}'.format(slice_name), 'image',
                                   os.path.join(rec_path, 'Temp', slice_name + '_ldf.mha'))
            output_targets_1.append(t_udf)
            output_targets_1.append(t_ldf)
            input_targets_2['{},sl'.format(i)] = all_targets['slice_{}'.format(slice_name)]
            input_targets_2['{},u'.format(i)] = t_udf
            input_targets_2['{},l'.format(i)] = t_ldf
        _create_task('process_transforms', b,
                     {'nonrigid': param['nonrigid']},
                     input_targets_1, output_targets_1)
        t_brain = _create_target('brain_{}'.format(b), 'reconstructed_brain',
                                 os.path.join(rec_path, 'BrainTransform', 'visor_brain.txt'),
                                 ['BrainTransform', 'BrainTransform'])
        _create_task('create_brain', b, {'internal_pixel_size': internal_pixel_size,
                                         'slice_thickness': param['slice_thickness']}, input_targets_2, [t_brain])

        # Generate brain image
        for j in range(len(channels)):
            c = channels[j]
            channel_name = dataset.channels[c]['ChannelName']
            for i, r in source_d[b][c].items():
                slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                t_dummy = _create_target('d_{}'.format(slice_name), 'file',
                                         os.path.join(rec_path, 'BrainImage', str(output_pixel_size), slice_name + '.txt'),
                                         ['BrainImage', 'BrainImage'],
                                         {"SliceID": i, "ChannelName": channel_name, "PixelSize": output_pixel_size})
                n_start = (i - 1) * int(param['slice_thickness'] / output_pixel_size)
                name_format = os.path.join(rec_path, 'BrainImage', str(output_pixel_size), 'Z{:05d}_' + 'C{}.tif'.format(c))
                if param['separate_brain_image']:
                    name_format = os.path.join(rec_path,
                                               'BrainImage',
                                               str(output_pixel_size),
                                               slice_name,
                                               'Z{:05d}_' + 'C{}.tif'.format(c))
                _create_task('generate_brain_image', slice_name,
                             {'slice_index': i, 'input_pixel_size': output_pixel_size, 'name_format': name_format,
                              'output_pixel_size': output_pixel_size, 'n_start': n_start},
                             {'brain': t_brain, 'img':all_targets['slice_image_{}'.format(slice_name)]}, [t_dummy])

        # Generate brain projection
        if param['generate_brain_projection']:
            for j in range(len(channels)):
                c = channels[j]
                channel_name = dataset.channels[c]['ChannelName']
                for i, r in source_d[b][c].items():
                    slice_name = '{}_{:03d}_{}'.format(b, i, channel_name)
                    t_dummy = _create_target('brain_projection_{}'.format(slice_name), 'file',
                                             os.path.join(rec_path, 'BrainProjection', str(output_pixel_size),
                                                          slice_name + '.txt'),
                                             ['BrainProjection', 'BrainProjection'],
                                             {"SliceID": i, "ChannelName": channel_name,
                                              "PixelSize": output_pixel_size})
                    _create_task('generate_brain_projection', slice_name,
                                 {'thickness': param['brain_projection_thickness'],
                                  'output_path': os.path.join(rec_path, 'BrainProjection', str(output_pixel_size))},
                                 {'input_image_list': all_targets['d_{}'.format(slice_name)]}, [t_dummy])


    metadata = {
        "Projection": {"ProjectionInfo": {
            "PixelSize": output_pixel_size,
            "Type": "Projection",
            "Software": "VISOR_Reconstruction",
            "Parameter": "../Parameters.json",
            "Version": VERSION,
            "Time": time.asctime(),
            "Transform": "../SliceTransform/SliceTransform.json"
        }},
        "SliceTransform": {"SliceTransformInfo": {
            "Type": "SliceTransform",
            "Software": "VISOR_Reconstruction",
            "Parameter": "../Parameters.json",
            "Version": VERSION,
            "Time": time.asctime()
        }},
        "SliceImage": {"SliceImageInfo": {
            "Type": "SliceImage",
            "Software": "VISOR_Reconstruction",
            "Parameter": "../Parameters.json",
            "Version": VERSION,
            "Time": time.asctime(),
            "Transform": "../SliceTransform/SliceTransform.json"
        }},
        "BrainTransform": {"BrainTransformInfo": {
            "Type": "BrainTransform",
            "Software": "VISOR_Reconstruction",
            "Parameter": "../Parameters.json",
            "Version": VERSION,
            "Time": time.asctime()
        }},
        "BrainImage": {"BrainImageInfo": {
            "Type": "BrainImage",
            "Software": "VISOR_Reconstruction",
            "Parameter": "../Parameters.json",
            "Version": VERSION,
            "Time": time.asctime(),
            "Transform": "../BrainTransform/BrainTransform.json"
        }},
        "BrainProjection": {"BrainProjectionInfo": {
            "Type": "BrainProjection",
            "Software": "VISOR_Reconstruction",
            "Parameter": "../Parameters.json",
            "Version": VERSION,
            "Time": time.asctime(),
            "BrainImage": "../BrainImage/BrainImage.json"
        }}
    }
    doc = {'tasks': tasks, 'name': name,
           'path': param['output_path'],
           'parameters': param,
           'metadata': metadata,
           'raw_data_info': raw_data_info}
    return json.dumps(doc, indent=2)

