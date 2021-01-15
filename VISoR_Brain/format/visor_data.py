import json, os, copy

class VISoRData():
    def __init__(self, file=None, raw_data_list=None):
        self.file = None
        self.path = None
        self.name = ''
        self.channels = {}
        self.misc = {}
        self.project_info = {
            "Animal ID": '',
            "Date": '',
            "Personnel": '',
            "Project Name": '',
            "Slide": '',
            "Subproject Name": ''
        }
        self.acquisition_results = {}
        self.reconstruction_info = {}
        self.slice_transform = {}
        self.brain_transform = None
        if file is not None:
            self.load(file)
        if raw_data_list is not None:
            self.create(raw_data_list)

    def load(self, file):
        self.file = file
        self.path = os.path.dirname(file)
        with open(file) as f:
            doc = json.load(f)
            self.channels = {i['ChannelId']:i for i in doc['Channels']}
            self.project_info = doc['Project Info']
            self.name = '{}_{}_{}'.format(self.project_info['Project Name'],
                                          self.project_info['Subproject Name'],
                                          self.project_info['Animal ID'])
            self.acquisition_results = {i:{} for i in self.channels}
            for d in doc['Acquisition Results']:
                for i in range(len(self.channels)):
                    path = d['FlsmList'][i]
                    if len(path) == 0:
                        continue
                    if not os.path.isabs(path):
                        path = os.path.join(self.path, path)
                    self.acquisition_results[doc['Channels'][i]['ChannelId']][int(d['SliceID'])] = path
                    if not os.path.exists(path):
                        print("Warning: Raw data file {} not found.")
            if 'Reconstruction' in doc:
                for k in doc['Reconstruction']:
                    path = os.path.join(self.path, doc['Reconstruction'][k])
                    try:
                        with open(path) as f_:
                            d = json.load(f_)
                            self.reconstruction_info[k] = d
                    except FileNotFoundError:
                        self.reconstruction_info[k] = {}
                try:
                    for k, v in self.reconstruction_info['SliceTransform']['SliceTransform'].items():
                        c = [c for c in self.channels if self.channels[c]['ChannelName'] == v['ChannelName']][0]
                        i = int(v['SliceID'])
                        if c not in self.slice_transform:
                            self.slice_transform[c] = {}
                        self.slice_transform[c][i] = os.path.join(self.path, 'Reconstruction/SliceTransform', k)
                    for k, v in self.reconstruction_info['BrainTransform']['BrainTransform'].items():
                        self.brain_transform = os.path.join(self.path, 'Reconstruction/BrainTransform', k)
                except KeyError as e:
                    print(e)

        self.misc = {k: doc[k] for k in doc if k not in {'Acquisition Results', 'Channels', 'Project Info'}}

    def to_dict(self, root_path=None):
        doc = {'Acquisition Results':[],
               'Channels':[i for i in self.channels.values()],
               'Project Info': self.project_info,
               **self.misc}
        d = {}
        for c in self.channels:
            for i in self.acquisition_results[c]:
                i_ = str(i)
                if i_ not in d:
                    d[i_] = {}
                d[i_][c] = self.acquisition_results[c][i]
        idx = [i for i in d]
        idx.sort(key=lambda x: int(x))
        for i in idx:
            s = {'FlsmList': [], 'SliceID': i}
            for c in self.channels:
                if c not in d[i]:
                    path = ''
                else:
                    path = d[i][c]
                    try:
                        path = os.path.relpath(path, root_path)
                    except:
                        pass
                s['FlsmList'].append(path)
            doc['Acquisition Results'].append(s)
        return doc

    def save(self, file):
        doc = self.to_dict(os.path.dirname(file))
        with open(file, 'w') as f:
            json.dump(doc, f, indent=4)

    def create(self, raw_data_list: list):
        d = {}
        for r in raw_data_list:
            if r.wave_length not in d:
                d[r.wave_length] = {}
            d[r.wave_length][r.z_index] = r

        caption = ''
        for i in range(len(d)):
            c = list(d)[i]
            r = list(d[c].values())[0]
            o = '10'
            if r.pixel_size < 0.6:
                o = '20'
            caption = r.caption
            self.channels[c] = {
                "ChannelId": str(i + 1),
                "ChannelName": '{}nm_{}X'.format(c, o),
                "EmissionFilter": r.info['filter'],
                "ExposureTime": r.info['exposure'],
                "Folder": '',
                "LaserPower": r.info['power'],
                "LaserWavelength": r.info['wave_length'],
                "MaxVolts":  r.info['max_volts'],
                "Objective": o,
                "SoftwareVersion": r.info['version'],
                "Spacing": r.info['move_y'],
                "Velocity": r.info['velocity'],
                "VoltsOffset": r.info['volts_offset']
            }

        for c, v in d.items():
            cid = self.channels[c]['ChannelId']
            self.acquisition_results[cid] = {}
            for i in v:
                self.acquisition_results[self.channels[c]['ChannelId']][i] = d[c][i].file
        self.channels = {self.channels[c]['ChannelId']: v for c, v in self.channels.items()}

        try:
            c = caption.split('_')
            self.project_info['Animal ID'] = c[-1]
            self.project_info['Project Name'] = c[0]
            self.project_info['Subproject Name'] = c[1]
        except:
            pass
