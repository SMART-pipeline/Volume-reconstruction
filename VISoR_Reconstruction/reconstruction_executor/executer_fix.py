import multiprocessing

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def __init__(self, **kwargs):
        super(NoDaemonProcess, self).__init__(**kwargs, daemon=False)
