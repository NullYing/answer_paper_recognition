

class ScanException(Exception):
    def __init__(self, value):
        self.value = value


class ImageException(ScanException):
    def __init__(self, value):
        super(ImageException, self).__init__(value)
