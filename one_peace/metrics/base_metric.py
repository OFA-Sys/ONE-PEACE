
class BaseMetric(object):

    def __init__(self):
        pass

    def initialize(self):
        raise NotImplementedError

    def compute(self, models, sample):
        raise NotImplementedError

    def merge_results(self):
        raise NotImplementedError
