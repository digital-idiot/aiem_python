from types import SimpleNamespace

__all__ = ['Stash']


class Stash(SimpleNamespace):
    def __init__(self, **kwargs):
        super(Stash, self).__init__(**kwargs)

    def as_dict(self):
        return vars(self)

    def as_list(self):
        d = self.as_dict()
        return d.keys(), d.values()
