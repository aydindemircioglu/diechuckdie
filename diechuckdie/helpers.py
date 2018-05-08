
from copy import deepcopy


def getSliceList (slice, nSlices):
    '''
    Produce an iteratble list (for a for loop) from a subselection of slice indicies.
    If slice is None, return the whole range of nSlices,
    if it is a single entry or a list, we return this as a list.
    '''

    if slice is None:
        slice = list(range(nSlices))
    try:
        iterator = iter(slice)
    except TypeError:
        slice = [slice]
    return slice


# credit:
# https://stackoverflow.com/questions/1500718/what-is-the-right-way-to-override-the-copy-deepcopy-operations-on-an-object-in-p
def deepcopy_with_sharing(obj, shared_attribute_names, memo=None):
    '''
    Deepcopy an object, except for a given list of attributes, which should
    be shared between the original object and its copy.

    obj is some object
    shared_attribute_names: A list of strings identifying the attributes that
        should be shared between the original and its copy.
    memo is the dictionary passed into __deepcopy__.  Ignore this argument if
        not calling from within __deepcopy__.
    '''
    assert isinstance(shared_attribute_names, (list, tuple))
    shared_attributes = {k: getattr(obj, k) for k in shared_attribute_names}

    deepcopy_method = None
    if hasattr(obj, '__deepcopy__'):
        # Do hack to prevent infinite recursion in call to deepcopy
        deepcopy_method = obj.__deepcopy__
        obj.__deepcopy__ = None

    for attr in shared_attribute_names:
        del obj.__dict__[attr]

    clone = deepcopy(obj)

    for attr, val in shared_attributes.items():
        setattr(obj, attr, val)
        setattr(clone, attr, val)

    if deepcopy_method is not None:
        obj.__deepcopy__ = deepcopy_method
        clone.__deepcopy__ = None

    return clone
