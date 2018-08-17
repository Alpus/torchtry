from collections import OrderedDict


def flatten(object_):
    """Traverse all objects in nested dicts and lists.

    Args:
        object_: any object including nested dicts and lists.

    Returns:
        traversed and changed object
    """

    if isinstance(object_, Untraversable):
        yield object_
    elif isinstance(object_, dict):
        for value in object_.values():
            yield from flatten(value)
    elif isinstance(object_, list) or isinstance(object_, tuple):
        for value in object_:
            yield from flatten(value)
    else:
        yield object_


def apply_collection(function_, object_):
    """Apply function over each object including nested dicts and lists.

    Args:
        object_: any object including nested dicts and lists.
        function_ (callable): function to apply

    Returns:
        mapped object
    """

    if hasattr(object_, 'is_untraversable') and object_.is_untraversable is True:
        new_object = function_(object_)
    elif isinstance(object_, dict):
        new_object = dict()
        for key in object_.keys():
            new_object[key] = apply_collection(function_, object_[key])
    elif isinstance(object_, list) or isinstance(object_, tuple):
        new_object = list()
        for item_number in range(len(object_)):
            new_object[item_number] = apply_collection(function_, object_[item_number])

        if isinstance(object_, tuple):
            new_object = tuple(new_object)
    else:
        new_object = function_(object_)

    return new_object


class Items:
    def __init__(self, **items):
        for key, value in items.items():
            self.__setattr__(key, value)


class Untraversable:
    pass


def untraversablate(class_):
    return type('Untraversable' + object_.__name__, (object_, Untraversable), dict())


UntraversableOrderedDict = untraversablate(OrderedDict)
