import numpy as np

__all__ = ['map_coordinates']


def pad_edge_1(array):
    return np.pad(array, 1, mode='edge')


def map_coordinates(image, coords, **kwargs):

    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by extending the array, updating the pixel coordinates,
    # then getting rid of values that were sampled in the range -1 to -0.5
    # and n to n - 0.5.

    # from cupyx.scipy.ndimage import map_coordinates as scipy_map_coordinates
    # import cupy

    # original_shape = image.shape

    # #image = pad_edge_1(image)
    # image = cupy.asarray(image)
    # image = cupy.pad(image,1,mode="edge")
    # coords = cupy.asarray(coords)
    # output = cupy.asarray(kwargs["output"])
    # kwargs["output"]=output
    # #print(kwargs)
    # #print(type(image))
    # values = scipy_map_coordinates(image, coords + 1, **kwargs).get()
    # #print(type(values))
    # coords = coords.get()
    # reset = np.zeros(coords.shape[1], dtype=bool)

    # for i in range(coords.shape[0]):
    #     reset |= (coords[i] < -0.5)
    #     reset |= (coords[i] > original_shape[i] - 0.5)

    # values[reset] = kwargs.get('cval', 0.)

    from scipy.ndimage import map_coordinates as scipy_map_coordinates


    original_shape = image.shape

    #image = pad_edge_1(image)
    image = np.asarray(image)
    image = np.pad(image,1,mode="edge")
    coords = np.asarray(coords)
    output = np.asarray(kwargs["output"])
    kwargs["output"]=output
    #print(kwargs)
    #print(type(image))
    values = scipy_map_coordinates(image, coords + 1, **kwargs)
    #print(type(values))
    coords = coords
    reset = np.zeros(coords.shape[1], dtype=bool)

    for i in range(coords.shape[0]):
        reset |= (coords[i] < -0.5)
        reset |= (coords[i] > original_shape[i] - 0.5)

    values[reset] = kwargs.get('cval', 0.)

    return values
