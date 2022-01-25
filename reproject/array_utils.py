
import numpy as np
from concurrent import futures
from scipy.ndimage import map_coordinates as scipy_map_coordinates

__all__ = ['map_coordinates']


def pad_edge_1(array):
    return np.pad(array, 1, mode='edge')


def _block_coords(image,coords,st,ed,**kwargs):
    block_out = scipy_map_coordinates(image,coords,**kwargs)
    return {'data':block_out,'st':st,'ed':ed}



def map_coordinates(image, coords, shapout ,threads,**kwargs):

    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by extending the array, updating the pixel coordinates,
    # then getting rid of values that were sampled in the range -1 to -0.5
    # and n to n - 0.5.
    threads = 1
    if threads > 1:
        thread_pool = futures.ThreadPoolExecutor(max_workers=threads)
    else:
        thread_pool = None
    

    original_shape = image.shape

    image = pad_edge_1(image)
    feed_coords = coords + 1
    if thread_pool:
        print("use_thread_coord.......")
        futures_list = []
        n_work_per_thread = int(np.ceil(feed_coords.shape[1]/threads))
        for idx in range(0,feed_coords.shape[1],n_work_per_thread):
            block_feed_coords = feed_coords[:,idx:idx+n_work_per_thread]
            futures_list.append(
                thread_pool.submit(
                    _block_coords,
                    image,
                    block_feed_coords,
                    idx,
                    idx+n_work_per_thread,
                    **kwargs
                    )
            )
        values = np.empty(shapout,dtype=image.dtype).ravel()

        for completed_future in futures.as_completed(futures_list):
            completed_block = completed_future.result()
            st = completed_block['st']
            ed = completed_block['ed']
            data = completed_block['data']
            values[st:ed] = data
            del data
            idx = futures_list.index(completed_future)
            completed_future._result = None
            del futures_list[idx],completed_future

    else:
        values = scipy_map_coordinates(image,feed_coords, **kwargs)

    reset = np.zeros(coords.shape[1], dtype=bool)

    for i in range(coords.shape[0]):
        reset |= (coords[i] < -0.5)
        reset |= (coords[i] > original_shape[i] - 0.5)

    values[reset] = kwargs.get('cval', 0.)

    return values
