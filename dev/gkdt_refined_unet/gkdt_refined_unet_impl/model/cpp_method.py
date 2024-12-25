
import numpy as np
import tensorflow as tf

from multiprocessing import Pool

from .cpp_densecrf import DenseCRF
# from .dcrf_base import DenseCRFBase

from ..config.global_config import GlobalConfig

def unary_from_image(image: np.ndarray, ugenerator: tf.Module) -> np.ndarray:
    unary = ugenerator(image[tf.newaxis, ...])[0]
    return unary.numpy()

def reference_from_image(image: np.ndarray, refgenerator: tf.Module) -> np.ndarray:
    ref = refgenerator(image)
    return ref.numpy()

def refine(unary: np.ndarray, reference: np.ndarray, dcrf: DenseCRF) -> np.ndarray:
    rfn = dcrf.mean_field_approximation(unary, reference)
    return rfn

def refine_mp(utiles: list[np.ndarray], reftiles: list[np.ndarray], config: GlobalConfig) -> list[np.ndarray]:
    # if config.dcrf_impl == "NP":
    #     from .np_densecrf import DenseCRF
    # elif config.dcrf_impl == "CPP":
    #     from .cpp_densecrf import DenseCRF
    # else:
    #     raise Exception("No implmentation.")

    if not len(utiles) == len(reftiles):
        raise Exception("len(utiles) != len(reftiles).")
    if not len(utiles) % config.n_processes == 0:
        raise Exception("Cannot deploy evenly.")
    
    n_parallels = len(utiles) // config.n_processes
    rfntiles = list()
    dcrfs = [DenseCRF(config) for _ in range(config.n_processes)]

    for i in range(n_parallels):
        print("Parallel", i + 1)
        pool = Pool(processes=config.n_processes)
        results = list()

        for j in range(config.n_processes):
            utile = utiles[i * config.n_processes + j]
            reftile = reftiles[i * config.n_processes + j]
            dcrf = dcrfs[j]
            results.append(pool.apply_async(
                refine, 
                args=(utile, reftile, dcrf, )
            ))

        pool.close()
        pool.join()

        for r in results:
            rfntiles.append(r.get())

    return rfntiles
    
    # - MP disabled, name to `refine`
    # dcrf = DenseCRF(config)
    # rfntiles = list()
    # for i in range(len(utiles)):
    #     utile, reftile = utiles[i], reftiles[i]
    #     rfntile = dcrf.mean_field_approximation(utile, reftile, )
    #     rfntiles.append(rfntile)

    # return rfntiles


