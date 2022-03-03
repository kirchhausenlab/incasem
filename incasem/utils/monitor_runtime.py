import logging
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

logger = logging.getLogger(__name__)


def monitor_runtime(
        func,
        description=None,
        update_every=1,
        max_seconds=24 * 3600
):
    """Continuously update the time elapsed for a function call.

    Call with `monitor_runtime(func)(func_param_1, func_param_2, ...)`.
    It exploits the tqdm progress bar package.

    Args:

        func:

            An arbitrary function.

        description (str):

            Optional description to print instead of the plain function name.

        update_every (float):

            Check whether the function has terminated every `update_every`
            seconds and update the elapsed time if still running.

        max_seconds (int):

            `func` will be terminated forcefully after this many after this
            many seconds.
    """

    func_name = description if description is not None else func.__name__

    def wrapper(*args, **kwargs):
        pool = ThreadPool(processes=1)
        start = time.time()
        async_result = pool.apply_async(func, args, kwargs)
        pool.close()
        iterator = tqdm(
            range(max_seconds),
            leave=False,
            bar_format=f"{func_name} for {{elapsed}} mins ...")
        for i in iterator:
            try:
                if async_result.successful():
                    break
            except ValueError:
                if i == 0:
                    time.sleep(min(1, update_every))
                elif i == 1:
                    time.sleep(max(0, update_every - 1))
                else:
                    time.sleep(update_every)
        pool.join()
        out = async_result.get()
        total = int(time.time() - start)
        logger.info(
            f"Completed {func_name} in {total//60:02d}:{total%60:02d} mins"
        )

        return out

    return wrapper
