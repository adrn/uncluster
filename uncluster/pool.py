# -*- coding: utf-8 -*-
"""
Original author: Rodrigo Luger

Included in this project and licensed under MIT with his permission.

Implementations of four different types of processing pools:

    - MPIPool: An MPI pool borrowed from ``emcee``. This pool passes Python
      objects back and forth to the workers and communicates once per task.

    - MPIOptimizedPool: An attempt at an optimized version of the MPI pool,
      specifically for passing arrays of numpy floats. If the length of the
      array passed to the ``map`` method is larger than the number of processes,
      the iterable is passed in chunks, which are processed *serially* on each
      processor. This minimizes back-and-forth communication and should increase
      the speed a bit.

    - MultiPool: A multiprocessing for local parallelization, borrowed from
      ``emcee``

    - SerialPool: A serial pool, which uses the built-in ``map`` function

"""

from __future__ import division, print_function, absolute_import, unicode_literals

import logging
VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")

import numpy as np
import sys

import signal
import functools
import multiprocessing
import multiprocessing.pool
from astropy import log

__all__ = ['MultiPool', 'SerialPool', 'choose_pool']

class GenericPool(object):
    """ A generic multiprocessing pool object with a ``map`` method. """

    def __init__(self, **kwargs):
        self.rank = 0

    @staticmethod
    def enabled():
        return False

    def is_master(self):
        return self.rank == 0

    def is_worker(self):
        return self.rank != 0

    def wait(self):
        return NotImplementedError('Method ``wait`` must be called from subclasses.')

    def map(self, *args, **kwargs):
        return NotImplementedError('Method ``map`` must be called from subclasses.')

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

class SerialPool(GenericPool):

    def __init__(self, **kwargs):
        self.size = 0
        self.rank = 0

    @staticmethod
    def enabled():
        return True

    def wait(self):
        raise Exception('``SerialPool`` told to wait!')

    def map(self, function, iterable):
        return list(map(function, iterable))

# ----------------------------------------------------------------------------

def _initializer_wrapper(actual_initializer, *rest):
    """
    We ignore SIGINT. It's up to our parent to kill us in the typical
    condition of this arising from ``^C`` on a terminal. If someone is
    manually killing us with that signal, well... nothing will happen.

    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if actual_initializer is not None:
        actual_initializer(*rest)

class MultiPool(multiprocessing.pool.Pool):
    """
    This is simply ``emcee``'s :class:`InterruptiblePool`.

    A modified version of :class:`multiprocessing.pool.Pool` that has better
    behavior with regard to ``KeyboardInterrupts`` in the :func:`map` method.

    Contributed by Peter K. G. Williams <peter@newton.cx>.

    :param processes: (optional)
        The number of worker processes to use; defaults to the number of CPUs.

    :param initializer: (optional)
        Either ``None``, or a callable that will be invoked by each worker
        process when it starts.

    :param initargs: (optional)
        Arguments for *initializer*; it will be called as
        ``initializer(*initargs)``.

    :param kwargs: (optional)
        Extra arguments. Python 2.7 supports a ``maxtasksperchild`` parameter.

    """
    wait_timeout = 3600

    def __init__(self, processes=None, initializer=None, initargs=(),
                 **kwargs):
        new_initializer = functools.partial(_initializer_wrapper, initializer)
        super(MultiPool, self).__init__(processes, new_initializer,
                                        initargs, **kwargs)
        self.size = 0

    @staticmethod
    def enabled():
        '''

        '''

        return True

    def map(self, func, iterable, chunksize=None):
        """
        Equivalent of ``map()`` built-in, without swallowing
        ``KeyboardInterrupt``.

        :param func:
            The function to apply to the items.

        :param iterable:
            An iterable of items that will have `func` applied to them.

        """
        # The key magic is that we must call r.get() with a timeout, because
        # a Condition.wait() without a timeout swallows KeyboardInterrupts.
        r = self.map_async(func, iterable, chunksize)

        while True:
            try:
                return r.get(self.wait_timeout)
            except multiprocessing.TimeoutError:
                pass
            except KeyboardInterrupt:
                self.terminate()
                self.join()
                raise

def choose_pool(mpi=False, processes=1, **kwargs):
    """
    Chooses between the different pools.
    """

    if mpi:
        raise NotImplementedError("No MPI support yet!")

    elif processes != 1 and MultiPool.enabled():
        log.info("Running with multiprocessing on {} cores".format(processes))
        return MultiPool(processes=processes, **kwargs)

    else:
        log.info("Running serial")
        return SerialPool(**kwargs)
