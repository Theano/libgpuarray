from __future__ import print_function

import os
import unittest
from six.moves import range
from six import PY3
import pickle

import numpy

import pygpu
from pygpu.collectives import COMM_ID_BYTES, GpuCommCliqueId, GpuComm

from .support import (guard_devsup, check_meta, check_flags, check_all,
                      check_content, gen_gpuarray, context as ctx, dtypes_all,
                      dtypes_no_complex, skip_single_f)


def get_user_gpu_rank():
    for name in ['GPUARRAY_TEST_DEVICE', 'DEVICE']:
        if name in os.environ:
            devname = os.environ[name]
            if devname.startswith("opencl"):
                return -1
            if devname[-1] == 'a':
                return 0
            return int(devname[-1])
    return -1

# check for py4mpi to test
# check if rank == -1
gpurank = get_user_gpu_rank()
# mpirank = ...
# in both cases skip GpuComm tests

class TestGpuCommCliqueId(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cid = GpuCommCliqueId(context=ctx)

    def _create_in_scope_from_string(self):
        comm_id = bytearray(b'pipes' * (COMM_ID_BYTES // 5 + 1))
        return GpuCommCliqueId(context=ctx, comm_id=comm_id)

    def test_create_from_string_id(self):
        cid2 = self._create_in_scope_from_string()
        a = bytearray(b'pipes' * (COMM_ID_BYTES // 5 + 1))
        assert cid2.comm_id == a[:COMM_ID_BYTES], (cid2.comm_id, a[:COMM_ID_BYTES])
        b = bytearray(b'mlkies' * (COMM_ID_BYTES // 6 + 1))
        cid2.comm_id = b
        assert cid2.comm_id == b[:COMM_ID_BYTES], (cid2.comm_id, b[:COMM_ID_BYTES])
        with self.assertRaises(ValueError):
            cid2.comm_id = bytearray(b'testestestest')

    def test_pickle(self):
        with self.assertRaises(RuntimeError):
            pickle.dumps(self.cid)
        with self.assertRaises(RuntimeError):
            pickle.dumps(self.cid, protocol=0)
        with self.assertRaises(RuntimeError):
            pickle.dumps(self.cid, protocol=1)
        with self.assertRaises(RuntimeError):
            pickle.dumps(self.cid, protocol=2)
        if PY3:
            with self.assertRaises(RuntimeError):
                pickle.dumps(self.cid, protocol=3)
        with self.assertRaises(RuntimeError):
            pickle.dumps(self.cid, protocol=-1)

    def test_richcmp(self):
        other_cid1 = GpuCommCliqueId(context=ctx, comm_id=self.cid.comm_id)
        other_cid2 = GpuCommCliqueId(context=ctx)
        assert self.cid == other_cid1
        assert self.cid != other_cid2
        assert other_cid2 > other_cid1
        assert other_cid2 >= other_cid1
        assert self.cid >= other_cid1
        assert other_cid1 < other_cid2
        assert other_cid1 <= other_cid2
        assert other_cid1 <= self.cid
        with self.assertRaises(TypeError):
            a = other_cid1 > "asdfasfa"


#  class TestGpuComm(unittest.TestCase):
#      @classmethod
#      def setUpClass(cls):
#          # init mpi ?? make different main?
#          cls.cid = GpuCommCliqueId(context=ctx)
#          # broadcast common unique id
#          # set unique id
