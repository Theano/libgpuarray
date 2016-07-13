from __future__ import print_function

import os
import sys
import unittest
from six.moves import range
from six import PY3
import pickle

import numpy as np

from pygpu import gpuarray
from pygpu.collectives import COMM_ID_BYTES, GpuCommCliqueId, GpuComm

from pygpu.tests.support import (check_all, gen_gpuarray, context as ctx)


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

try:
    from mpi4py import MPI
    MPI_IMPORTED = True
except:
    MPI_IMPORTED = False
print("mpi4py found: " + str(MPI_IMPORTED), file=sys.stderr)


@unittest.skipIf(get_user_gpu_rank() == -1, "Collective operations supported on CUDA devices only.")
class TestGpuCommCliqueId(unittest.TestCase):
    def setUp(self):
        self.cid = GpuCommCliqueId(context=ctx)

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

    def test_create_from_previous(self):
        cid2 = GpuCommCliqueId(context=ctx, comm_id=bytearray(b'y' * COMM_ID_BYTES))
        cid3 = GpuCommCliqueId(context=ctx, comm_id=cid2.comm_id)
        assert cid2.comm_id == cid3.comm_id

    def test_richcmp(self):
        cid1 = GpuCommCliqueId(context=ctx, comm_id=bytearray(b'y' * COMM_ID_BYTES))
        cid2 = GpuCommCliqueId(context=ctx, comm_id=cid1.comm_id)
        cid3 = GpuCommCliqueId(context=ctx, comm_id=bytearray(b'z' * COMM_ID_BYTES))
        assert cid1 == cid2
        assert cid1 != cid3
        assert cid3 > cid2
        assert cid3 >= cid2
        assert cid1 >= cid2
        assert cid2 < cid3
        assert cid2 <= cid3
        assert cid2 <= cid1
        with self.assertRaises(TypeError):
            a = cid2 > "asdfasfa"

    def test_as_buffer(self):
        a = np.asarray(self.cid)
        assert np.allclose(a, self.cid.comm_id)
        a[:] = [ord(b'a')] * COMM_ID_BYTES
        assert np.allclose(a, self.cid.comm_id)


@unittest.skipUnless(MPI_IMPORTED, "Needs mpi4py module")
@unittest.skipIf(get_user_gpu_rank() == -1, "Collective operations supported on CUDA devices only")
class TestGpuComm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if get_user_gpu_rank() == -1 or not MPI_IMPORTED:
            return
        cls.mpicomm = MPI.COMM_WORLD
        cls.size = cls.mpicomm.Get_size()
        cls.rank = cls.mpicomm.Get_rank()
        cls.ctx = gpuarray.init("cuda" + str(cls.rank))
        print("*** Collectives testing for", cls.ctx.devname, file=sys.stderr)
        cls.cid = GpuCommCliqueId(context=cls.ctx)
        cls.mpicomm.Bcast(cls.cid, root=0)
        cls.gpucomm = GpuComm(cls.cid, cls.size, cls.rank)

    def test_count(self):
        assert self.gpucomm.count == self.size, (self.gpucomm.count, self.size)

    def test_rank(self):
        assert self.gpucomm.rank == self.rank, (self.gpucomm.rank, self.rank)

    def test_reduce(self):
        cpu, gpu = gen_gpuarray((3, 4, 5), order='c', incr=self.rank, ctx=self.ctx)
        rescpu = np.empty_like(cpu)

        resgpu = gpu._empty_like_me()
        if self.rank != 0:
            self.gpucomm.reduce(gpu, 'sum', resgpu, root=0)
            self.mpicomm.Reduce([cpu, MPI.FLOAT], None, op=MPI.SUM, root=0)
        else:
            self.gpucomm.reduce(gpu, 'sum', resgpu)
            self.mpicomm.Reduce([cpu, MPI.FLOAT], [rescpu, MPI.FLOAT], op=MPI.SUM, root=0)
        if self.rank == 0:
            assert np.allclose(resgpu, rescpu)

        resgpu = self.gpucomm.reduce(gpu, 'sum', root=0)
        if self.rank == 0:
            assert resgpu.shape == gpu.shape, (resgpu.shape, gpu.shape)
            assert resgpu.dtype == gpu.dtype, (resgpu.dtype, gpu.dtype)
            assert resgpu.flags['C'] == gpu.flags['C']
            assert resgpu.flags['F'] == gpu.flags['F']
            assert np.allclose(resgpu, rescpu)
        else:
            assert resgpu is None

        if self.rank == 0:
            resgpu = self.gpucomm.reduce(gpu, 'sum')
            assert resgpu.shape == gpu.shape, (resgpu.shape, gpu.shape)
            assert resgpu.dtype == gpu.dtype, (resgpu.dtype, gpu.dtype)
            assert resgpu.flags['C'] == gpu.flags['C']
            assert resgpu.flags['F'] == gpu.flags['F']
            assert np.allclose(resgpu, rescpu)
        else:
            resgpu = self.gpucomm.reduce(gpu, 'sum', root=0)
            assert resgpu is None

    def test_all_reduce(self):
        cpu, gpu = gen_gpuarray((3, 4, 5), order='c', incr=self.rank, ctx=self.ctx)
        rescpu = np.empty_like(cpu)
        resgpu = gpu._empty_like_me()

        self.gpucomm.all_reduce(gpu, 'sum', resgpu)
        self.mpicomm.Allreduce([cpu, MPI.FLOAT], [rescpu, MPI.FLOAT], op=MPI.SUM)
        assert np.allclose(resgpu, rescpu)

        resgpu = self.gpucomm.all_reduce(gpu, 'sum')
        assert resgpu.shape == gpu.shape, (resgpu.shape, gpu.shape)
        assert resgpu.dtype == gpu.dtype, (resgpu.dtype, gpu.dtype)
        assert resgpu.flags['C'] == gpu.flags['C']
        assert resgpu.flags['F'] == gpu.flags['F']
        assert np.allclose(resgpu, rescpu)

    def test_reduce_scatter(self):
        texp = self.size * np.arange(5 * self.size) + sum(range(self.size))
        exp = texp[self.rank * 5:self.rank * 5 + 5]

        # order c
        cpu = np.arange(5 * self.size) + self.rank
        np.reshape(cpu, (self.size, 5), order='C')
        gpu = gpuarray.asarray(cpu, context=self.ctx)

        resgpu = gpuarray.empty((5,), dtype='int64', order='C', context=self.ctx)

        self.gpucomm.reduce_scatter(gpu, 'sum', resgpu)
        assert np.allclose(resgpu, exp)

        # order f
        cpu = np.arange(5 * self.size) + self.rank
        np.reshape(cpu, (5, self.size), order='F')
        gpu = gpuarray.asarray(cpu, context=self.ctx)

        resgpu = gpuarray.empty((5,), dtype='int64', order='F', context=self.ctx)

        self.gpucomm.reduce_scatter(gpu, 'sum', resgpu)
        assert np.allclose(resgpu, exp)

        # make result order c (one less dim)
        cpu = np.arange(5 * self.size) + self.rank
        np.reshape(cpu, (self.size, 5), order='C')
        gpu = gpuarray.asarray(cpu, context=self.ctx)

        resgpu = self.gpucomm.reduce_scatter(gpu, 'sum')
        check_all(resgpu, exp)
        assert resgpu.flags['C_CONTIGUOUS'] is True

        # c-contiguous split problem (for size == 1, it can always be split)
        if self.size != 1:
            cpu = np.arange(5 * (self.size + 1), dtype='int32') + self.rank
            np.reshape(cpu, (self.size + 1, 5), order='C')
            gpu = gpuarray.asarray(cpu, context=self.ctx)
            with self.assertRaises(TypeError):
                resgpu = self.gpucomm.reduce_scatter(gpu, 'sum')

        # make result order f (one less dim)
        cpu = np.arange(5 * self.size) + self.rank
        np.reshape(cpu, (5, self.size), order='F')
        gpu = gpuarray.asarray(cpu, context=self.ctx)

        resgpu = self.gpucomm.reduce_scatter(gpu, 'sum')
        check_all(resgpu, exp)
        assert resgpu.flags['F_CONTIGUOUS'] is True

        # f-contiguous split problem (for size == 1, it can always be split)
        if self.size != 1:
            cpu = np.arange(5 * (self.size + 1), dtype='int32') + self.rank
            np.reshape(cpu, (5, self.size + 1), order='F')
            gpu = gpuarray.asarray(cpu, context=self.ctx)
            with self.assertRaises(TypeError):
                resgpu = self.gpucomm.reduce_scatter(gpu, 'sum')

        # make result order c (same dim - less size)
        texp = self.size * np.arange(5 * self.size * 3) + sum(range(self.size))
        exp = texp[self.rank * 15:self.rank * 15 + 15]
        np.reshape(exp, (3, 5), order='C')
        cpu = np.arange(5 * self.size * 3) + self.rank
        np.reshape(cpu, (self.size * 3, 5), order='C')
        gpu = gpuarray.asarray(cpu, context=self.ctx)

        resgpu = self.gpucomm.reduce_scatter(gpu, 'sum')
        check_all(resgpu, exp)
        assert resgpu.flags['C_CONTIGUOUS'] is True

        # make result order f (same dim - less size)
        texp = self.size * np.arange(5 * self.size * 3) + sum(range(self.size))
        exp = texp[self.rank * 15:self.rank * 15 + 15]
        np.reshape(exp, (5, 3), order='F')
        cpu = np.arange(5 * self.size * 3) + self.rank
        np.reshape(cpu, (5, self.size * 3), order='F')
        gpu = gpuarray.asarray(cpu, context=self.ctx)

        resgpu = self.gpucomm.reduce_scatter(gpu, 'sum')
        check_all(resgpu, exp)
        assert resgpu.flags['F_CONTIGUOUS'] is True

    def test_broadcast(self):
        if self.rank == 0:
            cpu, gpu = gen_gpuarray((3, 4, 5), order='c', incr=self.rank, ctx=self.ctx)
        else:
            cpu = np.zeros((3, 4, 5), dtype='float32')
            gpu = gpuarray.asarray(cpu, context=self.ctx)

        if self.rank == 0:
            self.gpucomm.broadcast(gpu)
        else:
            self.gpucomm.broadcast(gpu, root=0)
        self.mpicomm.Bcast(cpu, root=0)
        assert np.allclose(gpu, cpu)

    def test_all_gather(self):
        texp = np.arange(self.size * 10, dtype='int32')
        cpu = np.arange(self.rank * 10, self.rank * 10 + 10, dtype='int32')

        a = cpu
        gpu = gpuarray.asarray(a, context=self.ctx)
        resgpu = self.gpucomm.all_gather(gpu, nd_up=0)
        check_all(resgpu, texp)

        a = cpu.reshape((2, 5), order='C')
        exp = texp.reshape((2 * self.size, 5), order='C')
        gpu = gpuarray.asarray(a, context=self.ctx)
        resgpu = self.gpucomm.all_gather(gpu, nd_up=0)
        check_all(resgpu, exp)

        a = cpu.reshape((2, 5), order='C')
        exp = texp.reshape((self.size, 2, 5), order='C')
        gpu = gpuarray.asarray(a, context=self.ctx)
        resgpu = self.gpucomm.all_gather(gpu, nd_up=1)
        check_all(resgpu, exp)

        a = cpu.reshape((2, 5), order='C')
        exp = texp.reshape((self.size, 1, 1, 2, 5), order='C')
        gpu = gpuarray.asarray(a, context=self.ctx)
        resgpu = self.gpucomm.all_gather(gpu, nd_up=3)
        check_all(resgpu, exp)

        a = cpu.reshape((5, 2), order='F')
        exp = texp.reshape((5, 2 * self.size), order='F')
        gpu = gpuarray.asarray(a, context=self.ctx)
        resgpu = self.gpucomm.all_gather(gpu, nd_up=0)
        check_all(resgpu, exp)

        a = cpu.reshape((5, 2), order='F')
        exp = texp.reshape((5, 2, self.size), order='F')
        gpu = gpuarray.asarray(a, context=self.ctx)
        resgpu = self.gpucomm.all_gather(gpu, nd_up=1)
        check_all(resgpu, exp)

        a = cpu.reshape((5, 2), order='F')
        exp = texp.reshape((5, 2, 1, 1, self.size), order='F')
        gpu = gpuarray.asarray(a, context=self.ctx)
        resgpu = self.gpucomm.all_gather(gpu, nd_up=3)
        check_all(resgpu, exp)

        with self.assertRaises(Exception):
            resgpu = self.gpucomm.all_gather(gpu, nd_up=-2)
