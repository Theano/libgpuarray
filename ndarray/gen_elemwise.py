"""
This file implement 1 version of the elemwise op on the gpu.

The elemwise fct are also used with scalar operation! So it can happen
that ndim is 0 as with all scalar type.
"""


import numpy
import StringIO

import pygpu_ndarray as gpu_ndarray
_CL_MODE = hasattr(gpu_ndarray, "set_opencl_context")


if _CL_MODE:
    # THIS IS NOT FINISHED
    import pyopencl as cl
    import pyopencl.array as cl_array
    from pyopencl.tools import dtype_to_ctype
#    import pyopencl._mymako as mako
    from pyopencl._cluda import CLUDA_PREAMBLE
    # TODO: use mako to get rid of the %if
    CLUDA_PREAMBLE = CLUDA_PREAMBLE[:455]
    CLUDA_PREAMBLE += """
#define LDIM_0 get_local_size(0)
#define LDIM_1 get_local_size(1)
#define LDIM_2 get_local_size(2)

#define GDIM_0 get_num_groups(0)
#define GDIM_1 get_num_groups(1)
#define GDIM_2 get_num_groups(2)
 """
    # TODO, reuse the same context as the use used to create the memory.
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
else:
    import pycuda.autoinit
    import pycuda.driver as driver
    from pycuda.compiler import SourceModule
    from pycuda.tools import dtype_to_ctype
#    import pycuda._mymako as mako
    from pycuda._cluda import CLUDA_PREAMBLE
    CLUDA_PREAMBLE += """
#define LDIM_0 blockDim.x
#define LDIM_1 blockDim.y
#define LDIM_2 blockDim.z

#define GDIM_0 gridDim.x
#define GDIM_1 gridDim.y
#define GDIM_2 gridDim.z
 """

from theano import Apply
from theano import scalar
from theano.tensor import TensorType
import theano

import logging
_logger_name = 'compyte.gen_elemwise'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())  # TO REMOVE


def warning(*msg):
    _logger.warning(_logger_name + 'WARNING: ' + ' '.join(str(m) for m in msg))


def info(*msg):
    _logger.info(_logger_name + 'INFO: ' + ' '.join(str(m) for m in msg))


def debug(*msg):
    _logger.debug(_logger_name + 'DEBUG: ' + ' '.join(str(m) for m in msg))


if _CL_MODE:
    gpu_ndarray.set_opencl_context(ctx.obj_ptr)


cast_int = numpy.intc
cast_uint = numpy.uintc


def _logical_scalar(x):
    return numpy.all(x.type.broadcastable)


def get_str_list_logical_scalar(inputs, value_str='ii_i%i_value',
                                data_str='ii_i%i_data[0]'):
    l = []
    for ipos, i in enumerate(inputs):
        if _logical_scalar(i):
            l += [value_str % ipos]
        else:
            l += [data_str % ipos]
    return l


class WrapOpenCLFunction(object):
    def __init__(self, fct):
        self.fct = fct

    def _param_wrap(self, p):
        if isinstance(p, MyGpuNdArray):
            p = p.gpu_nd_array
        if isinstance(p, gpu_ndarray.GpuNdArrayObject):
            p = cl.MemoryObject.from_cl_mem_as_int(p.bytes)
        return p

    def set_block_shape(self, *shape):
        self.local_size = shape

    def param_set(self, *param):
        self.param = [self._param_wrap(p) for p in param]

    def launch_grid(self, *global_shape):
        global_size = global_shape + (1,)

        d = {"g_times_l": True}
        return self.fct(queue, global_size, self.local_size,
                        *self.param, **d)


def compile_gpu_code(code, fct_name):
    if _CL_MODE:
        # Compile the gpu function with pyopencl
        prg = cl.Program(ctx, code).build()
        fct2 = getattr(prg, fct_name)

        fct = WrapOpenCLFunction(fct2)
    else:
        # Compile the gpu function with pycuda
        mod = SourceModule(code)
        fct = mod.get_function(fct_name)
    return fct


class ElemwiseAlgo(object):
    verbose = 0  # 1, 2 or 3 for more verbose output.
    cache_version = ()
    cache_version = ('debug', 14, verbose)

    def __init__(self, scalar_op, inplace_pattern={}):
        """
        :param scalar_op: the scalar operation to execute on each element.
        """
        self.scalar_op = scalar_op
        self.inplace_pattern = inplace_pattern

    def task_code(self, inputs, outputs, sio,
                  nodename, iname=None, oname=None):
        if iname == None:
            iname = get_str_list_logical_scalar(inputs)
        if oname == None:
            oname = ['ii_o%i_data[0]' % ipos for ipos, i in enumerate(outputs)]
        print >> sio, self.scalar_op.c_code(
            Apply(self.scalar_op,
                  [scalar.Scalar(dtype=input.type.dtype)()
                   for input in inputs],
                  [scalar.Scalar(dtype=output.type.dtype)()
                   for output in outputs]),
            nodename + '_scalar_',
            iname,
            oname,
            sub=dict(fail='return;'))  # TODO: set a failure code somehow!!!

    def c_src_kernel(self, inputs, outputs, nodename, nd, static="static"):
        sio = StringIO.StringIO()
        #print 'C_SRC_KERNEL', sio.getvalue()

        for ipos, i in enumerate(inputs):
            print >> sio, "//    Input  ", ipos, str(i.type)
        for ipos, i in enumerate(outputs):
            print >> sio, "//    Output ", ipos, str(i.type)
        print >> sio, static, (
            "KERNEL void kernel_%s_%s(unsigned int numEls" % (nodename, nd))
        if (nd):
            print >> sio, "\t,", ", ".join("const int dim%i" % i
                                           for i in xrange(nd))
        #declare inputs
        for ipos, i in enumerate(inputs):
            s = ", ".join(["GLOBAL_MEM const %s * i%i_data" % (
                        dtype_to_ctype(i.dtype), ipos)] +
                          list("int i%i_str_%i" % (ipos, d)
                               for d in xrange(nd)))
            print >> sio, "\t,", s
        #declare outputs
        for ipos, i in enumerate(outputs):
            s = ", ".join(["GLOBAL_MEM %s * o%i_data" % (
                        dtype_to_ctype(i.dtype), ipos)]
                          + list("int o%i_str_%i" % (ipos, d)
                                 for d in xrange(nd)))
            print >> sio, "\t,", s
            #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d)
            #                               for d in xrange(nd))
            #print >> sio, "\t,", "float * o%i_data" % ipos
        print >> sio, "\t)\n{"
        print >> sio, "    const int idx = GID_0 * LDIM_0 + LID_0;"
        print >> sio, "    const int numThreads = LDIM_0 * GDIM_0;"

        # For each input that is a scalar which has been broadcasted
        #     to a tensor, load it into a local variable
        for ipos, i in enumerate(inputs):
            if _logical_scalar(i):
                print >> sio, "    const %s ii_i%i_value = i%i_data[0];" % (
                    dtype_to_ctype(i.dtype), ipos, ipos)

        #loop over the elements to be treated by this kernel call
        print >> sio, "    for (int i = idx; i < numEls; i += numThreads) {"
        # calculate the data pointers for all arguments
        print >> sio, "        int ii = i;"
        for ipos, i in enumerate(inputs):
            if not _logical_scalar(i):
                print >> sio, ("        GLOBAL_MEM const "
                               "%s * ii_i%i_data = i%i_data;" % (
                    dtype_to_ctype(i.dtype), ipos, ipos))
        for ipos, i in enumerate(outputs):
            print >> sio, "        GLOBAL_MEM %s * ii_o%i_data = o%i_data;" % (
                dtype_to_ctype(i.dtype), ipos, ipos)
        for d in xrange(nd - 1, -1, -1):
            if d > 0:
                print >> sio, "        int pos%i = ii %% dim%i;" % (d, d)
                print >> sio, "        ii = ii / dim%i;" % d
            else:
                print >> sio, "        int pos%i = ii;" % d

            for ipos, i in enumerate(inputs):
                if not _logical_scalar(i):
                    print >> sio, ("        ii_i"
                                   "%i_data += pos%i * i%i_str_%i;" % (ipos, d, ipos, d))
            for ipos, i in enumerate(outputs):
                print >> sio, "        ii_o%i_data += pos%i * o%i_str_%i;" % (
                    ipos, d, ipos, d)

        # perform the scalar operation on the input and output references
        #TODO: What if the scalar_op needs support_code??
        self.task_code(inputs, outputs, sio, nodename)
        print >> sio, "    }"

        #indent = " "*(4*d+7)
        #for ipos, i in enumerate(inputs):
            #print >> sio, indent, "const float * i%i" % ipos, '= i%i_data', ''
        print >> sio, "}"

        #print sio.getvalue()
        return sio.getvalue()

    def c_src_kernel_Ccontiguous(self, inputs, outputs,
                                 nodename, static="static"):
        nd = outputs[0].type.ndim
        sio = StringIO.StringIO()
        #print 'C_SRC_KERNEL', sio.getvalue()

        for ipos, i in enumerate(inputs):
            print >> sio, "//    Input  ", ipos, str(i.type)
        for ipos, i in enumerate(outputs):
            print >> sio, "//    Output ", ipos, str(i.type)
        print >> sio, static, ("KERNEL void kernel_%s_Ccontiguous"
                               " (unsigned int numEls" % (nodename))
        #declare inputs
        for ipos, i in enumerate(inputs):
            print >> sio, "\t,", "GLOBAL_MEM const %s * i%i_data" % (
                dtype_to_ctype(i.dtype), ipos)
        #declare outputs
        for ipos, i in enumerate(outputs):
            print >> sio, "\t,", "GLOBAL_MEM %s * o%i_data" % (
                dtype_to_ctype(i.dtype), ipos)
        print >> sio, "\t)\n{"
        print >> sio, "    const int idx = GID_0 * LDIM_0 + LID_0;"
        print >> sio, "    const int numThreads = LDIM_0 * GDIM_0;"

        # For each input that is a scalar which has been broadcasted
        #     to a tensor, load it into a local variable
        for ipos, i in enumerate(inputs):
            if _logical_scalar(i):
                print >> sio, "    const %s ii_i%i_value = i%i_data[0];" % (
                    dtype_to_ctype(i.dtype), ipos, ipos)

        #loop over the elements to be treated by this kernel call
        print >> sio, "    for (int i = idx; i < numEls; i += numThreads) {"
        # perform the scalar operation on the input and output references
        #TODO: What if the scalar_op needs support_code??
        self.task_code(inputs, outputs, sio, nodename,
                       iname=get_str_list_logical_scalar(
                inputs, data_str='i%i_data[i]'),
                       oname=['o%i_data[i]' % ipos
                                for ipos, i in enumerate(outputs)])
        print >> sio, "    }"
        print >> sio, "}"

        #print sio.getvalue()
        return sio.getvalue()

    def c_src_callkernel(self, inputs, outputs, nodename):
        #
        # This function serves three main goals:
        #
        # The first is stride unpacking:
        # it accepts input and output arguments as
        #    float * , int*
        # pairs, and it constructs a kernel function call where inputs
        # and arguments are named like
        #    float *, int, int, int ...
        #
        # The second is to recognize when any dimensions can be collapsed as
        # being contiguous. That mean that we can merge that dimensions with
        # another one for all inputs/outputs and have the same retusuls
        # (confusing... read code)
        #
        # The thrid is to make a special case for scalar element. We allow
        # the collapsing of them.  In the ccontiguous and not contiguous case,
        # we use registers to lower the number of memory access.

        # TODO: make a special case for broadcasting, to store the
        # data in shared memory.

        nd = outputs[0].type.ndim
        nb_inputs = len(inputs)
        nb_outputs = len(outputs)
        d = dict()
        # input_params and output_params go into the function
        # declaration/definition
        input_params = ", ".join("const %s * i%i_data, const int * i%i_str" % (
                dtype_to_ctype(inputs[i].dtype), ipos, ipos)
                                 for ipos in xrange(len(inputs)))
        output_params = ", ".join("%s * o%i_data, const int * o%i_str" % (
                dtype_to_ctype(outputs[i].dtype),
                ipos, ipos)
                                  for ipos in xrange(len(outputs)))

        #input_args and output_args go into the recursive call.
        input_args = ", ".join("i%i_data, i%i_str" % (ipos, ipos)
                for ipos in xrange(len(inputs)))
        output_args = ", ".join("o%i_data, o%i_str" % (ipos, ipos)
                for ipos in xrange(len(outputs)))

        prod_dims = '*'.join(["dims[%i]" % di for di in xrange(nd)] + ['1'])

        sio = StringIO.StringIO()
        print >> sio, """
        static void can_collapse_%(nodename)s(int nd, const int * dims,
                                              const int * strides,
                                              int collapse[])
        {
            //can we collapse dims[i] and dims[i-1]
            for(int i=nd-1;i>0;i--){
                if(strides[i]*dims[i]==strides[i-1]){
                    //the dims nd-1 are not strided again dimension nd
                    collapse[i]=1;
                }else collapse[i]=0;
            }
        }
        """ % locals()
        print >> sio, """
        static int callkernel_%(nodename)s(unsigned int numEls, const int d,
            const int * dims,
            %(input_params)s,
            %(output_params)s)
        {
            numEls = %(prod_dims)s;
        """ % locals()
        if self.verbose:
            print >> sio, """
                std::cerr << "calling kernel_%(nodename)s     w numEls" << numEls << " dims"<< d << "\\n";
            """ % locals()
            print >> sio, 'std::cerr << ' + " << ' ' <<  ".join(['"  "']+list("dims[%i]"%di
                for di in xrange(nd)) + ["'\\n';"])
        if self.verbose > 1:
            for ipos in xrange(len(inputs)):
                print >> sio, """
                std::cerr << "   %(ipos)s data strides" <<
                """ % locals() + " << ' ' <<  ".join(["i%s_data" % ipos]
                + list("i%s_str[%i]" % (ipos, di)
                       for di in xrange(nd))) + ''' << "\\n"; '''

            for ipos in xrange(len(outputs)):
                print >> sio, """
                std::cerr << "   %(ipos)s data strides" <<
                """ % locals() + " << ' ' <<  ".join(["o%s_data" % ipos]
                    + list("o%s_str[%i]" % (ipos, di)
                           for di in xrange(nd))) + ''' << "\\n"; '''
    # collapse dimension that are broadcast in all inputs.
    # need to be done before contiguous collapse as it will break it.
    # do the dimensions and the strides
        print >> sio, """
        int local_dims[%(nd)s];
        int local_str[%(nb_inputs)s][%(nd)s];
        int local_ostr[%(nb_inputs)s][%(nd)s];
        int nd_collapse = %(nd)s;
        for(int i=0;i<%(nd)s;i++){//init new dim
          local_dims[i]=dims[i];
        }
        """ % locals()
        for ipos in xrange(len(inputs)):
            print >> sio, """
            for(int i=0;i<%(nd)s;i++){//init new strides
              local_str[%(ipos)s][i]=i%(ipos)s_str[i];
            }
            """ % locals()
        for ipos in xrange(len(outputs)):
            print >> sio, """
            for(int i=0;i<%(nd)s;i++){//init new strides
              local_ostr[%(ipos)s][i]=o%(ipos)s_str[i];
            }
            """ % locals()
        if self.verbose > 2:
            print >>sio, 'std::cerr <<"before broadcast collapse\\n";'
            print >>sio, 'std::cerr<< "nd_collapse "<< nd_collapse << "\\n"; '
            print >> sio, 'std::cerr << "local_dims";'
            for d in xrange(nd):
                print >> sio, 'std::cerr << " " << local_dims[%(d)s]; ' % locals()
            print >> sio, 'std::cerr << "\\n";'

            for ipos in xrange(len(inputs)):
                print >> sio, 'std::cerr << " local_str inputs %(ipos)s: " <<' % locals()+' << " " << '.join(["local_str[%(ipos)s][%(x)s]"%locals() for x in range(nd)])+'<<"\\n";'
            for ipos in xrange(len(outputs)):
                print >> sio, 'std::cerr << " local_ostr inputs %(ipos)s: " <<' % locals()+' << " " << '.join(["local_ostr[%(ipos)s][%(x)s]"%locals() for x in range(nd)])+'<<"\\n";'

        print >> sio, """
        for(int id=0;id<nd_collapse;id++){

          bool all_broadcast=true;
          for(int input_id=0;input_id<%(nb_inputs)s;input_id++){
            if(local_str[input_id][id]!=0 || local_dims[id]!=1) all_broadcast= false;
          }
          for(int input_id=0;input_id<%(nb_outputs)s;input_id++){
            if(local_ostr[input_id][id]!=0 || local_dims[id]!=1) all_broadcast= false;
          }
          if(all_broadcast){
            for(int j=id+1;j<nd_collapse;j++)//remove dims i from the array
              local_dims[j-1]=local_dims[j];
            for(int input_id=0;input_id<%(nb_inputs)s;input_id++){
              for(int j=id+1;j<nd_collapse;j++){//remove dims i from the array
                local_str[input_id][j-1]=local_str[input_id][j];
              }
            }
            for(int output_id=0;output_id<%(nb_outputs)s;output_id++){
              for(int j=id+1;j<nd_collapse;j++){//remove dims i from the array
                local_ostr[output_id][j-1]=local_ostr[output_id][j];
              }
            }
            nd_collapse--; id--;
          }
        }
        """ % locals()

        if self.verbose > 2:
            print >>sio, 'std::cerr <<"after broadcast collapse\\n";'
            print >>sio, 'std::cerr<< "nd_collapse "<< nd_collapse << "\\n"; '
            print >> sio, 'std::cerr << "local_dims";'
            for d in xrange(nd):
                print >> sio, 'std::cerr << " " << local_dims[%(d)s]; ' % locals()
            print >> sio, 'std::cerr << "\\n";'

            for ipos in xrange(len(inputs)):
                print >> sio, 'std::cerr << " local_str %(ipos)s: " <<' % locals()+' << " " << '.join(["local_str[%(ipos)s][%(x)s]"%locals() for x in range(nd)])+'<<"\\n";'
            for ipos in xrange(len(outputs)):
                print >> sio, 'std::cerr << " local_ostr %(ipos)s: " <<' % locals()+' << " " << '.join(["local_ostr[%(ipos)s][%(x)s]"%locals() for x in range(nd)])+'<<"\\n";'
    # collapse contiguous dimensions (ignoring scalars, generic version(collapse any dimensions, right, left, middle))
    # this is a good idea because we make less index calculation in the gpu.

        print >> sio, "int nd_collapse_[%(nd)s] = {" % locals() +','.join(['1' for x in range(nd)]) +"};"
        for ipos in xrange(len(inputs)):
            if not _logical_scalar(inputs[ipos]):
                print >> sio, """
                    int nd_collapse_%(ipos)s[%(nd)s] = {""" % locals() +','.join(['1' for x in range(nd)]) +"};"
                print >> sio, """
can_collapse_%(nodename)s(nd_collapse, local_dims, local_str[%(ipos)s], nd_collapse_%(ipos)s);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_%(ipos)s[i]==0)
nd_collapse_[i]=0;
}
                """ % locals()
                if self.verbose > 1:
                    print >>sio, """
                    std::cerr<< "nd_collapse_%(ipos)s "<<
                    """ % locals()
                    print >>sio, ' << " " << '.join(
                        ["nd_collapse_%(ipos)s[" % locals() + str(i) + "]"
                         for i in range(nd)])
                    print >>sio, '<< "\\n";'
                    print >>sio, """
                    std::cerr<< "nd_collapse_ "<<
                    """ % locals()
                    print >>sio, ' << " " << '.join(
                        ["nd_collapse_[" % locals() + str(i) + "]"
                         for i in range(nd)])
                    print >>sio, '<< "\\n";'

    # update the local stride.
        for ipos in xrange(len(inputs)):
            print >> sio, """
            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[%(ipos)s][i-1]=local_str[%(ipos)s][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[%(ipos)s][j-1]=local_str[%(ipos)s][j];
                }
            }
            """ % locals()

        for ipos in xrange(len(outputs)):
            print >> sio, """
            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_ostr[%(ipos)s][i-1]=local_ostr[%(ipos)s][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_ostr[%(ipos)s][j-1]=local_ostr[%(ipos)s][j];
                }
            }
            """ % locals()

    # update the local dims.
        print >> sio, """
        for(int i=nd_collapse-1;i>0;i--){
          if(nd_collapse_[i]==1){
            local_dims[i-1]*=local_dims[i];//set new dims
            for(int j=i+1;j<nd_collapse;j++)//remove dims i from the array
              local_dims[j-1]=local_dims[j];
          }
        }
        """ % locals()

    #update the new number of dim
        print >> sio, """
        for(int i=1, end=nd_collapse;i<end;i++){
          if(nd_collapse_[i]==1)nd_collapse--;
        }
        if(nd_collapse == 1 """ % locals()
        l = ["local_str[%(ipos)s][nd_collapse-1]==1 " % locals()
             for ipos in range(len(inputs))
             if not _logical_scalar(inputs[ipos])]
        l += ["local_ostr[%(ipos)s][nd_collapse-1]==1 " % locals()
              for ipos in range(len(outputs))
              if not _logical_scalar(outputs[ipos])]
        if len(l) > 0:
            print >> sio, " && ", " && ".join(l)
        print >> sio, """){nd_collapse=0;} """

        if self.verbose:
            print >> sio, 'std::cerr <<"after can_collapse\\n";'
            print >> sio, """std::cerr << "nd_collapse " << nd_collapse << "\\n"; """  % locals()
        if self.verbose > 1:
            for d in xrange(nd):
                print >> sio, 'std::cerr << " " << local_dims[%(d)s]; ' % locals()
            print >> sio, 'std::cerr << "\\n";'

            for ipos in xrange(len(inputs)):
                print >> sio, ('std::cerr << " local_str %(ipos)s: " <<' %
                               locals() + ' << " " << '.join(
                        ["local_str[%(ipos)s][%(x)s]" % locals()
                         for x in range(nd)]) + '<<"\\n";')
            for ipos in xrange(len(outputs)):
                print >> sio, ('std::cerr << " local_ostr %(ipos)s: " <<' %
                               locals() + ' << " " << '.join(
                        ["local_ostr[%(ipos)s][%(x)s]" % locals()
                         for x in range(nd)]) + '<<"\\n";')

        def launch_Ccontiguous(nodename, scalar_op):
            kernel_call_args = ["numEls"]
            for ipos in xrange(len(inputs)):
                kernel_call_args.append("i%i_data" % ipos)
            for ipos in xrange(len(outputs)):
                kernel_call_args.append("o%i_data" % ipos)
            kernel_call_args = ", ".join(kernel_call_args)
            verb = ""
            if self.verbose:
                verb = 'std::cerr << "   Running ccontiguous version\\n";'
            print >> sio, """
                //first use at least a full warp
                int threads_per_block = std::min(numEls,  (unsigned int)32); //WARP SIZE

                //next start adding multiprocessors
                int n_blocks = std::min(numEls/threads_per_block + (numEls %% threads_per_block?1:0), (unsigned int)30); // UP TO NUMBER OF MULTIPROCESSORS

                // next start adding more warps per multiprocessor
                if (threads_per_block * n_blocks < numEls)
                    threads_per_block = std::min(numEls/n_blocks, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                kernel_%(nodename)s_Ccontiguous<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);

                //std::cerr << "calling callkernel returned\\n";
                """  % locals()

            print >> sio, """
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n    n_blocks=%%i threads_per_block=%%i\\n   Call: %%s\\n",
                         "GpuElemwise %(nodename)s", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_%(nodename)s_Ccontiguous<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s)");
                    return -1;

                }
                %(verb)s
                return 0;
                """  % locals()

        def launch_General(nodename, scalar_op, force_nd):
            # kernel_call_args are used to invoke the cuda kernel
            local = "local_"
            kernel_call_args = ["numEls"]
            kernel_call_args.extend(local + "dims[%i]" % di
                                    for di in xrange(force_nd))
            for ipos in xrange(len(inputs)):
                kernel_call_args += ["i%i_data" % ipos] + list(
                    local + "str[%i][%i]" % (ipos, di)
                    for di in xrange(force_nd))
                #strides = ", ".join("i%i_str[%i]"%(ipos, di) for di in xrange(force_nd))
                #kernel_call_args.append( "%s, i%i_data" % (strides, ipos))
            for ipos in xrange(len(outputs)):
                kernel_call_args += ["o%i_data" % ipos] + list(
                    local + "ostr[%i][%i]" % (ipos, di)
                    for di in xrange(force_nd))
                #strides = ", ".join("o%i_str[%i]"%(ipos, di) for di in xrange(force_nd))
                #kernel_call_args.append( "%s, o%i_data" % (strides, ipos))
            if self.verbose:
                print >> sio, """
                    std::cerr << "   Running general version with %(force_nd)s  dims\\n";
                    """ % locals()
                print >> sio, "std::cerr << "+ ' << " " << '.join(
                    kernel_call_args)+' << "\\n";'
                #std::cerr << numEls << dims[0] << i0_data, i0_str[0] << o0_data, o0_str[0]\n;

            kernel_call_args = ", ".join(kernel_call_args)

            print >> sio, """
                //first use at least a full warp
                int threads_per_block = std::min(numEls, (unsigned int)32); //WARP SIZE

                //next start adding multiprocessors
                int n_blocks = std::min(numEls/threads_per_block + (numEls %% threads_per_block?1:0), (unsigned int)30); // UP TO NUMBER OF MULTIPROCESSORS

                // next start adding more warps per multiprocessor
                if (threads_per_block * n_blocks < numEls)
                    threads_per_block = std::min(numEls/n_blocks, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);

                kernel_%(nodename)s_%(force_nd)s<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);
                """  % locals()
            print >> sio, """
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n    n_blocks=%%i threads_per_block=%%i\\n   Call: %%s\\n",
                         "GpuElemwise %(nodename)s", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_%(nodename)s_Ccontiguous<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s)");
                    return -1;

                }
                return 0;
                """  % locals()

        print >> sio, "if(numEls==0) return 0;"
        print >> sio, "switch (nd_collapse==0?0:min(%(nd)s,nd_collapse)) {"%locals()
        print >> sio, "case 0: {"
        launch_Ccontiguous(nodename, scalar_op)
        print >> sio, "        } break;"
        for i in range(1, nd + 1):
            print >> sio, "case " + str(i) + ": {"
            launch_General(nodename, scalar_op, i)
            print >> sio, "        } break;"

        print >> sio, "}"  # end case
        print >> sio, "return -2;"  # should not get to this point
        print >> sio, "}"  # end fct

        #N.B. cudaGetLastError is called by c_code
        return sio.getvalue()

    def c_support_code_apply(self, inputs, outputs, nodename):
        nd = outputs[0].type.ndim
        return "".join(
            CLUDA_PREAMBLE,
            [self.c_src_kernel(inputs, outputs, nodename, x)
             for x in range(1, nd + 1)] +
            [self.c_src_kernel_Ccontiguous(inputs, outputs, nodename),
             self.c_src_callkernel(inputs, outputs, nodename),
             ])

    def c_code(self, ninputs, noutputs, nodename, inputs, outputs, sub):
        d = dict(sub)
        nd = noutputs[0].type.ndim
        d.update(locals())
        sio = StringIO.StringIO()
        nin = len(inputs)
        nout = len(outputs)
        fail = sub['fail']
        opname = str(self.scalar_op)
        initial_dims = ','.join('1' for i in xrange(nd))
        if 1 or self.scalar_op == scalar.pow:
            print >> sio, """
        //std::cerr << "C_CODE %(opname)s START\\n";
        //standard elemwise size checks
            """ % locals()
        print >> sio, """
        int dims[%(nd)s] = {%(initial_dims)s};
        """ % locals()

        #check that all inputs have valid dimensions
        emitted_inames = {}
        for id, iname in enumerate(inputs):
            if iname in emitted_inames:
                assert emitted_inames[iname] is ninputs[id]
                continue
            broadcasts = ', '.join(map(str, map(int,
                                                ninputs[id].broadcastable)))
            nd = ninputs[id].ndim
            print >> sio, """
        int broadcasts_%(iname)s[%(nd)s] = {%(broadcasts)s};
""" % locals()
            emitted_inames[iname] = ninputs[id]
        #check that all inputs have valid dimensions
        emitted_inames = {}
        for id, iname in enumerate(inputs):
            if iname in emitted_inames:
                continue
            print >> sio, """
        //std::cerr << "C_CODE %(opname)s checking input %(iname)s\\n";
        if (%(nd)s != %(iname)s->nd)
        {
            PyErr_Format(PyExc_TypeError, "need %(nd)s dims, not %%i", %(iname)s->nd);
            %(fail)s;
        }
        for (int i = 0; i< %(nd)s; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(%(iname)s)[i] : dims[i];
            if ((!(broadcasts_%(iname)s[i] && CudaNdarray_HOST_DIMS(%(iname)s)[i] == 1))&& (dims[i] != CudaNdarray_HOST_DIMS(%(iname)s)[i]))
            {
                //std::cerr << "C_CODE %(opname)s checking input %(iname)s failed\\n";
                PyErr_Format(PyExc_ValueError, "GpuElemwise. Input dimension mis-match. One of your inputs has shape[%%i] == %%i, but the output's size on that axis is %%i.",
                    i,
                    CudaNdarray_HOST_DIMS(%(iname)s)[i],
                    dims[i]
                    );
                %(fail)s;
            }
        }
            """ % locals()
            emitted_inames[iname] = True

        #check that all outputs have valid dimensions
        for idx, oname in enumerate(outputs):
            if idx not in self.inplace_pattern.keys():
                print >> sio, """
        for (int i = 0; (i< %(nd)s) && (%(oname)s); ++i) {
            if (dims[i] != CudaNdarray_HOST_DIMS(%(oname)s)[i])
            {
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
            }
        }
        if (NULL == %(oname)s)
        {
            %(oname)s = (CudaNdarray*)CudaNdarray_New();
            if (!%(oname)s)
            {
                //error string already set
                %(fail)s;
            }
            if (CudaNdarray_alloc_contiguous(%(oname)s, %(nd)s, dims))
            {
                //error string already set
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
                %(fail)s;
            }
        }
        //std::cerr << "ELEMWISE NEW %(oname)s nd" << %(oname)s->nd << "\\n";
        //std::cerr << "ELEMWISE NEW %(oname)s data" << %(oname)s->devdata << "\\n";
        """ % locals()
            else:
                input_idx = self.inplace_pattern[idx]
                iname = inputs[input_idx]
                print >> sio, """
        Py_XDECREF(%(oname)s);
        %(oname)s = %(iname)s;
        Py_INCREF(%(oname)s);
        for (int i = 0; (i< %(nd)s) && (%(oname)s); ++i) {
            if (dims[i] != CudaNdarray_HOST_DIMS(%(oname)s)[i])
            {
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
                %(fail)s;
            }
        }
        //std::cerr << "ELEMWISE NEW %(oname)s nd" << %(oname)s->nd << "\\n";
        //std::cerr << "ELEMWISE NEW %(oname)s data" << %(oname)s->devdata << "\\n";
        """ % locals()

        print >> sio, """
        {
            //new block so that failure gotos don't skip over variable initialization
            //std::cerr << "calling callkernel\\n";
            if (callkernel_%(nodename)s(1, 0, dims
            """ % locals()
        for iname in inputs:
            print >> sio, """
                        , CudaNdarray_DEV_DATA(%(iname)s), CudaNdarray_HOST_STRIDES(%(iname)s)
            """ % locals()
        for oname in outputs:
            print >> sio, """
                        , CudaNdarray_DEV_DATA(%(oname)s), CudaNdarray_HOST_STRIDES(%(oname)s)
            """ % locals()
        print >> sio, """
                        ))
            {
                 // error
            """
        for oname in outputs:
            print >> sio, """
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
                """ % locals()
        print >> sio, """
                %(fail)s;
            }
            else // no error
            {
            }
        }
        //std::cerr << "C_CODE %(opname)s END\\n";
        """ % locals()
        #print sio.getvalue()
        return sio.getvalue()

    def c_support_code(self):
        return """
        #define INTDIV_POW2(a, b) (a >> b)
        #define INTMOD_POW2(a, b) (a & ((1<<b)-1))
        """

def dummy_holder_for_code_not_used():

    def c_src_kernel_tiling(self, inputs, outputs, nodename):
        """ The kernel applies to problems with <= 5 dimensions """

        #The kernel is intended to be structured roughly like this:
        """
        static __global__ void kernel()
        {
            for (int v = blockIdx.y; v < dim0; v += gridDim.x)
            {
                for (int w = blockIdx.y; w < dim1; w += gridDim.y)
                {
                    for (int x = threadIdx.x; x < dim2; x += blockDim.x)
                    {
                        for (int y = threadIdx.y; y < dim3; y += blockDim.y)
                        {
                            for (int z = threadIdx.z; z < dim4; z += blockDim.z)
                            {
                                out[v * out_stride[0] + ...] = f(in1[...],  in2[...])
                            }
                        }
                    }
                }
            }
        }

        """

        nd = outputs[0].type.ndim
        sio = StringIO.StringIO()
        #print 'C_SRC_KERNEL', sio.getvalue()

        if nd in (4,):
            # print some leading comments to make the code easier to read
            for ipos, i in enumerate(inputs):
                print >> sio, "//    Input  ", ipos, str(i.type)
            for ipos, i in enumerate(outputs):
                print >> sio, "//    Output ", ipos, str(i.type)
            print >> sio, """static __global__ void kernel_%s_%s(
                             unsigned int numEls""" % (
                nodename,
                'tiling%i' % nd)
            if (nd):
                print >> sio, "\t,", ", ".join("const int dim%i" % i
                                               for i in xrange(nd))
            #declare inputs
            for ipos, i in enumerate(inputs):
                s = ", ".join(["const float * i%i_data" % ipos] + list(
                        "int i%i_str_%i" % (ipos, d) for d in xrange(nd)))
                print >> sio, "\t,", s
            #declare outputs
            for ipos, i in enumerate(outputs):
                s = ", ".join(["float * o%i_data" % ipos] + list(
                        "int o%i_str_%i" % (ipos, d) for d in xrange(nd)))
                print >> sio, "\t,", s
                #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
                #print >> sio, "\t,", "float * o%i_data" % ipos
            print >> sio, "\t)\n{"

            # For each input that is a scalar which has been broadcasted to a tensor,
            #     load it into a local variable
            print >> sio, "    __shared__ float value0[%i];" % len(inputs)
            print >> sio, "    __shared__ int shared_dims[%(nd)s];" % locals()
            #print >> sio, "    __shared__ int shared_i_str[%(n_in)s][%(nd)s]"
            print >> sio, "    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {"
            for ipos, i in enumerate(inputs):
                if _logical_scalar(i):
                    print >> sio, "    value0[%i] = i%i_data[0];" % (ipos,
                                                                     ipos)
            for ipos in xrange(nd):
                print >> sio, "    shared_dims[%i] = dim%i;" % (ipos, ipos)
            print >> sio, "    }"
            print >> sio, "    __syncthreads();"

            if (nd == 4):
                print >> sio, """
                for (int pos0 = blockIdx.x; pos0 < shared_dims[0]; pos0 += gridDim.x)
                {
                    for (int pos1 = blockIdx.y; pos1 < shared_dims[1]; pos1 += gridDim.y)
                    {
                        //for (int pos2 = threadIdx.x; pos2 < shared_dims[2]; pos2 += blockDim.x)
                        for (int pos2 = threadIdx.y; pos2 < shared_dims[2]; pos2 += blockDim.y)
                        {
                            //for (int pos3 = threadIdx.y; pos3 < shared_dims[3]; pos3 += blockDim.y)
                            for (int pos3 = threadIdx.x; pos3 < shared_dims[3]; pos3 += blockDim.x)
                            {
                """
            else:
                raise NotImplementedError()

            for ipos, i in enumerate(inputs):
                if not _logical_scalar(i):
                    print >> sio, "        const float * ii_i%i_data = i%i_data;" % (ipos, ipos)
            for ipos, i in enumerate(outputs):
                print >> sio, "        float * ii_o%i_data = o%i_data;" % (ipos, ipos)
            for d in xrange(nd):
                for ipos, i in enumerate(inputs):
                    if not _logical_scalar(i):
                        print >> sio, "        ii_i%i_data += pos%i * i%i_str_%i;" % (ipos, d, ipos, d)
                for ipos, i in enumerate(outputs):
                    print >> sio, "        ii_o%i_data += pos%i * o%i_str_%i;" % (ipos, d, ipos, d)

            # perform the scalar operation on the input and output references
            #TODO: What if the scalar_op needs support_code??
            self.task_code(inputs, outputs, sio, nodename,
                           iname=get_str_list_logical_scalar(
                    inputs, value_str='value0[%i]'))
            print >> sio, "    }" * nd

            #TODO: insert runtime stride checks that select the best loop order either here, or in
            # the host code that launched the  kernel (host code probably better spot)

            #indent = " "*(4*d+7)
            #for ipos, i in enumerate(inputs):
                #print >> sio, indent, "const float * i%i" % ipos, '= i%i_data', ''
            print >> sio, "}"

        print sio.getvalue()
        return sio.getvalue()

    def c_src_kernel_tiling_less_registers(self, inputs, outputs, nodename):
        """ The kernel applies to problems with <= 5 dimensions """

        nd = outputs[0].type.ndim
        n_in = len(inputs)
        n_out = len(outputs)
        sio = StringIO.StringIO()

        if nd not in (2,):
            return sio.getvalue()

        # print some leading comments to make the code easier to read
        for ipos, i in enumerate(inputs):
            print >> sio, "//    Input  ", ipos, str(i.type)
        for ipos, i in enumerate(outputs):
            print >> sio, "//    Output ", ipos, str(i.type)
        print >> sio, "static __global__ void kernel_%s_%s(unsigned int numEls" %(
                nodename,
                'tiling%i_less_registers'%nd)
        if (nd):
            print >> sio, "\t,", ", ".join("const int dim%i" % i
                                           for i in xrange(nd))
        #declare inputs
        for ipos, i in enumerate(inputs):
            s = ", ".join(["const float * i%i_data_0" % ipos] + list(
                    "int i%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print >> sio, "\t,", s
        #declare outputs
        for ipos, i in enumerate(outputs):
            s = ", ".join(["float * o%i_data_0" % ipos] + list(
                    "int o%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print >> sio, "\t,", s
            #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
            #print >> sio, "\t,", "float * o%i_data" % ipos
        print >> sio, "\t)\n{"

        # TODO: Setting these to true makes the function fail SOMETIMES.  I don't know why yet.
        use_shared_stride = False
        use_shared_limits = False

        def decl_limits(nd):
            if use_shared_limits:
                print >> sio, "__shared__ float * limits[%(nd)s];" % locals()

        def stride(io, p, d):
            if use_shared_stride:
                return "s%s_str[%i][%i]" % (io, p, d)
            else:
                return "%s%i_str_%i" % (io, p, d)

        def limits(d):
            if use_shared_limits:
                return "limits[%i]" % d
            else:
                return "limits%i" % d

        def decl_shared_stride(nin, nout, nd):
            if not use_shared_stride:
                return
            print >> sio, """
            __shared__ int si_str[%(nin)s][%(nd)s];
            __shared__ int so_str[%(nout)s][%(nd)s];
            if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
            """ % locals()
            for i in xrange(nin):
                for d in xrange(nd):
                    print >> sio, "si_str[%(i)s][%(d)s] = i%(i)s_str_%(d)s;" % locals()
            for i in xrange(n_out):
                for d in xrange(nd):
                    print >> sio, "so_str[%(i)s][%(d)s] = o%(i)s_str_%(d)s;" % locals()
            print >> sio, "} __syncthreads();"

        def calc_limit(d):
            s = stride('o', 0, d)
            lname = limits(d)
            if use_shared_limits:
                print >> sio, "if ((threadIdx.x == 0) && (threadIdx.y == 0)) {"
                if d == 0:
                    print >> sio, "%(lname)s = o0_data_0 + dim%(d)s * %(s)s;" % locals()
                else:
                    dm1 = d - 1
                    print >> sio, "%(lname)s = o0_data_%(dm1)s + dim%(d)s * %(s)s;" % locals()
                print >> sio, "} __syncthreads();"
            else:
                if d == 0:
                    print >> sio, "const float * %(lname)s = o0_data_0 + dim%(d)s * %(s)s;" % locals()
                else:
                    dm1 = d - 1
                    print >> sio, "const float * %(lname)s = o0_data_%(dm1)s + dim%(d)s * %(s)s;" % locals()

        def decl_ptrs(d, offset):
            dm1 = d - 1
            assert dm1 >= 0
            for i in xrange(n_in):
                s = stride('i', i, d)
                print >> sio, "const float * i%(i)s_data_%(d)s = i%(i)s_data_%(dm1)s + %(offset)s * %(s)s;" % locals()
            for i in xrange(n_out):
                s = stride('o', i, d)
                print >> sio, "float * o%(i)s_data_%(d)s = o%(i)s_data_%(dm1)s + %(offset)s * %(s)s;" % locals()

        def inc_ptrs(d, amt):
            for i in xrange(n_in):
                s = stride('i', i, d)
                print >> sio, "i%(i)s_data_%(d)s += %(amt)s * %(s)s;" % locals()
            for i in xrange(n_out):
                s = stride('o', i, d)
                print >> sio, "o%(i)s_data_%(d)s += %(amt)s * %(s)s;" % locals()

        def while_limit(d):
            lname = limits(d)
            print >> sio, "while (o0_data_%(d)s < %(lname)s) { " % locals()

        def end_while(d):
            print >> sio, "}"

        def task_code(d):
            self.task_code(inputs, outputs, sio, nodename,
                           iname=['i%i_data_%i[0]' % (ipos, d)
                                    for ipos, i in enumerate(inputs)],
                           oname=['o%i_data_%i[0]' % (ipos, d)
                                    for ipos, i in enumerate(outputs)])

        if nd == 4:
            decl_shared_stride(n_in, n_out, nd)
            decl_limits(nd)
            calc_limit(0)
            inc_ptrs(0, 'blockIdx.x')
            while_limit(0)
            if 1:
                calc_limit(1)
                decl_ptrs(1, 'blockIdx.y')
                while_limit(1)
                if 1:
                    calc_limit(2)
                    decl_ptrs(2, 'threadIdx.y')
                    while_limit(2)
                    if 1:
                        calc_limit(3)
                        decl_ptrs(3, 'threadIdx.x')
                        while_limit(3)
                        if 1:
                            task_code(3)
                            inc_ptrs(3, 'blockDim.x')
                        end_while(3)
                        inc_ptrs(2, 'blockDim.y')
                    end_while(2)
                    inc_ptrs(1, 'gridDim.y')
                end_while(1)
                inc_ptrs(0, 'gridDim.x')
            end_while(0)

        print >> sio, "}"
        print sio.getvalue()
        return sio.getvalue()


def elemwise_collapses(inputs, outputs, out_shape=None, verbose=0):
    """
    This collapse dimensions that are not needed when computing
    elemwise.  This is usefull as it lower the indexing computation
    that is heavier on gpu then on cpu.

    This is a generic version. It collapse dimensions at any place in
    the shape. It handle broadcasted dimensions correctly.

    There is no special handling needed for broadcasted scalar at this level.

    @return: ndims, tuple(dims, strides) after collapsing.
    """
    in_out = inputs + outputs
    del inputs
    if out_shape is not None:
        local_dims = tuple(out_shape)
    else:
        # TODO, use the right algo here or make the parameter not optional
        # We should always have the same shape for all outputs
        # If there is more then one outputs
        local_dims = tuple(outputs[0].shape)
    del outputs
    nd_orig = len(local_dims)
    if nd_orig == 1:
        # This have a lower overhead
        all_c_contig = True
        for inp in in_out:
            if not inp.flags['C_CONTIGUOUS'] or inp.shape != local_dims:
                all_c_contig = False
                break
        if all_c_contig:
            return 0, (local_dims, [])

    collapsable = [1] * nd_orig

    local_str = [None] * len(in_out)
    nd_collapse = nd_orig
    for ipos in xrange(len(in_out)):
        inp = in_out[ipos]
        assert len(inp.shape) == nd_orig, "All inputs/outputs must have the same number of dimensions. You must broadcast before calling elemwise_collapse"
        local_str[ipos] = list(inp.strides)
        # We set the strides of broacastable dims to 0
        # This make indexing in gpu simpler and is needed
        # For collapsing the dimensions.
        for dim_pos in range(inp.ndim):
            if inp.shape[dim_pos] == 1:
                local_str[ipos][dim_pos] = 0

    if nd_orig == 1:
        # We already covered the contiguous case before
        # So we are sure it is not contiguous
        # TODO: Add a test that f contiguous are also collapsed by the first case.
        #       I think that for 1d array when the flags f contiguous is true, c contiguous is also true.
        return 1, (local_dims, local_str)

    if verbose > 2:
        print "before broadcast collapse"
        print " nd_collapse", nd_collapse
        print " local_dims", local_dims
        for ipos in xrange(len(local_str)):
            print " local_str inputs", ipos, local_str[ipos]
    local_dims = list(local_dims)
    # Collapse dimension that are broadcast in all inputs.
    # need to be done before contiguous collapse as it will break it.
    # Update the dimensions and the strides
    for id in range(nd_collapse):
        if local_dims[id] == 1:
            # remove dims i from the array
            for j in range(id + 1, nd_collapse):
                local_dims[j - 1] = local_dims[j]
            # remove dims i from the array
            for input_id in range(len(in_out)):
                for j in range(id + 1, nd_collapse):
                    local_str[input_id][j - 1] = local_str[input_id][j]
            nd_collapse -= 1
            id -= 1  # TODO: what is this? How this work?

    if verbose > 2:
        print "after broadcast collapse"
        print " nd_collapse", nd_collapse
        print " local_dims", local_dims
        for ipos in xrange(len(local_str)):
            print " local_str inputs", ipos, local_str[ipos]

    nd_collapse_ = [1] * nd_orig
    for ipos in xrange(len(local_str)):
        # Can we collapse dims[i] and dims[i-1]?
        strides = local_str[ipos]
        for i in range(nd_collapse - 1, 0, -1):
            if strides[i] * local_dims[i] != strides[i - 1]:
                # The dims nd-1 are not strided again dimension nd
                nd_collapse_[i] = 0

        if verbose > 1:
            print "nd_collapse_", nd_collapse_

    nd_collapse2 = nd_collapse
    for i in range(nd_collapse - 1, 0, -1):
        if nd_collapse_[i] == 1:
            # update the local dims.
            local_dims[i - 1] *= local_dims[i]
            for j in range(i + 1, nd_collapse):
                local_dims[j - 1] = local_dims[j]

            # update the local stride.
            for ipos in xrange(len(local_str)):
                local_str[ipos][i - 1] = local_str[ipos][i]  # set new strides
                # remove stride i from the array
                for j in range(i + 1, nd_collapse):
                    local_str[ipos][j - 1] = local_str[ipos][j]

            # update the new number of dim
            nd_collapse2 -= 1
    nd_collapse = nd_collapse2

    if nd_collapse == 1:
        l = [local_str[ipos][nd_collapse - 1] == in_out[ipos].itemsize
             for ipos in range(len(local_str))]
        if all(l):
            nd_collapse = 0

    if verbose:
        print "end collapsing"
        print " nd_collapse", nd_collapse
    if verbose > 1:
        print " local_dims", local_dims
        for ipos in xrange(len(local_str)):
            print " local_str inputs", ipos, local_str[ipos]

    return nd_collapse, (local_dims, local_str)


def reduction_collapses(inout, axis, verbose=0):
    """
    This collapse dimensions that are not needed when computing
    reduction.  This is usefull as it lower the indexing computation
    that is heavier on gpu then on cpu.

    This is a generic version. It collapse dimensions at any place in
    the shape.
    @param: inout: tuple(input, output)
    @param: axis: None, interger, list of 1 interger
                  The axis over witch we will do reduction.
    @return: (ndims, (input dims, input strides, input pattern), out strides)
             after collapsing.

    :note: we suppose that we can always collapse the output dimensions.
    """
    input = inout[0]
    out = inout[1]
    # Some quick check. It is faster then the full version.
    if axis is None:
        # The output size is always 1, so we don't care about this strides
        if (input.flags['C_CONTIGUOUS'] or input.flags['F_CONTIGUOUS']):
            return 0, ((input.size,), (input.itemsize,), axis), (0,)
    if input.ndim == 1:
        assert axis == [0] or axis == 0 or axis is None
        # not c contiguous as the first if should have catched it.
        return 1, (input.shape, input.strides, axis), (0,)

    if not isinstance(axis, (list, tuple)):
        local_axis = [axis]
    else:
        local_axis = list(axis)

    # This is needed for the computing of the output strides
    assert axis is None or len(local_axis) == 1

    local_dims = list(input.shape)
    local_str = list(input.strides)
    out_strides = list(out.strides)

    nd_orig = len(local_dims)
    collapsable = [1] * nd_orig
    nd_collapse = nd_orig

    if verbose > 2:
        print "before broadcast collapse"
        print " nd_collapse", nd_collapse
        print " local_dims", local_dims
        print " local_str inputs", local_str
        print " local_axis", local_axis

    # Collapse dimension that are broadcast in all inputs.
    # need to be done before contiguous collapse as it will break it.
    # Update the dimensions and the strides
    for id in range(nd_collapse):
        if local_dims[id] == 1:
            for j in range(id + 1, nd_collapse):
                # remove dims i from the array
                local_dims[j - 1] = local_dims[j]
                # remove strides i from the array
                local_str[j - 1] = local_str[j]
                # remove output strides i from the array
                if axis is not None:
                    out_strides[j - 2] = out_strides[j - 1]
            if id in local_axis:
                local_axis.remove(id)
            for axis_pos in range(len(local_axis)):
                if local_axis[axis_pos] > id:
                    local_axis[axis_pos] -= 1

            nd_collapse -= 1
            id -= 1  # TODO: how this work?

    if verbose > 2:
        print "after broadcast collapse"
        print " nd_collapse", nd_collapse
        print " local_dims", local_dims
        print " local_str inputs", local_str
        print " local_axis", local_axis
        print " out_strides", out_strides

    nd_collapse_ = [1] * nd_orig
    # Can we collapse dims[i] and dims[i-1]?
    for i in range(nd_collapse - 1, 0, -1):
        if ((local_str[i] * local_dims[i] != local_str[i - 1])):
            # The dims nd-1 are not strided again dimension nd
            nd_collapse_[i] = 0
        elif (i in local_axis) != ((i - 1) in local_axis):
            nd_collapse_[i] = 0

    if verbose > 1:
        print "nd_collapse_", nd_collapse_

    nd_collapse2 = nd_collapse
    for i in range(nd_collapse - 1, 0, -1):
        if nd_collapse_[i] == 1:
            # update the local dims.
            local_dims[i - 1] *= local_dims[i]
            # set new strides
            local_str[i - 1] = local_str[i]
            #remove the old dims and strides
            for j in range(i + 1, nd_collapse):
                local_dims[j - 1] = local_dims[j]
                local_str[j - 1] = local_str[j]
                if axis is not None:
                    out_strides[j - 2] = out_strides[j - 1]

            if i in local_axis:
                local_axis.remove(i)
            for axis_pos in range(len(local_axis)):
                if local_axis[axis_pos] > i:
                    local_axis[axis_pos] -= 1

            # update the new number of dim
            nd_collapse2 -= 1

    nd_collapse = nd_collapse2

    if nd_collapse == 1:
        if local_str[nd_collapse - 1] == input.itemsize:
            nd_collapse = 0

    if verbose:
        print "end collapsing"
        print " nd_collapse", nd_collapse
    if verbose > 1:
        print " local_dims", local_dims
        print " local_str inputs", local_str
        print " local_axis", local_axis
        print " out_strides", out_strides

    #print input.shape, input.strides
    #print nd_collapse, (local_dims, local_str, local_axis)
    local_dims = local_dims[:nd_collapse]
    local_str = local_str[:nd_collapse]
    out_strides = out_strides[:nd_collapse]
    return nd_collapse, (local_dims, local_str, local_axis), out_strides


def call_elemwise(fct, input_vals, block=None, grid=None, out=None,
                  out_shape=None,
                  strides=None):
    """ Call an elemwise gpu function with gived inputs and block size.

    :param fct: The gpu function to call
    :param input_vals: a list of inputs to pass to fct
    :param block: int, the size of the block wanted
    :param grid: int, the size of the grid wanted
    :param out: Optional, the preallocated output. Must have the right shape
                and dtype.

    :param out_shape: Optional, if provided, we will suppose that the output,
                      have this shape event if it is not true.
    :param strides: Optional, if provided, we will use those strides for
                    the inputs and outputs.

    :note: param out_shape and strides are used for the collapsing of
           dimensions.
    """
    inp = input_vals[0]

    # Get the output and output shape to us
    if out_shape is None and out is None:
        out_shape = list(inp.shape)
        for i in input_vals[1:]:
        # dtype checked by pycuda before gpu call
            for s_i in range(len(inp.shape)):
                assert (inp.shape[s_i] == i.shape[s_i]
                        or inp.shape[s_i] == 1
                        or  i.shape[s_i] == 1)
                out_shape[s_i] = max(out_shape[s_i], inp.shape[s_i],
                                     i.shape[s_i])
    if out is None:
        out = gpu_ndarray.empty(out_shape, dtype=inp.dtype)
    elif out_shape is None:
        out_shape = out.shape

    # Arg: nb element
    args = [cast_uint(out.size)]
    # Arg: output shape to the arguments.
    for i in range(len(out_shape)):
        args.append(cast_int(out_shape[i]))

    # for each inputs and the output
    # add its ptr and strides
    nd = len(out_shape)
    idx = 0
    for i in list(input_vals) + [out]:
        itemsize = i.dtype.itemsize
        args.append(i)
        for j in range(nd):
            # We force a stride of 0 for broadcastable dimensions
            # This lower the index computation in the kernel.
            if strides is not None:
                # strides should have a strides of 0 for broadcasting.
                args.append(cast_int(strides[idx][j] / itemsize))
            elif i.shape[j] == 1:
                args.append(cast_int(0))
            else:
                args.append(cast_int(i.strides[j] / itemsize))
        idx += 1
    out_size = out.size
    # First use at least a full warp
    if block is None:
        block_ = min(32, out_size)
    else:
        block_ = block
    # Next start adding multiprocessors
    if grid is None:
        grid_ = min(out_size / block_ + (out_size % block_ != 0), 60)
    else:
        grid_ = grid
    # Next start adding more warps per multiprocessor
    if block is None:
        if block_ * grid_ < out_size:
            block_ = min(out_size / grid_, 512)

    # We bypass the pycuda wrapper gpu function call.
    # by calling directly the gpu function.
    # This is faster and lower the overhead.
    # Here is code that allow you to use the pycuda fct call.
    # d = {"block":(block_,1,1), "grid":(grid_,1)}
    # fct(*args, **d)
    fct.set_block_shape(block_, 1, 1)  # time_kernel
    fct.param_set(*args)
    fct.launch_grid(grid_, 1)
    return out


class MyGpuNdArray():
    _compiled_fct = {}

    def __init__(self, gpu_nd_array):
        #assert isinstance(gpu_nd_array, gpu_ndarray.GpuNdArrayObject)
        self.gpu_nd_array = gpu_nd_array
        self.ctype = dtype_to_ctype(self.gpu_nd_array.dtype)

    @staticmethod
    def gen_fct(op, inputs, nd, nodename="TestNodeName",
                collapse=True):
        if _CL_MODE:
            npy_ty = "typedef float npy_float32;\n"
        else:
            npy_ty = "typedef double npy_float64;\n typedef float npy_float32;\n"

        # Generate the gpu functions
        nb_in = len(inputs)
        fcts = [None]
        for nd in range(1, nd + 1):  # 1 to nd
            out = op(*[TensorType(i.gpu_nd_array.dtype,
                                  (False,) * nd)() for i in inputs])
            out_dtype = out.dtype
            node = out.owner
            elemwise_algo = ElemwiseAlgo(node.op.scalar_op)

            code = (CLUDA_PREAMBLE +
                    npy_ty +
                    elemwise_algo.c_src_kernel(node.inputs,
                                               node.outputs,
                                               nodename, nd,
                                               static=""))
            fct_name = "kernel_%s_%d" % (nodename, nd)
            fct = compile_gpu_code(code, fct_name)
            fcts.append(fct)

        # All inputs/outputs C contiguous case
        code = (npy_ty +
                CLUDA_PREAMBLE +
                elemwise_algo.c_src_kernel_Ccontiguous(
                node.inputs, node.outputs, nodename, static=""))
        fct_name = "kernel_%s_Ccontiguous" % nodename
        fcts[0] = compile_gpu_code(code, fct_name)

        def call_fct2(inputs, out=None):
            " Do dimensions collapsing before call the gpu code "
            assert len(inputs) == nb_in
            # dtype checked by pycuda
            # TODO: assert nb dim?

            inp = inputs[0]

            # Compute the output shape.
            out_shape = list(inp.shape)
            for i in inputs[1:]:
                for s_i in range(len(inp.shape)):
                    assert (inp.shape[s_i] == i.shape[s_i]
                            or inp.shape[s_i] == 1
                            or  i.shape[s_i] == 1)
                    out_shape[s_i] = max(out_shape[s_i], i.shape[s_i])
            # Create the output object
            if (out is None
                or out.dtype != out_dtype
                or out.shape != tuple(out_shape)):
                out = MyGpuNdArray(gpu_ndarray.empty(out_shape,
                                                     dtype=out_dtype))

            if collapse:
                # Do the collapsing.
                nd_col, info = elemwise_collapses(list(inputs), [out])
                # The two next line are usefull to force a call to the
                # c contiguous version:
                #nd_col = 0
                #info = [[],[]]
                out = call_elemwise(fcts[nd_col], inputs,
                                    out=out, out_shape=info[0][:nd_col],
                                    strides=info[1])
            else:
                out = call_elemwise(fcts[-1], inputs, out=out,
                                    out_shape=out_shape)
            return out
        return call_fct2

    def __elemwise2__(self, other, name, op):
        """ Call this code on this op with 2 inputs """
        nd = len(self.gpu_nd_array.shape)  # self.gpu_nd_array.ndim
        assert nd == len(other.gpu_nd_array.shape)  # ndim
        tag = (name + '_' + str(self.gpu_nd_array.dtype)
               + str(self.gpu_nd_array.ndim))
        tag += ('_' + str(other.gpu_nd_array.dtype)
                + str(other.gpu_nd_array.ndim))
        fct = self._compiled_fct.get(tag, None)
        if fct is None:
#            print "compile", tag
            fct = MyGpuNdArray.gen_fct(op, [self, other], nd)
            self._compiled_fct[tag] = fct
        return fct((self, other))

    @classmethod
    def __elemwise__(cls, inputs, name, op, out=None):
        """ Call this code on this op with * inputs """
        nd = len(inputs[0].gpu_nd_array.shape)  # self.gpu_nd_array.ndim
        for i in inputs[1:]:
            assert nd == len(i.gpu_nd_array.shape)  # ndim
        nb = len(inputs)
        tag = name + "_".join([str(i.gpu_nd_array.dtype) +
                             str(i.gpu_nd_array.ndim) for i in inputs])
        fct = cls._compiled_fct.get(tag, None)
        if fct is None:
#            print "compile", tag
            fct = MyGpuNdArray.gen_fct(op, inputs, nd)
            cls._compiled_fct[tag] = fct
        return fct(inputs, out=out)

    base = property(lambda self: self.gpu_nd_array.base)
    bytes = property(lambda self: self.gpu_nd_array.bytes)
    dtype = property(lambda self: self.gpu_nd_array.dtype)
    flags = property(lambda self: self.gpu_nd_array.flags)
    itemsize = property(lambda self: self.gpu_nd_array.itemsize)
    ndim = property(lambda self: self.gpu_nd_array.ndim,
                    doc="number of dimensions")
    offset = property(lambda self: self.gpu_nd_array.offset)
    shape = property(lambda self: self.gpu_nd_array.shape)
    size = property(lambda self: self.gpu_nd_array.size)
    strides = property(lambda self: self.gpu_nd_array.strides)

    def __array__(self):
        return numpy.asarray(self.gpu_nd_array)

    def copy(self):
        return MyGpuNdArray(self.gpu_nd_array.copy())

    def view(self):
        return MyGpuNdArray(self.gpu_nd_array.view())

    def __copy__(self):
        return MyGpuNdArray(self.gpu_nd_array.__copy__())

    def __deepcopy__(self):
        return MyGpuNdArray(self.gpu_nd_array.__deepcopy__())

    @property
    def gpudata(self):
        # TODO: Add this assert when PyCUDA/PyOpenCL can use the bytes
        # attributes. Without this assert old code that don't support
        # strides can receive as input object that are strided and no
        # error will be gived

        #assert (self.gpu_nd_array.flags['C_CONTIGUOUS'] or
        #         self.gpu_nd_array.flags['F_CONTIGUOUS'])

        # TODO: find a way to pass to a pycuda/pyopencl function the
        #       bytes + offset directly.
        return self.bytes + self.offset

    def __getitem__(self, *inputs):
        return MyGpuNdArray(self.gpu_nd_array.__getitem__(*inputs))

    def __add__(self, other):
        return self.__elemwise2__(other, "add", theano.tensor.add)

    def __sub__(self, other):
        return self.__elemwise2__(other, "sub", theano.tensor.sub)

    def __mul__(self, other):
        return self.__elemwise2__(other, "mul", theano.tensor.mul)

    def __div__(self, other):
        assert (str(self.gpu_nd_array.dtype).startswith("float") or
                str(other.gpu_nd_array.dtype).startswith("float"))
        return self.__elemwise2__(other, "true_div", theano.tensor.true_div)

    @classmethod
    def add(cls, x, y, out=None):
        """ add all inputs togethers element-wise """
        return cls.__elemwise__([x, y], "add", theano.tensor.add, out=out)

    @classmethod
    def adds(cls, *inputs):
        """ add all inputs togethers element-wise """
        return cls.__elemwise__(inputs, "add", theano.tensor.add)

    @classmethod
    def multiplys(cls, *inputs):
        """ multiply all inputs togethers element-wise """
        return cls.__elemwise__(inputs, "mul", theano.tensor.mul)

    def sum(self, axis=None, collapse=True):
        import gen_reduction
        max_thread_per_block = 512
        max_block = 4096
        if isinstance(axis, (list, tuple)):
            if len(axis) == 1:
                axis = axis[0]
            else:
                assert len(axis) == self.ndim
                axis.sort()
                assert axis == range(self.ndim)
                axis = None

        # TODO: Why this?
        if self.size == 0:
            make_out = gpu_ndarray.zeros
        else:
            make_out = gpu_ndarray.empty

        if axis is None:
            out = make_out((), self.dtype)
            out = MyGpuNdArray(out)
        else:
            out_shape = [self.shape[i] for i in range(self.ndim)
                         if i != axis]
            out = make_out(out_shape, self.dtype)
            out = MyGpuNdArray(out)

        if self.size == 0:
            return out

        args_set = False

        if collapse:
            coll_ndim, (coll_shape, coll_strides, coll_axis), coll_out_str = (
                reduction_collapses([self, out], axis))
        else:
            coll_ndim = self.ndim
            coll_shape = self.shape
            coll_strides = self.strides
            coll_axis = [axis]
            coll_out_str = out.strides

        if axis is not None:
            coll_axis = coll_axis[0]

        args_set = False

        if coll_ndim == 0:
            sum_op = gen_reduction.GpuSum([1], self.dtype)
            c_code = sum_op.c_support_code_apply("nodename", contig=True)
            fctname = "kernel_reduce_sum_ccontig_nodename"
            fct = compile_gpu_code(c_code, fctname)
            block_ = min(coll_shape[0], max_thread_per_block)
            block = (block_, 1, 1)

            grid = (1, 1)
            shared_ = self.dtype.itemsize * block_
            args = [cast_int(coll_shape[0]), self, out]
            args_set = True
        elif axis is None:
            pattern = [1] * coll_ndim
            str_pattern = [str(i) for i in pattern]
            sum_op = gen_reduction.GpuSum(pattern, self.dtype)
            c_code = sum_op.c_support_code_apply("nodename")
            if not c_code:
                raise NotImplementedError(
                    "GpuNdArray sum case not implemented")
            fctname = "kernel_reduce_sum_" + "".join(str_pattern) + "_nodename"
            fct = compile_gpu_code(c_code, fctname)
            if coll_ndim == 1:
                bx = min(max_thread_per_block, coll_shape[0])
                block = (bx, 1, 1)
                block_ = bx
            elif coll_ndim == 2:
                bx = min(max_thread_per_block, coll_shape[1])
                by = min(max_thread_per_block // coll_shape[1], coll_shape[0])
                by = max(by, 1)
                block = (bx, by, 1)
                block_ = bx * by
            elif coll_ndim == 3:
                bx = min(max_thread_per_block, coll_shape[2])
                by = min(max_thread_per_block // bx, coll_shape[1])
                bz = min(max_thread_per_block // (bx * by), coll_shape[0])
                by = max(by, 1)
                bz = min(max(bz, 1), 64)
                block = (bx, by, bz)
                block_ = bx * by * bz
            elif coll_ndim == 4:
                bx = min(max_thread_per_block, coll_shape[3])
                by = min(max_thread_per_block // bx, coll_shape[2])
                bz = min(max_thread_per_block // (bx * by), coll_shape[1])
                by = max(by, 1)
                bz = min(max(bz, 1), 64)
                block = (bx, by, bz)
                block_ = bx * by * bz
            grid = (1, 1)
            shared_ = self.dtype.itemsize * block_
        elif coll_ndim in [1, 2, 3]:
            if coll_ndim == 1:
                assert coll_axis == 0
                # pattern 1
                sum_op = gen_reduction.GpuSum([1], self.dtype)
                fctname = "kernel_reduce_sum_1_nodename"

                grid = (1, 1)

                block_ = min(max_thread_per_block, coll_shape[0])
                block = (block_, 1, 1)
            elif coll_ndim == 3 and coll_axis == 0:
                # pattern 100
                sum_op = gen_reduction.GpuSum([1, 0, 0], self.dtype)
                fctname = "kernel_reduce_sum_100_nodename"

                gx = min(coll_shape[1], max_block)
                gy = min(max_block // (gx * coll_shape[2]), coll_shape[2])
                gy = max(gy, 1)
                grid = (gx, gy)

                block_ = min(max_thread_per_block, coll_shape[0])
                block = (block_, 1, 1)
            elif coll_ndim == 3 and coll_axis == 1:
                # pattern 010
                sum_op = gen_reduction.GpuSum([0, 1, 0], self.dtype)
                fctname = "kernel_reduce_sum_010_AD_nodename"

                A = coll_shape[0]
                B = coll_shape[1]
                C = coll_shape[2]
                D = C / 32
                if (32 * D < C):
                    D += 1
                assert ((C <= 32 * D) and (32 * D < C + 32))
                shared_ = 0

                gx = min(A, max_block)
                gy = min(max_block // (D * A), D)
                gy = max(gy, 1)
                grid = (gx, gy)

                block = (32, 1, 1)
                block_ = 32

                args_set = True
                # input shape
                args = [cast_int(A), cast_int(B),
                        cast_int(C), cast_int(D)]
                # input
                args.append(self)
                # input strides
                args += [cast_int(i / self.dtype.itemsize)
                         for i in coll_strides]
                # output
                args.append(out)
                # output strides
                args.append(cast_int(coll_out_str[0] / out.dtype.itemsize))
                args.append(cast_int(coll_out_str[1] / out.dtype.itemsize))
            elif coll_ndim == 3 and coll_axis == 2:
                # pattern 001
                sum_op = gen_reduction.GpuSum([0, 0, 1], self.dtype)
                fctname = "kernel_reduce_sum_001_nodename"

                gx = min(coll_shape[0], max_block)
                gy = min(max_block // (gx * coll_shape[1]), coll_shape[1])
                gy = max(gy, 1)
                grid = (gx, gy)

                block_ = min(max_thread_per_block, coll_shape[2])
                block = (block_, 1, 1)
            elif coll_axis == 0:
                # pattern 10
                sum_op = gen_reduction.GpuSum([1, 0], self.dtype)
                fctname = "kernel_reduce_sum_010_nodename"
                block_ = min(coll_shape[1], max_thread_per_block)
                block = (block_, 1, 1)
                grid = (1, coll_shape[0])
                args_set = True
                # input shape
                args = [cast_int(1)]
                args += [cast_int(i) for i in coll_shape]
                # input
                args.append(self)
                # input strides
                args.append(cast_int(1))
                args += [cast_int(i / self.dtype.itemsize)
                         for i in coll_strides]
                # output
                args.append(out)
                # output strides
                args.append(cast_int(1))
                # We must take the last dimensions in the case of
                # dimensions collapsing.
                args.append(cast_int(coll_out_str[-1] / out.dtype.itemsize))
            elif coll_axis == 1:
                # pattern 01
                sum_op = gen_reduction.GpuSum([0, 1], self.dtype)
                fctname = "kernel_reduce_sum_01_nodename"
                block_ = min(coll_shape[1], max_thread_per_block)
                block = (block_, 1, 1)
                grid = (1, min(coll_shape[0], max_block))
            else:
                raise Exception("Bad axis")

            c_code = sum_op.c_support_code_apply("nodename")
            fct = compile_gpu_code(c_code, fctname)

            shared_ = self.dtype.itemsize * block_
        else:
            raise Exception("Not implemented")

        if not args_set:
            # input shape
            args = [cast_int(i) for i in coll_shape]
            # input
            args.append(self)
            # input strides
            args += [cast_int(i / self.dtype.itemsize)
                     for i in coll_strides]
            # output
            args.append(out)
            # output strides
            args += [cast_int(i / self.dtype.itemsize)
                     for i in coll_out_str]

        pycuda._driver.Context.synchronize()
        #print fctname, block, grid, shared_, axis
        #print self.ndim, self.shape, self.strides, axis, out.strides
        #print coll_ndim, coll_shape, coll_strides, coll_axis, coll_out_str
        #print args

        if False:
            d = {"block": block,
                 "shared": shared_,
                 "grid": grid}
            fct(*args, **d)
        else:
            # We bypass the pycuda wrapper gpu function call.
            # by calling directly the gpu function.
            # This is faster and lower the overhead.
            fct.set_block_shape(*block)
            fct.set_shared_size(shared_)
            fct.param_set(*args)
            fct.launch_grid(*grid)
        return out
