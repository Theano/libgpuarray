from mako.template import Template

from tools import ArrayArg, check_args
from elemwise import parse_c_args, massage_op

import numpy
import gpuarray

basic_kernel = Template("""
${preamble}

#define REDUCE(a, b) (${reduce_expr})

KERNEL void ${name}(const unsigned int n, ${out_arg.decltype()} *out
% for d in range(nd):
                    , const unsigned int dim${d}
% endfor
% for arg in arguments:
    % if arg.isarray():
                    , ${arg.decltype()} ${arg.name}_data
                    , const unsigned int ${arg.name}_offset
        % for d in range(nd):
                    , const int ${arg.name}_str_${d}
        % endfor
    % else:
                    , ${arg.decltype()} ${arg.name}
    % endif
% endfor
) {
  LOCAL_MEM ${out_arg.ctype()} ldata[${local_size}];
  const unsigned int lid = LID_0;
  const unsigned int idx = LDIM_0 * GID_0 + lid;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;
  GLOBAL_MEM char *tmp;

% for arg in arguments:
  % if arg.isarray():
  tmp = (GLOBAL_MEM char *)${arg.name}_data; tmp += ${arg.name}_offset;
  ${arg.name}_data = (${arg.decltype()})tmp;
  % endif
% endfor

  ${out_arg.ctype()} acc = ${neutral};
  for (i = idx; i < n; i += numThreads) {
    int ii = i;
    int pos;
% for arg in arguments:
    % if arg.isarray():
        GLOBAL_MEM char *${arg.name}_p = (GLOBAL_MEM char *)${arg.name}_data;
    % endif
% endfor
% for i in range(nd-1, -1, -1):
    % if i > 0:
        pos = ii % dim${i};
        ii = ii / dim${i};
    % else:
        pos = ii;
    % endif
    % for arg in arguments:
        % if arg.isarray():
            ${arg.name}_p += pos * ${arg.name}_str_${i};
        % endif
    % endfor
% endfor
% for arg in arguments:
    % if arg.isarray():
    ${arg.decltype()} ${arg.name} = (${arg.decltype()})${arg.name}_p;
    % endif
% endfor
    acc = REDUCE(acc, ${expression});
  }
  ldata[lid] = acc;

  <% cur_size = local_size %>
  % while cur_size > 1:
    <% cur_size = cur_size / 2 %>
    local_barrier();
    if (lid < ${cur_size}) {
      ldata[lid] = REDUCE(ldata[lid], ldata[lid+${cur_size}]);
    }
  % endwhile
  if (lid == 0) out[GID_0] = ldata[0];
}
""")

contig_kernel = Template("""
${preamble}

#define REDUCE(a, b) (${reduce_expr})

KERNEL void ${name}(const unsigned int n, ${out_arg.decltype()} out
% for arg in arguments:
                    , ${arg.decltype()} ${arg.name}
    % if arg.isarray():
                    , const unsigned int ${arg.name}_offset
    % endif
% endfor
) {
  LOCAL_MEM ${out_arg.ctype()} ldata[${local_size}];
  const unsigned int lid = LID_0;
  const unsigned int idx = LDIM_0 * GID_0 + lid;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;
  GLOBAL_MEM char *tmp;

% for arg in arguments:
  % if arg.isarray():
  tmp = (GLOBAL_MEM char *)${arg.name}; tmp += ${arg.name}_offset;
  ${arg.name} = (${arg.decltype()})tmp;
  % endif
% endfor

  ${out_arg.ctype()} acc = ${neutral};
  for (i = idx; i < n; i += numThreads) {
    acc = REDUCE(acc, ${expression});
  }
  ldata[lid] = acc;

  <% cur_size = local_size %>
  % while cur_size > 1:
    <% cur_size = cur_size / 2 %>
    local_barrier();
    if (lid < ${cur_size}) {
      ldata[lid] = REDUCE(ldata[lid], ldata[lid+${cur_size}]);
    }
  % endwhile
  if (lid == 0) out[GID_0] = ldata[0];
}
""")

stage2_kernel = Template("""
${preamble}

#define REDUCE(a, b) (${reduce_expr})

KERNEL void ${name}(const unsigned int n, ${out_arg.decltype()} out,
                    ${out_arg.decltype()} in) {
  LOCAL_MEM ${out_arg.ctype()} ldata[${local_size}];
  const unsigned int lid = LID_0;
  const unsigned int idx = LDIM_0 * GID_0 + lid;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;

  ${out_arg.ctype()} acc = ${neutral};
  for (i = idx; i < n; i += numThreads) {
    acc = REDUCE(acc, in[i]);
  }
  ldata[lid] = acc;

  <% cur_size = local_size %>
  % while cur_size > 1:
    <% cur_size = cur_size / 2 %>
    local_barrier();
    if (lid < ${cur_size}) {
      ldata[lid] = REDUCE(ldata[lid], ldata[lid+${cur_size}]);
    }
  % endwhile
  if (lid == 0) out[GID_0] = ldata[0];
}
""")

class ReductionKernel(object):
    def __init__(self, kind, context, dtype_out, neutral, reduce_expr, map_expr=None, arguments=None, preamble=""):
        self.kind = kind
        self.context = context
        self.neutral = neutral
        self.dtype_out = dtype_out
        self.out_arg = ArrayArg(numpy.dtype(self.dtype_out), 'out')

        if isinstance(arguments, str):
            self.arguments = parse_c_args(arguments)
        elif arguments is None:
            self.arguments = [ArrayArg(numpy.dtype(self.dtype_out), '_reduce_input')]
        else:
            self.arguments = arguments

        self.reduce_expr = reduce_expr
        if map_expr is None:
            if len(self.arguments) != 1:
                raise ValueError("Don't know what to do with more than one "
                                 "argument. Specify map_expr to explicitly "
                                 "state what you want.")
            self.operation = "%s[i]" % (self.arguments[0].name,)
            self.expresssion = "%s[0]" % (self.arguments[0].name,)
        else:
            self.operation = map_expr
            self.expression = massage_op(map_expr)

        if not any(isinstance(arg, ArrayArg) for arg in self.arguments):
            raise ValueError("ReductionKernel can only be used with "
                             "functions that have at least one vector "
                             "argument.")
        
        have_small = False
        have_double = False
        have_complex = False
        for arg in self.arguments:
            if arg.dtype.itemsize < 4 and type(arg) == ArrayArg:
                have_small = True
            if arg.dtype in [numpy.float64, numpy.complex128]:
                have_double = True
            if arg.dtype in [numpy.complex64, numpy.complex128]:
                have_complex = True

        self.flags = dict(have_small=have_small, have_double=have_double,
                          have_complex=have_complex)
        self.preamble = preamble

        local_size = min(int(gpuarray.get_lmemsize(kind, context) /
                             self.out_arg.dtype.itemsize),
                         gpuarray.get_maxlsize(kind, context))
        loop_count = 0
        while True:
            src = contig_kernel.render(preamble=self.preamble,
                                       name="reduk",
                                       reduce_expr=self.reduce_expr,
                                       out_arg=self.out_arg,
                                       arguments=self.arguments,
                                       local_size=local_size,
                                       neutral=self.neutral,
                                       expression=self.operation)
            try:
                k = gpuarray.GpuKernel(src, "reduk", kind=self.kind,
                                   context=self.context,
                                   cluda=True, **self.flags)
            except gpuarray.GpuArrayException:
                print "Failed kernel was:"
                print src
                raise
            if local_size <= k.maxlsize:
                self.contig_k = k
                self.contig_ls = local_size
                break
            local_size = k.maxlsize
            
            loop_count += 1
            if loop_count > 2:
                raise RuntimeError("Can't stabilize the local_size for kernel."
                                   " Please report this along with your "
                                   "reduction code.")

        src = stage2_kernel.render(preamble=self.preamble,
                                   name="reduk_2",
                                   reduce_expr=self.reduce_expr,
                                   out_arg=self.out_arg,
                                   local_size=local_size,
                                   neutral=self.neutral)
        self.stage2_k = gpuarray.GpuKernel(src, "reduk_2", kind=self.kind,
                                           context=self.context, cluda=True,
                                           **self.flags)
        self.stage2_ls = local_size
        if self.stage2_k.maxlsize < local_size:
            raise RuntimeError("Stage 2 kernel will not run. Please "
                               "report this along with your reduction code.")
    def _get_gs(self, n, ls, k):
        np = gpuarray.get_numprocs(self.kind, self.context)

        # special cases for OpenCL on CPU where the max local size is 1
        if n == np:
            return 1
        if ls == 1:
            return min(np, k.maxgsize)

        # Run enough threads to fully occupy the device but not so much
        # that it will take a large number of stage2 calls
        gs = np * 8

        # But don't run a bunch of useless threads
        gs = min(int(((n-1)/ls)+1), gs)

        # And don't go over the maximum
        return min(gs, k.maxgsize)

    def _alloc_out(self, gs):
        if gs == 1:
            out = gpuarray.empty((), kind=self.kind,
                                 context=self.context, dtype=self.dtype_out)
        else:
            out = gpuarray.empty((gs,), kind=self.kind,
                                 context=self.context, dtype=self.dtype_out)
        return out
        
    def call_stage2(self, n, inp):
        while n > 1:
            gs = self._get_gs(n, self.stage2_ls, self.stage2_k)
            out = self._alloc_out(gs)
            self.stage2_k(numpy.asarray(n, dtype='uint32'), out, inp,
                          ls=self.stage2_ls, gs=gs)
            n = gs
            inp = out
        return inp

    def _call_contig(self, n, args, offsets):
        gs = self._get_gs(n, self.contig_ls, self.contig_k)
        out = self._alloc_out(gs)
        kernel_args = [numpy.asarray(n, dtype='uint32'), out]
        for i, arg in enumerate(args):
            kernel_args.append(arg)
            if isinstance(arg, gpuarray.GpuArray):
                kernel_args.append(numpy.asarray(offsets[i], dtype='uint32'))
        self.contig_k(*kernel_args, ls=self.contig_ls, gs=gs)
        return gs, out

    def call_stage1(self, args):
        n, nd, dims, strs, offsets, contig = check_args(args)
        if not contig:
            raise NotImplementedError("not contig")
        return self._call_contig(n, args, offsets)

    def __call__(self, *args):
        return self.call_stage2(*self.call_stage1(args))
