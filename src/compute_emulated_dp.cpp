#include <clpeak.h>


int clPeak::runComputeEDP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  float timed, gflops;
  cl_uint workPerWI;
  cl::NDRange globalSize, localSize;
  cl_float2 A = {1.3f, 0.0f};
  int iters = devInfo.computeIters;

  if(!isComputeEDP)
    return 0;

  try
  {
    log->print(NEWLINE TAB TAB "Emulated double-precision compute (GFLOPS)" NEWLINE);
    log->xmlOpenTag("emulated_double_precision_compute");
    log->xmlAppendAttribs("unit", "gflops");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint globalWIs = (devInfo.numCUs) * (devInfo.computeWgsPerCU) * (devInfo.maxWGSize);
    uint t = MIN((globalWIs * sizeof(cl_double)), devInfo.maxAllocSize);
    t = roundToPowOf2(t);
    globalWIs = t / sizeof(cl_float2);
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_float2)));

    globalSize = globalWIs;
    localSize = devInfo.maxWGSize;

    cl::Kernel kernel_edp(prog, "compute_edp");
    kernel_edp.setArg(0, outputBuf), kernel_edp.setArg(1, A);

    ///////////////////////////////////////////////////////////////////////////
    log->print(TAB TAB TAB "emulated double   : ");

    workPerWI = 4096;      // Indicates flops executed per work-item

    timed = run_kernel(queue, kernel_edp, globalSize, localSize, iters);

    gflops = ((float)globalWIs * workPerWI) / timed / 1e3f;

    log->print(gflops);     log->print(NEWLINE);
    log->xmlRecord("emulated_double", gflops);
    ///////////////////////////////////////////////////////////////////////////
    log->xmlCloseTag();     // emulated_double_precision_compute
  }
  catch(cl::Error error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    return -1;
  }

  return 0;
}

