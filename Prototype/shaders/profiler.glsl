#ifndef PROFILER_GLSL
#define PROFILER_GLSL

#include "host_device.h"
#include "layouts.glsl"

/*
  ShadingTiming      shadeTiming;
  RayTraversalTiming rtTiming;
  SortingTiming      sortTiming;
*/
void setupProfiler()
{
    prd.shadeTiming = 0;

    prd.rtTiming = 0;

    prd.sortTiming = 0;
}

#endif