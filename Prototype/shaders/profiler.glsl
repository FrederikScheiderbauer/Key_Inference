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

    prd.traceTiming = 0;
}

void accumulateSamplesMax(uint sortMode,uint64_t duration)
{
  if(sortMode == eNoSorting)
  {
    atomicMax(timeData.noSortTime,duration);
    atomicAdd(timeData.noSortThreads,1);
  }
  if(sortMode == eHitObject)
  {
    atomicMax(timeData.hitObjectTime,duration);
    atomicAdd(timeData.hitObjectThreads,1);
  }
  if(sortMode == eOrigin)
  {
    atomicMax(timeData.originTime,duration);
    atomicAdd(timeData.originThreads,1);
  }
  if(sortMode == eReis)
  {
    atomicMax(timeData.reisTime,duration);
    atomicAdd(timeData.reisThreads,1);
  }
  if(sortMode == eCosta)
  {
    atomicMax(timeData.costaTime,duration);
    atomicAdd(timeData.costaThreads,1);
  }
  if(sortMode == eAila)
  {
    atomicMax(timeData.ailaTime,duration);
    atomicAdd(timeData.ailaThreads,1);
  }
  if(sortMode == eTwoPoint)
  {
    atomicMax(timeData.twoPointTime,duration);
    atomicAdd(timeData.twoPointThreads,1);
  }
  if(sortMode == eEndPointEst)
  {
    atomicMax(timeData.endPointEstTime,duration);
    atomicAdd(timeData.endPointEstThreads,1);
  }
  if(sortMode == eEndEstAdaptive)
  {
    atomicMax(timeData.endEstAdaptiveTime,duration);
    atomicAdd(timeData.endEstAdaptiveThreads,1);
  }
}

void accumulateSamplesAdd(uint sortMode,uint64_t duration)
{
  if(sortMode == eNoSorting)
  {
    atomicAdd(timeData.noSortTime,duration);
    atomicAdd(timeData.noSortThreads,1);
  }
  if(sortMode == eHitObject)
  {
    atomicAdd(timeData.hitObjectTime,duration);
    atomicAdd(timeData.hitObjectThreads,1);
  }
  if(sortMode == eOrigin)
  {
    atomicAdd(timeData.originTime,duration);
    atomicAdd(timeData.originThreads,1);
  }
  if(sortMode == eReis)
  {
    atomicAdd(timeData.reisTime,duration);
    atomicAdd(timeData.reisThreads,1);
  }
  if(sortMode == eCosta)
  {
    atomicAdd(timeData.costaTime,duration);
    atomicAdd(timeData.costaThreads,1);
  }
  if(sortMode == eAila)
  {
    atomicAdd(timeData.ailaTime,duration);
    atomicAdd(timeData.ailaThreads,1);
  }
  if(sortMode == eTwoPoint)
  {
    atomicAdd(timeData.twoPointTime,duration);
    atomicAdd(timeData.twoPointThreads,1);
  }
  if(sortMode == eEndPointEst)
  {
    atomicAdd(timeData.endPointEstTime,duration);
    atomicAdd(timeData.endPointEstThreads,1);
  }
  if(sortMode == eEndEstAdaptive)
  {
    atomicAdd(timeData.endEstAdaptiveTime,duration);
    atomicAdd(timeData.endEstAdaptiveThreads,1);
  }
}


void resetTimings(int ID)
{
  if(ID == 0)
      {
        atomicMin(timeData.noSortTime,0);
        atomicMin(timeData.noSortThreads,0);
        atomicMin(timeData.full_time,0);
        atomicMin(timeData.full_time_threads,0);
      }
      if(ID == 1)
      {
        atomicMin(timeData.hitObjectTime,0);
        atomicMin(timeData.hitObjectThreads,0);
      }
      if(ID == 2)
      {
        atomicMin(timeData.originTime,0);
        atomicMin(timeData.originThreads,0);
      } 
      if(ID == 3)
      {
        atomicMin(timeData.reisTime,0);
        atomicMin(timeData.reisThreads,0);
      }
      if(ID == 4)
      {
        atomicMin(timeData.costaTime,0);
        atomicMin(timeData.costaThreads,0);
      }
      if(ID == 5)
      {
        atomicMin(timeData.ailaTime,0);
        atomicMin(timeData.ailaThreads,0);
      }
      if(ID == 6)
      {
        atomicMin(timeData.twoPointTime,0);
        atomicMin(timeData.twoPointThreads,0);
      }
      if(ID == 7)
      {
        atomicMin(timeData.endPointEstTime,0);
        atomicMin(timeData.endPointEstThreads,0);
      } 
      if(ID == 8)
      {
        atomicMin(timeData.endEstAdaptiveTime,0);
        atomicMin(timeData.endEstAdaptiveThreads,0);
      }

}

#endif