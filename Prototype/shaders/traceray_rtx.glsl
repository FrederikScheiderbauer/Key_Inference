/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//-------------------------------------------------------------------------------------------------
// This file has the RTX functions for Closest-Hit and Any-Hit shader.
// The Ray Query pipeline implementation of thoses functions are in traceray_rq.
// This is used in pathtrace.glsl (Ray-Generation shader)

#ifndef TRACERAY_RTX_GLSL
#define TRACERAY_RTX_GLSL

#include "keyCreation.glsl"
#include "random.glsl"
//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void ClosestHit(Ray r,int  depth)
{
  uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT;
  prd.hitT      = INFINITY;
  uint64_t start; 
  uint64_t end; 
  int ID = int(gl_LaunchIDEXT.y) * int(gl_LaunchSizeEXT.x) + int(gl_LaunchIDEXT.x);

  if(SORTING_MODE == eNoSorting)
  {
    start = clockRealtimeEXT(); 
    traceRayEXT(topLevelAS,   // acceleration structure
                rayFlags,     // rayFlags
                0xFF,         // cullMask
                0,            // sbtRecordOffset
                0,            // sbtRecordStride
                0,            // missIndex
                r.origin,     // ray origin
                0.0,          // ray min range
                r.direction,  // ray direction
                INFINITY,     // ray max range
                0             // payload (location = 0)
    );
    end = clockRealtimeEXT();
    uint64_t duration = end -start;
    prd.rtTiming += duration;
  }
  else{

    uint64_t sortingStart; 
    uint64_t sortingEnd; 

    if(SORTING_MODE != eHitObject && SORTING_MODE != eTwoPoint)
    {
      sortingStart = clockRealtimeEXT();
      uint code = createSortingKey(SORTING_MODE,prd,r);

      reorderThreadNV(code,NUM_COHERENCE_BITS);
      sortingEnd = clockRealtimeEXT();
    }
    start = clockRealtimeEXT(); 
    hitObjectNV hObj;
    hitObjectRecordEmptyNV(hObj); //Initialize to an empty hit object
    hitObjectTraceRayNV(hObj,
                topLevelAS,
                rayFlags,
                0xFF,
                0,
                0,
                0,
                r.origin,
                0.0,
                r.direction,
                INFINITY,
                0);
    end = clockRealtimeEXT();
    uint64_t duration = end -start;
    prd.rtTiming += duration;

    if(SORTING_MODE == eHitObject)
    {
      sortingStart = clockRealtimeEXT();
      reorderThreadNV(hObj);
      sortingEnd = clockRealtimeEXT();
    }
    if(SORTING_MODE == eTwoPoint)
    {
      sortingStart = clockRealtimeEXT();
      uint code = createSortingKey(SORTING_MODE,prd,r);
      reorderThreadNV(code,NUM_COHERENCE_BITS);
      sortingEnd = clockRealtimeEXT();

    }
    hitObjectExecuteShaderNV(hObj, 0);
    uint64_t sortingDuration = sortingEnd -sortingStart;
    prd.sortTiming += sortingDuration;

  } 
}

//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
// use random sortingMode
//
void ClosestHitRandom(Ray r, uint s_Mode)
{

  uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT;
  prd.hitT      = INFINITY;
  uint64_t start; 
  uint64_t end; 
  int ID = int(gl_LaunchIDEXT.y) * int(gl_LaunchSizeEXT.x) + int(gl_LaunchIDEXT.x);



  if(s_Mode == eNoSorting)
  {
    start = clockRealtimeEXT(); 
    traceRayEXT(topLevelAS,   // acceleration structure
                rayFlags,     // rayFlags
                0xFF,         // cullMask
                0,            // sbtRecordOffset
                0,            // sbtRecordStride
                0,            // missIndex
                r.origin,     // ray origin
                0.0,          // ray min range
                r.direction,  // ray direction
                INFINITY,     // ray max range
                0             // payload (location = 0)
    );
    end = clockRealtimeEXT();
    uint64_t duration = end -start;
    prd.rtTiming += duration;
  }
  else{

    uint64_t sortingStart; 
    uint64_t sortingEnd; 

    if(s_Mode != eHitObject && s_Mode != eTwoPoint)
    {
      sortingStart = clockRealtimeEXT();
      uint code = createSortingKey(s_Mode,prd,r);
      reorderThreadNV(code,NUM_COHERENCE_BITS);
      sortingEnd = clockRealtimeEXT();
    }
    start = clockRealtimeEXT(); 
    hitObjectNV hObj;
    hitObjectRecordEmptyNV(hObj); //Initialize to an empty hit object
    hitObjectTraceRayNV(hObj,
                topLevelAS,
                rayFlags,
                0xFF,
                0,
                0,
                0,
                r.origin,
                0.0,
                r.direction,
                INFINITY,
                0);
    end = clockRealtimeEXT();
    uint64_t duration = end -start;
    prd.rtTiming += duration;

    if(s_Mode == eHitObject)
    {
      sortingStart = clockRealtimeEXT();
      reorderThreadNV(hObj);
      sortingEnd = clockRealtimeEXT();
    }
    if(s_Mode == eTwoPoint)
    {
      sortingStart = clockRealtimeEXT();
      uint code = createSortingKey(s_Mode,prd,r);
      reorderThreadNV(code,NUM_COHERENCE_BITS);
      sortingEnd = clockRealtimeEXT();

    }
    hitObjectExecuteShaderNV(hObj, 0);
    uint64_t sortingDuration = sortingEnd -sortingStart;
    prd.sortTiming += sortingDuration;

  } 
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool AnyHit(Ray r, float maxDist)
{
  shadow_payload.isHit = true;      // Asume hit, will be set to false if hit nothing (miss shader)
  shadow_payload.seed  = prd.seed;  // don't care for the update - but won't affect the rahit shader
  uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;

  traceRayEXT(topLevelAS,   // acceleration structure
              rayFlags,     // rayFlags
              0xFF,         // cullMask
              0,            // sbtRecordOffset
              0,            // sbtRecordStride
              1,            // missIndex
              r.origin,     // ray origin
              0.0,          // ray min range
              r.direction,  // ray direction
              maxDist,      // ray max range
              1             // payload layout(location = 1)
  );

  // add to ray contribution from next event estimation
  return shadow_payload.isHit;
}


#endif