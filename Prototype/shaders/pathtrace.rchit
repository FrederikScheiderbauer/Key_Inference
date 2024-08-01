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
// The Closest-Hit shader only returns the information of the hit. The shading will be done in
// the Ray-Generation shader or Ray-Query (compute)

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require  // This is about ray tracing
#extension GL_KHR_shader_subgroup_basic : require       // Special extensions to debug groups, warps, SM, ...
#extension GL_EXT_scalar_block_layout : enable          // Align structure layout to scalar
#extension GL_EXT_nonuniform_qualifier : enable         // To access unsized descriptor arrays
#extension GL_ARB_shader_clock : enable                 // Using clockARB
#extension GL_EXT_shader_image_load_formatted : enable  // The folowing extension allow to pass images as function parameters
#extension GL_EXT_scalar_block_layout : enable          // Usage of 'scalar' block layout
#extension GL_ARB_gpu_shader_int64 : enable       // Debug - heatmap value
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "globals.glsl"
#include "host_device.h"
#include "layouts.glsl"
layout(location = 0) rayPayloadInEXT PtPayload prd;
layout(location = 1) rayPayloadEXT ShadowHitPayload shadow_payload;
hitAttributeEXT vec2 bary;
layout(push_constant) uniform _RtxState
{
  RtxState rtxState;
};
#include "sun_and_sky.glsl"



void main()
{
  //prd.seed;
  prd.hitT                = gl_HitTEXT;
  prd.primitiveID         = gl_PrimitiveID;
  prd.instanceID          = gl_InstanceID;
  prd.instanceCustomIndex = gl_InstanceCustomIndexEXT;
  prd.baryCoord           = bary;
  prd.objectToWorld       = gl_ObjectToWorldEXT;
  prd.worldToObject       = gl_WorldToObjectEXT;

  //PerformShading();

}


