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


/*
 *  Implement the RTX ray tracing pipeline
 */



#include <thread>

#include "nvh/alignment.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"
#include "rtx_pipeline.hpp"
#include "scene.hpp"
#include "tools.hpp"
#include "sorting_grid.hpp"

// Shaders
#include "autogen/pathtrace.rahit.h"
#include "autogen/pathtrace.rchit.h"
#include "autogen/pathtrace.rgen.h"
#include "autogen/pathtrace.rmiss.h"
#include "autogen/pathtraceShadow.rmiss.h"

 	

// basic file operations
#include <iostream>
#include <fstream>

//Macros
#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))

void task()
{

  while(true)
  {
  std::cout << "task1 says: " << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
// Typical resource holder + query for capabilities
//
void RtxPipeline::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_pAlloc     = allocator;
  m_queueIndex = familyIndex;
  m_debug.setup(device);
  setupGLSLCompiler();

  // Requesting ray tracing properties
  //VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  //VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  properties.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(physicalDevice, &properties);

  createPipelineCache();

  m_sbtWrapper.setup(device, familyIndex, allocator, m_rtProperties);
  m_sbtWrapper_async.setup(device, familyIndex, allocator, m_rtProperties);

  wrappers[0].setup(m_device, m_queueIndex, m_pAlloc, m_rtProperties);
  wrappers[1].setup(m_device, m_queueIndex, m_pAlloc, m_rtProperties);


  
}

//--------------------------------------------------------------------------------------------------
// Destroy all allocated resources
//

void RtxPipeline::destroy()
{

  destroyAsyncPipelineBuffer();
  m_sbtWrapper.destroy();

  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  
  m_sbtWrapper_async.destroy();

  for(int i = 0; i < NUM_PIPELINES_IN_BUFFER;i++)
  { 
    vkDestroyPipeline(m_device,pipelines[i],nullptr);
    wrappers[i].destroy();
  }
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  
  vkDestroyPipeline(m_device, m_rtPipeline_async, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout_async, nullptr);
  vkDestroyPipelineCache(m_device,m_PipelineCache,nullptr);
  m_PipelineCache = VK_NULL_HANDLE;

  m_rtPipelineLayout = VkPipelineLayout();
  m_rtPipeline       = VkPipeline();
  for(int i = 0; i < NUM_PIPELINES_IN_BUFFER;i++)
  { 
    pipelines[i] = VkPipeline();
  }
  requiresNewPipeline = true;
  m_rtPipelineLayout_async = VkPipelineLayout();
  m_rtPipeline_async       = VkPipeline();
  //results = std::vector<shaderc::SpvCompilationResult>();
}

//--------------------------------------------------------------------------------------------------
// Creation of the pipeline and layout
//
void RtxPipeline::create(const VkExtent2D& size, const std::vector<VkDescriptorSetLayout>& rtDescSetLayouts, Scene* scene)
{
  if(m_PipelineCache == VK_NULL_HANDLE)
  {createPipelineCache();}

  MilliTimer timer;
  LOGI("Create RtxPipeline");


    
  createPipelineLayout(rtDescSetLayouts,m_rtPipelineLayout);

  activeElement = createPipeline(m_SERParameters);
  activeElement.parameters = m_SERParameters;

    
    
  timer.print();
}




//--------------------------------------------------------------------------------------------------
// The layout has a push constant and the incoming descriptors are:
// acceleration structure, offscreen image, scene data, hdr
//
void RtxPipeline::createPipelineLayout(const std::vector<VkDescriptorSetLayout>& rtDescSetLayouts, VkPipelineLayout& pipelineLayout)
{
  vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);

  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                   0, sizeof(RtxState)};

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;
  pipelineLayoutCreateInfo.setLayoutCount         = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts            = rtDescSetLayouts.data();
  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
PipelineStorage RtxPipeline::createPipeline(SortingParameters parameters)
{

  SBTWrapper newWrapper;
  newWrapper.setup(m_device,m_queueIndex,m_pAlloc,m_rtProperties);

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eAnyHit,
    eShaderGroupCount
  };


  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point

  // Raygen
  //shaderc::SpvCompilationResult compresult = CompileShader("pathtrace.rgen",shaderc_raygen_shader);
  //result3 = CompileShader("pathtrace.rgen",shaderc_raygen_shader);

  int hashCode = hashParameters(parameters);

  bool foundOne = false;

  VkShaderModule module;
  if(!madeOne)
  {
    result3 = CompileShader("pathtrace.rgen",shaderc_raygen_shader);
    hitshader = CompileShader("pathtrace.rchit",shaderc_closesthit_shader);
    madeOne = true;
    printf("made a new one\n");
  }
  module = glslCompiler.createModule(m_device,result3);
  

  //shaderc::SpvCompilationResult compresult2 = CompileShader("pathtrace.rgen",shaderc_raygen_shader);
  //module = glslCompiler.createModule(m_device,results[0]);
  //module = nvvk::createShaderModule(m_device, rgen.data(), sizeof(rgen));
  stage.module    = module;
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;


  //add specializations
  nvvk::Specialization specialization;
  for(int i= 0; i < hashedParameterizations.size(); i++)
  {
    if(hashedParameterizations[i]==hashCode)
    {
      specialization = storedSpecializations[i];
      stage.module    = module;
      foundOne = true;
      break;
    }
  }
  if(!foundOne)
  {
    specialization.add(0,m_sortingMode);
    specialization.add(1,m_enableProfiling);
    //Add Sorting parameters as specialization constants
    specialization.add(2,parameters.noSort); //No Sorting
    specialization.add(3,parameters.hitObject); //HitObject
    specialization.add(4,parameters.rayOrigin); //RayOrigin
    specialization.add(5,parameters.rayDirection); //RayDirection
    specialization.add(6,parameters.estimatedEndpoint); //EstEndPoint
    specialization.add(7,parameters.realEndpoint); //RealEndpoint
    specialization.add(8,parameters.sortAfterASTraversal); //AfterASTraversal
    specialization.add(9,parameters.isFinished); //isFinished

    storedSpecializations.emplace_back(specialization);
    hashedParameterizations.emplace_back(hashCode);
  }
  
  stages[eRaygen].pSpecializationInfo = specialization.getSpecialization();
  


  // Miss
  stage.module  = CompileAndCreateShaderModule("pathtrace.rmiss",shaderc_miss_shader);
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;

  // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
  stage.module   = CompileAndCreateShaderModule("pathtraceShadow.rmiss",shaderc_miss_shader);
  stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss2] = stage;

  // Hit Group - Closest Hit
  stage.module        =glslCompiler.createModule(m_device,hitshader);// CompileAndCreateShaderModule("pathtrace.rchit",shaderc_closesthit_shader);
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;

  // Hit Group - Any Hit
  stage.module    = CompileAndCreateShaderModule("pathtrace.rahit",shaderc_anyhit_shader);
  stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  stages[eAnyHit] = stage;


  // Shader groups
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
  VkRayTracingShaderGroupCreateInfoKHR              group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  groups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  groups.push_back(group);

  // Shadow Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss2;
  groups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  if(m_enableAnyhit)
    group.anyHitShader = eAnyHit;
  groups.push_back(group);

  // --- Pipeline ---
  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  m_createInfo.stageCount = static_cast<uint32_t>(stages.size());  // Stages are shaders
  m_createInfo.pStages    = stages.data();

  m_createInfo.groupCount = static_cast<uint32_t>(groups.size());  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  m_createInfo.pGroups    = groups.data();

  m_createInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  m_createInfo.layout                       = m_rtPipelineLayout;

  // Create a deferred operation (compiling in parallel)
  bool                   useDeferred{true};
  VkResult               result;
  VkDeferredOperationKHR deferredOp{VK_NULL_HANDLE};
  if(useDeferred)
  {
    result = vkCreateDeferredOperationKHR(m_device, nullptr, &deferredOp);
    assert(result == VK_SUCCESS);
  }
  //vkCreateRayTracingPipelinesKHR(m_device, deferredOp,m_PipelineCache, 1, &m_createInfo, nullptr, &pipeline);
  VkPipeline newPipeline{VK_NULL_HANDLE};
  vkCreateRayTracingPipelinesKHR(m_device, deferredOp,m_PipelineCache, 1, &m_createInfo, nullptr, &newPipeline);


  if(useDeferred)
  {
    // Query the maximum amount of concurrency and clamp to the desired maximum
    uint32_t maxThreads{8};
    uint32_t numLaunches = std::min(vkGetDeferredOperationMaxConcurrencyKHR(m_device, deferredOp), maxThreads);

    std::vector<std::future<void>> joins;
    for(uint32_t i = 0; i < numLaunches; i++)
    {
      VkDevice device{m_device};
      joins.emplace_back(std::async(std::launch::async, [device, deferredOp]() {
        // A return of VK_THREAD_IDLE_KHR should queue another job
        vkDeferredOperationJoinKHR(device, deferredOp);
      }));
    }

    for(auto& f : joins)
    {
      f.get();
    }

    // deferred operation is now complete.  'result' indicates success or failure
    result = vkGetDeferredOperationResultKHR(m_device, deferredOp);
    assert(result == VK_SUCCESS);
    vkDestroyDeferredOperationKHR(m_device, deferredOp, nullptr);
  }
  //storedPipelines.emplace_back(newPipeline);

  newWrapper.create(newPipeline,m_createInfo);
  //wrappers[0].create(pipelines[activePipeline],m_createInfo);

  PipelineStorage newStorageElement;
  newStorageElement.pipeline = newPipeline;
  newStorageElement.sbt = newWrapper;
  storage.emplace_back(newStorageElement);


  //storedSBTs.emplace_back(newWrapper);
  // --- Clean up ---
  for(auto& s : stages)
    vkDestroyShaderModule(m_device, s.module, nullptr);

  return newStorageElement;
}


//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void RtxPipeline::run(const VkCommandBuffer& cmdBuf, const VkExtent2D& size, nvvk::ProfilerVK& profiler, const std::vector<VkDescriptorSet>& descSets)
{
  LABEL_SCOPE_VK(cmdBuf);


  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, activeElement.pipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                     0, sizeof(RtxState), &m_state);


  auto& regions = activeElement.sbt.getRegions();
  
  vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], size.width, size.height, 1);
}

//--------------------------------------------------------------------------------------------------
// Toggle the usage of Anyhit. Not having anyhit can be faster, but the scene must but fully opaque
//
void RtxPipeline::useAnyHit(bool enable)
{
  m_enableAnyhit = enable;
  createPipeline(m_SERParameters);
}


void RtxPipeline::setupGLSLCompiler()
{
  std::vector<std::string> defaultSearchPaths;
  std::string shaderPath = "C:/Users/Frederik/Key_Inference/Prototype/shaders";
  defaultSearchPaths.push_back(shaderPath);
  nvvkhl::GlslIncluder glslIncluder(defaultSearchPaths);
  for (std::string path : defaultSearchPaths)
  {
    glslCompiler.addInclude(path);
  }
  glslCompiler.options()->SetTargetSpirv(shaderc_spirv_version_1_4);
  glslCompiler.options()->SetTargetEnvironment(shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_3);

}


shaderc::SpvCompilationResult RtxPipeline::CompileShader(std::string filename, shaderc_shader_kind shadertype)
{
  shaderc::SpvCompilationResult compResult = glslCompiler.compileFile(filename,shadertype);
  auto compilationSize = (compResult.end() - compResult.begin()) * sizeof(uint32_t);
  const uint32_t* pCode =  reinterpret_cast<const uint32_t*>(compResult.begin());
  
  rgen.clear();
  for(int i = 0; i < (compResult.end() - compResult.begin());i++)
  {
    rgen.emplace_back(pCode[i]);
  }


  //rgen.assign(pCode,pCode+compilationSize);
  std::ofstream myfile;
  myfile.open("example.txt");
  myfile.write(reinterpret_cast<const char*>(pCode),compilationSize);
  myfile.close();
  return compResult;
}

VkShaderModule RtxPipeline::CompileAndCreateShaderModule(std::string filename, shaderc_shader_kind shadertype)
{

  shaderc::SpvCompilationResult compResult = glslCompiler.compileFile(filename,shadertype);

  VkShaderModule resultModule = glslCompiler.createModule(m_device,compResult);

  return resultModule;
}


void RtxPipeline::setSortingMode(int index)
{
  m_sortingMode = index;
  createPipeline(m_SERParameters);
}

void RtxPipeline::enableProfiling(bool enable)
{
  m_enableProfiling = enable;
  createPipeline(m_SERParameters);
}


void RtxPipeline::setPipeline(int index)
{
  m_rtPipeline = m_cachedRtPipelines[index];
}

int RtxPipeline::hashParameters(SortingParameters parameters)
{
  int result = 0;
  if(parameters.noSort)
  {
    result |= 1;
    return result;
  }

  result |= parameters.sortAfterASTraversal ? 2: 0;
  result |= parameters.hitObject ? 4: 0;
  result |= parameters.rayOrigin ? 8: 0;
  result |= parameters.rayDirection ? 16: 0;
  result |= parameters.estimatedEndpoint ? 32: 0;
  result |= parameters.realEndpoint ? 64: 0;
  result |= parameters.isFinished ? 128: 0;

  return result;
}

SortingParameters RtxPipeline::rebuildFromhash(int hashCode)
{
  SortingParameters result;
  result.numCoherenceBitsTotal = 32;
  result.noSort = CHECK_BIT(hashCode,0);
  result.sortAfterASTraversal = CHECK_BIT(hashCode,1);
  result.hitObject = CHECK_BIT(hashCode,2);
  result.rayOrigin = CHECK_BIT(hashCode,3);
  result.rayDirection = CHECK_BIT(hashCode,4);
  result.estimatedEndpoint = CHECK_BIT(hashCode,5);
  result.realEndpoint = CHECK_BIT(hashCode,6);
  result.isFinished = CHECK_BIT(hashCode,7);

  return result;
}

shaderc::SpvCompilationResult* RtxPipeline::getRayGenShaderObject()
{
shaderc::SpvCompilationResult* result;

int hashCode = hashParameters(m_SERParameters);

if(raygenShaders.size() > 0)
{

  for (int i =0; i < raygenShaders.size();i++)
  {
    if(hashCode == raygenShaders[i].hashcode)
    {
      result = raygenShaders[i].compResult;
      return result;
    }
  }
}
RtxPipeline::ShaderObject newObject;
newObject.hashcode = hashCode;
shaderc::SpvCompilationResult compResult = CompileShader("pathtrace.rgen",shaderc_raygen_shader);
newObject.compResult = &compResult;
raygenShaders.emplace_back(newObject);
return newObject.compResult;
}


void RtxPipeline::createPipelineCache()
{

  VkPipelineCacheCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  createInfo.pNext = nullptr;
  createInfo.initialDataSize = 0;
  createInfo.flags = 0;

  VkResult result = vkCreatePipelineCache(m_device,&createInfo,nullptr,&m_PipelineCache);
  if(result == VK_SUCCESS)
  {
    printf("Success\n");
  }

}

void RtxPipeline::fillPipelineBuffer()
{
 
 if(PrebuildPipelineBuffer.size()<5)
 {
  MilliTimer timer;
  LOGI("Create RtxPipeline:");
  //create the new parameters
  //SortingParameters newSortingParameters = morphSortingParameters(mostRecentParameters);
  SortingParameters newSortingParameters = createSortingParameters1();
  mostRecentParameters = newSortingParameters;
  //create new pipeline
  PipelineStorage newElement = createPipeline(newSortingParameters);
  newElement.parameters = newSortingParameters;
  PrebuildPipelineBuffer.emplace_back(newElement);    
  timer.print();
 }

 

}


void RtxPipeline::setNewPipeline()
{

  activeElement = PrebuildPipelineBuffer[0];
  m_SERParameters = activeElement.parameters;
  PrebuildPipelineBuffer.erase(PrebuildPipelineBuffer.begin());

}


void RtxPipeline::setNewPipeline(PipelineStorage newPipelineElement)
{
  activeElement = newPipelineElement;
  m_SERParameters = activeElement.parameters;
  //PrebuildPipelineBuffer.erase(PrebuildPipelineBuffer.begin());
}
void RtxPipeline::activateAsyncPipelineCreation()
{
    std::thread([&,this]() 
    {
    while(useAsyncPipelineCreation)
    {
      if(m_rtPipelineLayout == VK_NULL_HANDLE)
      {
        //printf("still null\n");
      }
      else {
        fillPipelineBuffer();
      }
    }
  }).detach();
  

  //std::thread t1(buildStuff,m_rtPipelineLayout);

}

void RtxPipeline::destroyAsyncPipelineBuffer()
{
  
  for(PipelineStorage element : storage)
  {
    vkDestroyPipeline(m_device,element.pipeline,nullptr);
    element.sbt.destroy();
  }

  for(AsyncPipeline asyncPipeline : asyncPipelineBuffer)
  {
    vkDestroyPipeline(m_device,asyncPipeline.pipeline,nullptr);
  }
  asyncPipelineBuffer = std::vector<AsyncPipeline>();

storage = std::vector<PipelineStorage>();
  //pipelineCreateInfoBuffer = std::vector<VkRayTracingPipelineCreateInfoKHR>();
}