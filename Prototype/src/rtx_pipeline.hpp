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


#pragma once

#include <future>

#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvvk/specialization.hpp"

#include "renderer.h"
#include "shaders/host_device.h"
#include "nvvkhl/glsl_compiler.hpp"

using nvvk::SBTWrapper;

const int NUM_PIPELINES_IN_BUFFER = 2;

  struct PipelineStorage
  {
    VkPipeline pipeline;
    SBTWrapper sbt;
    SortingParameters parameters;
  };
/*

Creating the RtCore renderer 
* Requiring:  
  - Acceleration structure (AccelSctruct / Tlas)
  - An image (Post StoreImage)
  - The glTF scene (vertex, index, materials, ... )

* Usage
  - setup as usual
  - create
  - run
*/
class RtxPipeline : public Renderer
{
public:
  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator) override;
  void destroy() override;
  void create(const VkExtent2D& size, const std::vector<VkDescriptorSetLayout>& rtDescSetLayouts, Scene* scene) override;
  void create_async(const VkExtent2D& size, const std::vector<VkDescriptorSetLayout>& rtDescSetLayouts, Scene* scene);
  void run(const VkCommandBuffer& cmdBuf, const VkExtent2D& size, nvvk::ProfilerVK& profiler, const std::vector<VkDescriptorSet>& descSets) override;
  void useAnyHit(bool enable);
  void setSortingMode(int sortingModeIndex);
  int* getSortingMode() {return &m_sortingMode;};
  int* getNumCoherenceBits() {return &m_numCoherenceBits;};
  void enableProfiling(bool enable);
  int hashParameters(SortingParameters parameters);
  SortingParameters rebuildFromhash(int hashCode);

  const std::string name() override { return std::string("Rtx"); }
  bool     m_enableProfiling{false};
  void setNewPipeline();
  void setNewPipeline(PipelineStorage newPipelineElement);
  std::vector<PipelineStorage> PrebuildPipelineBuffer;

  SortingParameters m_SERParameters{
    32,     //numCoherenceBitsTotal: 0-32 Zero meaning No sorting
    true,   //sortAfterASTraversal; when to sort->  0: before TraceRay; 1: after TraceRay
            //Which Information to encode into sortingKey:
    false,  //No Sorting
    true,   //hitObject
    false,  //ray origin
    false,  //ray direction
    false,  //estimated endpoint/ray length
    false,  //real endpoint/ray length; only known after AS Traversal
    false,  //whether or not the path is finished after this bounce
  };
std::vector<VkPipeline> m_cachedRtPipelines;
  void activateAsyncPipelineCreation();
  void destroyAsyncPipelineBuffer();
  bool useAsyncPipelineCreation = false;

  bool visualizeSortingGrid{false};
  float displayCubeSize{1.0};

void setPipeline(int index);
private:



  PipelineStorage createPipeline(SortingParameters parameters);
  void createPipeline_async();
  void createPipelineLayout(const std::vector<VkDescriptorSetLayout>& rtDescSetLayouts,VkPipelineLayout& pipelineLayout);
  void createPipelineLayout_async(const std::vector<VkDescriptorSetLayout>& rtDescSetLayouts);


  uint32_t m_nbHit{1};
  bool     m_enableAnyhit{true};
  int      m_sortingMode{0};
  int      m_numCoherenceBits{32};
  VkPipelineCache m_PipelineCache;
  void createPipelineCache();


private:
  // Setup
  nvvk::ResourceAllocator* m_pAlloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;   // Utility to name objects
  VkDevice                 m_device;
  uint32_t                 m_queueIndex{0};

  VkPhysicalDeviceProperties2                     properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPipelineLayout                                m_rtPipelineLayout{VK_NULL_HANDLE};
  VkPipeline                                      m_rtPipeline{VK_NULL_HANDLE};
  SBTWrapper                                      m_sbtWrapper;

  VkPipelineLayout                                m_rtPipelineLayout_async{VK_NULL_HANDLE};
  VkPipeline                                      m_rtPipeline_async{VK_NULL_HANDLE};
  SBTWrapper                                      m_sbtWrapper_async;

  struct ShaderObject
  {
    int hashcode; // represents parameterization of this shader Object
    shaderc::SpvCompilationResult *compResult;
  };
  std::vector<ShaderObject> raygenShaders;

  shaderc::SpvCompilationResult* result2;
  shaderc::SpvCompilationResult result3;

  shaderc::SpvCompilationResult hitshader;
  shaderc::SpvCompilationResult missshader;
  std::vector<shaderc::SpvCompilationResult> results;
  std::vector<nvvk::Specialization> storedSpecializations;
  std::vector<int> hashedParameterizations;
  bool madeOne = false;
  bool creatingPipeline = false;

  
private:
  //nvvkhl::GlslIncluder glslIncluder;
  nvvkhl::GlslCompiler glslCompiler;
  std::unique_ptr<shaderc::CompileOptions> glslCompileOptions;
  void setupGLSLCompiler();
  VkShaderModule CompileAndCreateShaderModule(std::string filename, shaderc_shader_kind shadertype);
  shaderc::SpvCompilationResult CompileShader(std::string filename, shaderc_shader_kind shadertype);
  shaderc::SpvCompilationResult *getRayGenShaderObject();

  std::vector<uint32_t> rgen;


  bool m_busy = false;
  bool requiresNewPipeline = true;
  

  std::future<void> asyncPipeline;

  std::vector<VkPipeline> pipelineBuffer;
  std::vector<VkRayTracingPipelineCreateInfoKHR> pipelineCreateInfoBuffer;

  struct AsyncPipeline
  {
    VkPipeline pipeline;
    VkRayTracingPipelineCreateInfoKHR createInfo;
    int hashCode;
  };

  std::vector<AsyncPipeline> asyncPipelineBuffer;

  void fillPipelineBuffer();
  void buildPipeline();
  
  

  VkRayTracingPipelineCreateInfoKHR* asyncPipelineCreateInfo;

  VkRayTracingPipelineCreateInfoKHR m_createInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  
  
  VkPipeline pipelines[NUM_PIPELINES_IN_BUFFER] = {{VK_NULL_HANDLE},{VK_NULL_HANDLE}};
  SBTWrapper wrappers[NUM_PIPELINES_IN_BUFFER];
  int activePipeline = 0;




  //std::vector<VkPipeline> storedPipelines;
  //std::vector<SBTWrapper> storedSBTs;
  std::vector<PipelineStorage> storage;


  


  PipelineStorage activeElement;
  
  };
