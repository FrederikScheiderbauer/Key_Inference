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
 * Main class to render the scene, holds sub-classes for various work
 */

#include <glm/glm.hpp>

#include <filesystem>
#include <thread>
#include <iostream>
#include <algorithm>
#include <limits>
#include <fstream>


#define VMA_IMPLEMENTATION

#include "shaders/host_device.h"
#include "rayquery.hpp"
#include "rtx_pipeline.hpp"
#include "sample_example.hpp"
#include "sample_gui.hpp"
#include "tools.hpp"

#include "sorting_grid.hpp"

#include "nvml_monitor.hpp"



#if defined(NVP_SUPPORTS_NVML)
NvmlMonitor g_nvml(100, 100);
#endif

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void SampleExample::setup(const VkInstance&               instance,
                          const VkDevice&                 device,
                          const VkPhysicalDevice&         physicalDevice,
                          const std::vector<nvvk::Queue>& queues)
{
  AppBaseVk::setup(instance, device, physicalDevice, queues[eGCT0].familyIndex);

  m_gui = std::make_shared<SampleGUI>(this);  // GUI of this class

  // Memory allocator for buffers and images
  m_alloc.init(instance, device, physicalDevice);
  m_staging.init(m_alloc.getMemoryAllocator());

  m_debug.setup(m_device);

  // Compute queues can be use for acceleration structures
  m_picker.setup(m_device, physicalDevice, queues[eCompute].familyIndex, &m_alloc);
  m_accelStruct.setup(m_device, physicalDevice, queues[eCompute].familyIndex, &m_alloc);

  // Note: the GTC family queue is used because the nvvk::cmdGenerateMipmaps uses vkCmdBlitImage and this
  // command requires graphic queue and not only transfer.
  m_scene.setup(m_device, physicalDevice, queues[eGCT1], &m_alloc);

  // Transfer queues can be use for the creation of the following assets
  m_offscreen.setup(m_device, physicalDevice, queues[eTransfer].familyIndex, &m_alloc);
  m_skydome.setup(device, physicalDevice, queues[eTransfer].familyIndex, &m_alloc);

  // Create and setup all renderers
  m_pRender[eRtxPipeline] = new RtxPipeline;
  m_pRender[eRayQuery]    = new RayQuery;
  for(auto r : m_pRender)
  {
    r->setup(m_device, physicalDevice, queues[eTransfer].familyIndex, &m_alloc);
  }

  std::vector<ProfilingStats> stats;
  for(int i = 0; i < eNumSortModes;i++)
  {
    profilingStats.push_back(stats);
  }

  
  //std::random_device dev;
  std::mt19937 rng2(dev());
  rng = rng2;

  createStorageBuffer();


  buildSortingGrid();
}


//--------------------------------------------------------------------------------------------------
// Loading the scene file, setting up all scene buffers, create the acceleration structures
// for the loaded models.
//
void SampleExample::loadScene(const std::string& filename)
{
  m_scene.load(filename);
  m_accelStruct.create(m_scene.getScene(), m_scene.getBuffers(Scene::eVertex), m_scene.getBuffers(Scene::eIndex));

  // The picker is the helper to return information from a ray hit under the mouse cursor
  m_picker.setTlas(m_accelStruct.getTlas());
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Loading an HDR image and creating the importance sampling acceleration structure
//
void SampleExample::loadEnvironmentHdr(const std::string& hdrFilename)
{
  MilliTimer timer;
  LOGI("Loading HDR and converting %s\n", hdrFilename.c_str());
  m_skydome.loadEnvironment(hdrFilename);
  timer.print();

  m_rtxState.fireflyClampThreshold = m_skydome.getIntegral() * 4.f;  // magic
}


//--------------------------------------------------------------------------------------------------
// Loading asset in a separate thread
// - Used by file drop and menu operation
// Marking the session as busy, to avoid calling rendering while loading assets
//
void SampleExample::loadAssets(const char* filename)
{
  std::string sfile = filename;

  // Need to stop current rendering
  m_busy = true;
  vkDeviceWaitIdle(m_device);

  std::thread([&, sfile]() {
    LOGI("Loading: %s\n", sfile.c_str());

    // Supporting only GLTF and HDR files
    namespace fs          = std::filesystem;
    std::string extension = fs::path(sfile).extension().string();
    if(extension == ".gltf" || extension == ".glb")
    {
      m_busyReasonText = "Loading scene ";

      // Loading scene and creating acceleration structure
      loadScene(sfile);

      // Loading the scene might have loaded new textures, which is changing the number of elements
      // in the DescriptorSetLayout. Therefore, the PipelineLayout will be out-of-date and need
      // to be re-created. If they are re-created, the pipeline also need to be re-created.
      for(auto& r : m_pRender)
        r->destroy();

      m_pRender[m_rndMethod]->create(
          m_size, {m_accelStruct.getDescLayout(), m_offscreen.getDescLayout(), m_scene.getDescLayout(), m_descSetLayout}, &m_scene);
    }

    if(extension == ".hdr")  //|| extension == ".exr")
    {
      m_busyReasonText = "Loading HDR ";
      loadEnvironmentHdr(sfile);
      updateHdrDescriptors();
    }


    // Re-starting the frame count to 0
    SampleExample::resetFrame();
    m_busy = false;
  }).detach();
}


//--------------------------------------------------------------------------------------------------
// Called at each frame to update the UBO: scene, camera, environment (sun&sky)
//
void SampleExample::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  if(m_busy)
    return;

  LABEL_SCOPE_VK(cmdBuf);
  const float aspectRatio = m_renderRegion.extent.width / static_cast<float>(m_renderRegion.extent.height);

  m_scene.updateCamera(cmdBuf, aspectRatio);
  vkCmdUpdateBuffer(cmdBuf, m_sunAndSkyBuffer.buffer, 0, sizeof(SunAndSky), &m_sunAndSky);
  vkCmdUpdateBuffer(cmdBuf, m_sortingParametersBuffer.buffer, 0, sizeof(SortingParameters), &(dynamic_cast<RtxPipeline*>(m_pRender[m_rndMethod])->m_SERParameters));
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame otherwise, increments frame.
//
void SampleExample::updateFrame()
{
  static glm::mat4 refCamMatrix;
  static float     fov = 0;

  auto& m = CameraManip.getMatrix();
  auto  f = CameraManip.getFov();
  if(refCamMatrix != m || f != fov)
  {
    resetFrame();
    refCamMatrix = m;
    fov          = f;
  }

  if(m_rtxState.frame < m_maxFrames)
    m_rtxState.frame++;
}

//--------------------------------------------------------------------------------------------------
// Reset frame is re-starting the rendering
//
void SampleExample::resetFrame()
{
  m_rtxState.frame = -1;
}

void SampleExample::createStorageBuffer()
{
    m_GridSortingKeyBuffer = m_alloc.createBuffer(sizeof(GridCube) * MAXGRIDSIZE *MAXGRIDSIZE *MAXGRIDSIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  NAME_VK(m_GridSortingKeyBuffer.buffer);
}
GridCube SampleExample::determineBestTimesCube(GridSpace* currentGrid)
{
  GridCube cube;

  float fastestTime = std::numeric_limits<float>::min();
  int fastestHash = 0;
  std::vector<TimingObject>* cubeSideTimingsUP = &getCubeSideElements(CubeSide::CubeUp,currentGrid)->storedElements;
  for(TimingObject timing : *cubeSideTimingsUP)
    {
    if(timing.fps > fastestTime)
      {
      fastestTime = timing.fps;
      fastestHash =timing.hashCode;
      }
  }
  cube.up = fastestHash;

  fastestTime = std::numeric_limits<float>::min();
  fastestHash = 0;
  std::vector<TimingObject>* cubeSideTimingsDOWN = &getCubeSideElements(CubeSide::CubeDown,currentGrid)->storedElements;
  for(TimingObject timing : *cubeSideTimingsDOWN)
  {
    if(timing.fps > fastestTime)
    {
      fastestTime = timing.fps;
      fastestHash =timing.hashCode;
    }
  }
  cube.down = fastestHash;

  fastestTime = std::numeric_limits<float>::min();
  fastestHash = 0;
  std::vector<TimingObject>* cubeSideTimingsRight = &getCubeSideElements(CubeSide::CubeRight,currentGrid)->storedElements;
  for(TimingObject timing : *cubeSideTimingsRight)
  {
    if(timing.fps > fastestTime)
    {
      fastestTime = timing.fps;
      fastestHash =timing.hashCode;
    }
  }
  cube.right = fastestHash;

  fastestTime = std::numeric_limits<float>::min();
  fastestHash = 0;
  std::vector<TimingObject>* cubeSideTimingsLeft = &getCubeSideElements(CubeSide::CubeLeft,currentGrid)->storedElements;
  for(TimingObject timing : *cubeSideTimingsLeft)
  {
    if(timing.fps > fastestTime)
    {
      fastestTime = timing.fps;
      fastestHash =timing.hashCode;
    }
  }
  cube.left = fastestHash;

  fastestTime = std::numeric_limits<float>::min();
  fastestHash = 0;
  std::vector<TimingObject>* cubeSideTimingsFront = &getCubeSideElements(CubeSide::CubeFront,currentGrid)->storedElements;
  for(TimingObject timing : *cubeSideTimingsFront)
  {
    if(timing.fps > fastestTime)
    {
      fastestTime = timing.fps;
      fastestHash =timing.hashCode;
    }
  }
  cube.front = fastestHash;

  fastestTime = std::numeric_limits<float>::min();
  fastestHash = 0;
  std::vector<TimingObject>* cubeSideTimingsback = &getCubeSideElements(CubeSide::CubeBack,currentGrid)->storedElements;
  for(TimingObject timing : *cubeSideTimingsback)
  {
    if(timing.fps > fastestTime)
    {
      fastestTime = timing.fps;
      fastestHash =timing.hashCode;
    }
  }
  cube.back = fastestHash;

  return cube;
}
void SampleExample::updateStorageBuffer(const VkCommandBuffer& cmdBuf)
{
  if(m_busy)
    return;

  LABEL_SCOPE_VK(cmdBuf);


  //upddate best keys data
if(m_gui->VisualizeSortingGrid)
{
  for(int i = 0; i < grid_x;i++)
  {
    for(int j = 0; j < grid_y;j++)
    {
      for(int k = 0; k < grid_z;k++)
      {
        //determine index in buffer, densely packed
        int index = k*(grid_y*grid_x) + j*grid_x + i;
        
        //determine best Key seen yet for each gridspace and viewing direction



        bestKeys[index] = determineBestTimesCube(&grid.gridSpaces[k][j][i]);
      }
    }
  }

  
  vkCmdUpdateBuffer(cmdBuf,m_GridSortingKeyBuffer.buffer,0,sizeof(GridCube[1000]),&bestKeys);
}

}

//--------------------------------------------------------------------------------------------------
// Descriptors for the Sun&Sky buffer
//
void SampleExample::createDescriptorSetLayout()
{
  VkShaderStageFlags flags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                             | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;


  m_bind.addBinding({EnvBindings::eSunSky, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_MISS_BIT_KHR | flags});
  m_bind.addBinding({EnvBindings::eSortParameters, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_MISS_BIT_KHR | flags});
  m_bind.addBinding({EnvBindings::eHdr, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags});  // HDR image
  m_bind.addBinding({EnvBindings::eImpSamples, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags});   // importance sampling
  m_bind.addBinding({EnvBindings::eGridKeys, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags});   // importance sampling


  m_descPool = m_bind.createPool(m_device, 1);
  CREATE_NAMED_VK(m_descSetLayout, m_bind.createLayout(m_device));
  CREATE_NAMED_VK(m_descSet, nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

  // Using the environment
  std::vector<VkWriteDescriptorSet> writes;
  VkDescriptorBufferInfo            sunskyDesc{m_sunAndSkyBuffer.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo            sortParametersDesc{m_sortingParametersBuffer.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo            accelImpSmpl{m_skydome.m_accelImpSmpl.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo            gridKeysDesc{m_GridSortingKeyBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eSunSky, &sunskyDesc));
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eHdr, &m_skydome.m_texHdr.descriptor));
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eImpSamples, &accelImpSmpl));
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eSortParameters, &sortParametersDesc));
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eGridKeys, &gridKeysDesc));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Setting the descriptor for the HDR and its acceleration structure
//
void SampleExample::updateHdrDescriptors()
{
  std::vector<VkWriteDescriptorSet> writes;
  VkDescriptorBufferInfo            accelImpSmpl{m_skydome.m_accelImpSmpl.buffer, 0, VK_WHOLE_SIZE};

  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eHdr, &m_skydome.m_texHdr.descriptor));
  writes.emplace_back(m_bind.makeWrite(m_descSet, EnvBindings::eImpSamples, &accelImpSmpl));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the Sun&Sky structure
// - Buffer is host visible and will be set each frame
//
void SampleExample::createUniformBuffer()
{
  m_sunAndSkyBuffer = m_alloc.createBuffer(sizeof(SunAndSky), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  NAME_VK(m_sunAndSkyBuffer.buffer);

  m_sortingParametersBuffer = m_alloc.createBuffer(sizeof(SortingParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); 
  NAME_VK(m_sortingParametersBuffer.buffer);                           
  
}



//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void SampleExample::destroyResources()
{
  // Resources
  m_alloc.destroy(m_sunAndSkyBuffer);
  m_alloc.destroy(m_sortingParametersBuffer);
  m_alloc.destroy(m_GridSortingKeyBuffer);

  // Descriptors
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  // Other
  m_picker.destroy();
  m_scene.destroy();
  m_accelStruct.destroy();
  m_offscreen.destroy();
  m_skydome.destroy();
  m_axis.deinit();

  // All renderers
  for(auto p : m_pRender)
  {
    p->destroy();
    p = nullptr;
  }

  // Memory
  m_staging.deinit();
  m_alloc.deinit();

}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void SampleExample::onResize(int /*w*/, int /*h*/)
{
  m_offscreen.update(m_size);
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Call the rendering of all graphical user interface
//
void SampleExample::renderGui(nvvk::ProfilerVK& profiler)
{
  m_gui->titleBar();
  m_gui->menuBar();
  m_gui->render(profiler);

  auto& IO = ImGui::GetIO();
  if(ImGui::IsMouseDoubleClicked(ImGuiDir_Left) && !ImGui::GetIO().WantCaptureKeyboard)
  {
    screenPicking();
  }
}


//--------------------------------------------------------------------------------------------------
// Creating the render: RTX, Ray Query, ...
// - Destroy the previous one.
void SampleExample::createRender(RndMethod method)
{
  //if(method == m_rndMethod)
  //  return;

  LOGI("Switching renderer, from %d to %d \n", m_rndMethod, method);
  if(m_rndMethod != eNone)
  {
    vkDeviceWaitIdle(m_device);  // cannot destroy while in use
    m_pRender[m_rndMethod]->destroy();
  }
  m_rndMethod = method;

  m_pRender[m_rndMethod]->create(
      m_size, {m_accelStruct.getDescLayout(), m_offscreen.getDescLayout(), m_scene.getDescLayout(), m_descSetLayout}, &m_scene);
}

void SampleExample::rebuildRender()
{
  vkDeviceWaitIdle(m_device);
  m_pRender[m_rndMethod]->destroy();
  m_pRender[m_rndMethod]->create(
      m_size, {m_accelStruct.getDescLayout(), m_offscreen.getDescLayout(), m_scene.getDescLayout(), m_descSetLayout}, &m_scene);

}

//--------------------------------------------------------------------------------------------------
// Creating the render: RTX, Ray Query, ...
// - Destroy the previous one.
void SampleExample::reloadRender()
{


  LOGI("Reloading renderer\n");

  vkDeviceWaitIdle(m_device);  // cannot destroy while in use
  //m_pRender[m_rndMethod]->destroy();
  m_pRender[m_rndMethod]->create(
      m_size, {m_accelStruct.getDescLayout(), m_offscreen.getDescLayout(), m_scene.getDescLayout(), m_descSetLayout}, &m_scene);

}

//--------------------------------------------------------------------------------------------------
// The GUI is taking space and size of the rendering area is smaller than the viewport
// This is the space left in the center view.
void SampleExample::setRenderRegion(const VkRect2D& size)
{
  if(memcmp(&m_renderRegion, &size, sizeof(VkRect2D)) != 0)
    resetFrame();
  m_renderRegion = size;
}

//////////////////////////////////////////////////////////////////////////
// Post ray tracing
//////////////////////////////////////////////////////////////////////////

void SampleExample::createOffscreenRender()
{
  m_offscreen.create(m_size, m_renderPass);
  m_axis.init(m_device, m_renderPass, 0, 50.0f);
}

//--------------------------------------------------------------------------------------------------
// This will draw the result of the rendering and apply the tonemapper.
// If enabled, draw orientation axis in the lower left corner.
void SampleExample::drawPost(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  auto size = glm::vec2(m_size.width, m_size.height);
  auto area = glm::vec2(m_renderRegion.extent.width, m_renderRegion.extent.height);

  VkViewport viewport{static_cast<float>(m_renderRegion.offset.x),
                      static_cast<float>(m_renderRegion.offset.y),
                      static_cast<float>(m_size.width),
                      static_cast<float>(m_size.height),
                      0.0f,
                      1.0f};
  VkRect2D   scissor{m_renderRegion.offset, {m_renderRegion.extent.width, m_renderRegion.extent.height}};
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

  m_offscreen.m_tonemapper.zoom           = m_descaling ? 1.0f / m_descalingLevel : 1.0f;
  m_offscreen.m_tonemapper.renderingRatio = size / area;
  m_offscreen.run(cmdBuf);

  if(m_showAxis)
    m_axis.display(cmdBuf, CameraManip.getMatrix(), m_size);
}


//////////////////////////////////////////////////////////////////////////
// Ray tracing
//////////////////////////////////////////////////////////////////////////

void SampleExample::renderScene(const VkCommandBuffer& cmdBuf, nvvk::ProfilerVK& profiler)
{
#if defined(NVP_SUPPORTS_NVML)
  g_nvml.refresh();
#endif

  if(m_busy)
  {
    m_gui->showBusyWindow();  // Busy while loading scene
    return;
  }

  LABEL_SCOPE_VK(cmdBuf);

  auto sec = profiler.timeRecurring("Render", cmdBuf);

  // We are done rendering
  if(m_rtxState.frame >= m_maxFrames)
    return;

  // Handling de-scaling by reducing the size to render
  VkExtent2D render_size = m_renderRegion.extent;
  if(m_descaling)
    render_size = VkExtent2D{render_size.width / m_descalingLevel, render_size.height / m_descalingLevel};
  

  m_rtxState.size = {render_size.width, render_size.height};


  m_rtxState.SceneMax = m_scene.getScene().m_dimensions.max;
  m_rtxState.SceneMin = m_scene.getScene().m_dimensions.min;
  m_rtxState.gridX = grid_x;
  m_rtxState.gridY = grid_y;
  m_rtxState.gridZ = grid_z;
  m_rtxState.SceneCenter = m_scene.getScene().m_dimensions.center; 

//std::cout << "SceneCenter: " << m_rtxState.SceneCenter.x << " "<<m_rtxState.SceneCenter.y <<" " << m_rtxState.SceneCenter.z << std::endl;
//std::cout << "SceneMin: " << m_rtxState.SceneMin.x << " "<<m_rtxState.SceneMin.y <<" " << m_rtxState.SceneMin.z << std::endl;
//std::cout << "SceneMax: " << m_rtxState.SceneMax.x << " "<<m_rtxState.SceneMax.y <<" " << m_rtxState.SceneMax.z << std::endl;
glm::vec3 distScene = m_rtxState.SceneMax - m_rtxState.SceneMin;
glm::vec3 cameraPos = CameraManip.getEye();
glm::vec3 cameraInterest = glm::normalize(CameraManip.getCenter() - cameraPos);
glm::vec3 up{0.0,1.0,0.0};
glm::vec3 down{0.0,-1.0,0.0};
glm::vec3 right{1.0,0.0,0.0};
glm::vec3 left{-1.0,0.0,0.0};
glm::vec3 front{0.0,0.0,1.0};
glm::vec3 back{0.0,0.0,-1.0};
float closestMatch = 0.0;
CubeSide closestmatchedSide;
float angle = glm::dot(cameraInterest,up);
if(angle> closestMatch)
{
  closestMatch = angle;
  closestmatchedSide = CubeUp;
}
angle = glm::dot(cameraInterest,down);
if(angle > closestMatch)
{
  closestMatch = angle;
  closestmatchedSide = CubeDown;
}
angle = glm::dot(cameraInterest,left);
if(angle > closestMatch)
{
  closestMatch = angle;
  closestmatchedSide = CubeLeft;
}
angle = glm::dot(cameraInterest,right);
if(angle > closestMatch)
{
  closestMatch = angle;
  closestmatchedSide = CubeRight;
}
angle = glm::dot(cameraInterest,front);
if(angle > closestMatch)
{
  closestMatch = angle;
  closestmatchedSide = CubeFront;
}
angle = glm::dot(cameraInterest,back);
if(angle > closestMatch)
{
  closestMatch = angle;
  closestmatchedSide = CubeBack;
}
currentLookDirection = closestmatchedSide;

glm::vec3 gridSizes = glm::vec3(distScene.x/grid_x,distScene.y/grid_y, distScene.z/grid_z);

float epsilon = 0.001f; // to ensure correct grid placement

//clip cameraPos to bounds of Scene
glm::vec3 clippedCameraPos = glm::vec3(glm::min(glm::max(cameraPos.x,m_rtxState.SceneMin.x),m_rtxState.SceneMax.x-epsilon),glm::min(glm::max(cameraPos.y,m_rtxState.SceneMin.y),m_rtxState.SceneMax.y-epsilon),glm::min(glm::max(cameraPos.z,m_rtxState.SceneMin.z),m_rtxState.SceneMax.z-epsilon));

//if(cameraPos.x >m_rtxState.SceneMax.x || cameraPos.x < m_rtxState.SceneMin.x ||cameraPos.y >m_rtxState.SceneMax.y || cameraPos.y < m_rtxState.SceneMin.y ||cameraPos.z >m_rtxState.SceneMax.z || cameraPos.z < m_rtxState.SceneMin.z)
//{
  //currentGridSpace = glm::vec3(-1,-1,-1);
//} else 
{
  glm::vec3 relativeCamPosition = clippedCameraPos - m_rtxState.SceneMin;

  int gridSpaceX = glm::floor(relativeCamPosition.x / gridSizes.x);
  int gridSpaceY = glm::floor(relativeCamPosition.y / gridSizes.y);
  int gridSpaceZ = glm::floor(relativeCamPosition.z / gridSizes.z);

  currentGridSpace = glm::vec3(gridSpaceX,gridSpaceY,gridSpaceZ);
}
auto rtx = dynamic_cast<RtxPipeline*>(m_pRender[m_rndMethod]);
if(useBestParameters)
{

  CubeSideStorage* cubeSide = getCubeSideElements(currentLookDirection,&grid.gridSpaces[currentGridSpace.z][currentGridSpace.y][currentGridSpace.x]);
  PipelineStorage bestPipeline = cubeSide->bestPipeline;
  if(bestPipeline.pipeline !=VK_NULL_HANDLE)
  {
    vkDeviceWaitIdle(m_device);
    rtx->setNewPipeline(bestPipeline);
  }
  
}
  // State is the push constant structure
  m_pRender[m_rndMethod]->setPushContants(m_rtxState);
  // Running the renderer


  
  auto render_ID = profiler.beginSection("Render Section",cmdBuf);
  m_pRender[m_rndMethod]->run(cmdBuf, render_size, profiler,
                              {m_accelStruct.getDescSet(), m_offscreen.getDescSet(), m_scene.getDescSet(), m_descSet});
  profiler.endSection(render_ID,cmdBuf);








int sortMode = *(rtx->getSortingMode());




  // For automatic brightness tonemapping
  if(m_offscreen.m_tonemapper.autoExposure)
  {
    auto slot = profiler.timeRecurring("Mipmap", cmdBuf);
    m_offscreen.genMipmap(cmdBuf);
  }

}


//////////////////////////////////////////////////////////////////////////
// Keyboard / Drag and Drop
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Overload keyboard hit
// - Home key: fit all, the camera will move to see the entire scene bounding box
// - Space: Trigger ray picking and set the interest point at the intersection
//          also return all information under the cursor
//
void SampleExample::onKeyboard(int key, int scancode, int action, int mods)
{
  nvvkhl::AppBaseVk::onKeyboard(key, scancode, action, mods);

  if(m_busy || action == GLFW_RELEASE)
    return;

  switch(key)
  {
    case GLFW_KEY_HOME:
    case GLFW_KEY_F:  // Set the camera as to see the model
      fitCamera(m_scene.getScene().m_dimensions.min, m_scene.getScene().m_dimensions.max, false);
      break;
    case GLFW_KEY_SPACE:
      screenPicking();
      break;
    case GLFW_KEY_R:
      resetFrame();
      break;
    default:
      break;
  }
}

//--------------------------------------------------------------------------------------------------
//
//
void SampleExample::screenPicking()
{
  double x, y;
  glfwGetCursorPos(m_window, &x, &y);

  // Set the camera as to see the model
  nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = sc.createCommandBuffer();

  const float aspectRatio = m_renderRegion.extent.width / static_cast<float>(m_renderRegion.extent.height);
  const auto& view        = CameraManip.getMatrix();
  auto        proj        = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspectRatio, 0.1f, 1000.0f);
  proj[1][1] *= -1;

  nvvk::RayPickerKHR::PickInfo pickInfo;
  pickInfo.pickX          = float(x - m_renderRegion.offset.x) / float(m_renderRegion.extent.width);
  pickInfo.pickY          = float(y - m_renderRegion.offset.y) / float(m_renderRegion.extent.height);
  pickInfo.modelViewInv   = glm::inverse(view);
  pickInfo.perspectiveInv = glm::inverse(proj);


  m_picker.run(cmdBuf, pickInfo);
  sc.submitAndWait(cmdBuf);

  nvvk::RayPickerKHR::PickResult pr = m_picker.getResult();

  if(pr.instanceID == ~0)
  {
    LOGI("Nothing Hit\n");
    return;
  }

  glm::vec3 worldPos = glm::vec3(pr.worldRayOrigin + pr.worldRayDirection * pr.hitT);
  // Set the interest position
  glm::vec3 eye, center, up;
  CameraManip.getLookat(eye, center, up);
  CameraManip.setLookat(eye, worldPos, up, false);


  auto& prim = m_scene.getScene().m_primMeshes[pr.instanceCustomIndex];
  LOGI("Hit(%d): %s\n", pr.instanceCustomIndex, prim.name.c_str());
  LOGI(" - PrimId(%d)\n", pr.primitiveID);
}

//--------------------------------------------------------------------------------------------------
//
//
void SampleExample::onFileDrop(const char* filename)
{
  if(m_busy)
    return;

  loadAssets(filename);
}

//--------------------------------------------------------------------------------------------------
// Window callback when the mouse move
// - Handling ImGui and a default camera
//
void SampleExample::onMouseMotion(int x, int y)
{
  AppBaseVk::onMouseMotion(x, y);
  if(m_busy)
    return;

  if(ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard)
    return;

  if(m_inputs.lmb || m_inputs.rmb || m_inputs.mmb)
  {
    m_descaling = true;
  }
}

//--------------------------------------------------------------------------------------------------
//
//
void SampleExample::onMouseButton(int button, int action, int mods)
{
  AppBaseVk::onMouseButton(button, action, mods);
  if(m_busy)
    return;

  if((m_inputs.lmb || m_inputs.rmb || m_inputs.mmb) == false && action == GLFW_RELEASE && m_descaling == true)
  {
    m_descaling = false;
    resetFrame();
  }
}

bool parametersLegalCheck(SortingParameters parameters)
{ 
  //both th information for the actua ray endpoint or the hitobject sorting only exist after Ray traversal has been performed
  //So a parameter set that uses them before Ray traversal would be illegal
  // we also disqualify estimated Endpoint information with either setting, sicne they are redundant
  if((parameters.realEndpoint || parameters.hitObject) && (parameters.estimatedEndpoint || (!parameters.sortAfterASTraversal)))
  {
    return false;
  }
  // we  disqualify estimated Endpoint information being used after ray traversal, since that would be redundant
  if(parameters.estimatedEndpoint && parameters.sortAfterASTraversal)
  {
    return false;
  }
  if(parameters.noSort &&(parameters.hitObject))
  {
    return false;
  }
  return true;
}
SortingParameters SampleExample::createSortingParameters()
{
  SortingParameters result;
  bool isLegal = false;
  std::uniform_int_distribution<std::mt19937::result_type> dist32(1,32);
  std::uniform_int_distribution<std::mt19937::result_type> distBool(0,1);

  while(!isLegal)
  {

    //random number of coherence Bits
    result.numCoherenceBitsTotal = dist32(rng);
    result.sortAfterASTraversal = distBool(rng);
    result.estimatedEndpoint = distBool(rng);
    result.realEndpoint = distBool(rng);
    result.noSort = distBool(rng);
    result.hitObject = distBool(rng);
    result.rayDirection = distBool(rng);
    result.rayOrigin =  distBool(rng);
    result.isFinished = distBool(rng);



    isLegal = parametersLegalCheck(result);
  }
  return result;
}

void SampleExample::beginSortingGridTraining()
{
  constantGridlearningSpeed = 1.0f;
  trainingDirectionIndex = 0;
  trainingPosition = glm::vec3(0,0,0);

  glm::vec3 trainingStartPosition = calculateGridSpaceCenter(trainingPosition);

  glm::vec3 newCameraDirection = trainingStartPosition+ lookDirections[trainingDirectionIndex];
  trainingDirectionIndex++;

    //glm::vec3 newCameraDirection = CameraManip.getCenter();
  CameraManip.setLookat(trainingStartPosition,newCameraDirection,CameraManip.getUp());

}

void SampleExample::iterateTrainingPosition()
{
  glm::vec3 result;

  if(trainingPosition.z+1 == grid_z)
  {
    if(trainingPosition.y+1 == grid_y)
    {
      if(trainingPosition.x+1 == grid_x)
      {
        //finished training, each grid cell has been visited and fully tested.
        performAutomaticTraining = false;
      } else {
        trainingPosition = glm::vec3(trainingPosition.x+1,0,0);
      }
    } else {
      trainingPosition = glm::vec3(trainingPosition.x,trainingPosition.y+1,0);
    }
  } else {
    trainingPosition = glm::vec3(trainingPosition.x,trainingPosition.y,trainingPosition.z+1);
  }

}
void SampleExample::doCycle()
{
  //timer
  if(framesThisCycle == 0)
  {
    framesThisCycle++;

    return;
  }
 timeRemaining -= ImGui::GetIO().DeltaTime * 1000;
 framesThisCycle++;

 if(timeRemaining < 0.0)
 {


  printf(std::to_string(framesThisCycle).c_str());
  printf("\n");
  timeRemaining = timePerCycle;
  auto rtx = dynamic_cast<RtxPipeline*>(m_pRender[m_rndMethod]);
  int hashCode = rtx->hashParameters(rtx->m_SERParameters);
  bool foundOne = false;
  bool foundOne2 = false;

   //currentGrid = &sortingGrid[currentGridSpace.z][currentGridSpace.y][currentGridSpace.x];
  GridSpace* currentGrid = &grid.gridSpaces[currentGridSpace.z][currentGridSpace.y][currentGridSpace.x];
  
  CubeSideStorage* cubeSide = getCubeSideElements(currentLookDirection,currentGrid);
  std::vector<TimingObject>* observedData = &cubeSide->storedElements;
  

  for(int i = 0; i < observedData->size(); i++)
  {
    TimingObject* object = &observedData->at(i);
    if(hashCode == object->hashCode)
    {
      object->frames += framesThisCycle;
      object->totalCycles += 1;
      object->fps = object->frames*1000/(timePerCycle * object->totalCycles);

      int bestPipelineParameterHash = rtx->hashParameters(cubeSide->bestPipeline.parameters);
      

      // when current parameters and the ones of the best pipeline are identical, update best pipeline timing
      if(hashCode == bestPipelineParameterHash)
      {
        cubeSide->bestpipelineFPS = object->fps;
      }
      else {
        //if they are different test if current parameters are faster, update bestPipeline if they are
        if(object->fps > cubeSide->bestpipelineFPS)
          {
            cubeSide->bestpipelineFPS = object->fps;
            cubeSide->bestPipeline = rtx->activeElement;
          }
      }

      foundOne2 = true;
      break;
    }
  }
  if(!foundOne2)
  {
    TimingObject newTiming;
    newTiming.hashCode = hashCode;
    newTiming.frames = framesThisCycle;
    newTiming.fps = framesThisCycle*1000/timePerCycle;
    newTiming.totalCycles = 1;
    observedData->emplace_back(newTiming);

    if(newTiming.fps > cubeSide->bestpipelineFPS)
      {
        cubeSide->bestpipelineFPS = newTiming.fps;
        cubeSide->bestPipeline = rtx->activeElement;
        
      }
  }
  int minNumberTestedConfigs = 5;
  int numTestedConfigs = observedData->size();
  float randValue = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  float val = glm::max(numTestedConfigs-minNumberTestedConfigs,0) / 5;

  if(performAutomaticTraining && val > randValue)
  {
    if(trainingDirectionIndex == 0)
    {
      iterateTrainingPosition();
    }
    glm::vec3 newCameraPosition = calculateGridSpaceCenter(trainingPosition);
    glm::vec3 newCameraDirection = newCameraPosition+ lookDirections[trainingDirectionIndex];
    trainingDirectionIndex = trainingDirectionIndex+1;
    if(trainingDirectionIndex ==6)
    {
      trainingDirectionIndex = 0;
    }
    //glm::vec3 newCameraDirection = CameraManip.getCenter();
    CameraManip.setLookat(newCameraPosition,newCameraDirection,CameraManip.getUp());
  }

  framesThisCycle = 0;
  float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  
  //with probability 1-epsilon, or no pipeline is porea
  float epsilon = useConstantGridLearning ? constantGridlearningSpeed : currentGrid->adaptiveGridLearningRate;
  if(r > epsilon || rtx->PrebuildPipelineBuffer.empty())
  {


      printf("exploit\n");
      if(currentGrid->bestPipeline.pipeline != VK_NULL_HANDLE)
      {
        rtx->setNewPipeline(currentGrid->bestPipeline);
        printf("chose faster one\n");
      }



  }
  //otherwise explore
  else {
      vkDeviceWaitIdle(m_device);
      rtx->setNewPipeline();
      printf("explore\n");
      if(!useConstantGridLearning)
      {
        currentGrid->adaptiveGridLearningRate -= currentGrid->adaptiveGridLearningRate/10.0f;
        currentGrid->adaptiveGridLearningRate = glm::max(currentGrid->adaptiveGridLearningRate,0.1f);
      }
  }
 }

}


void SampleExample::buildSortingGrid()
{
  std::vector<std::vector<std::vector<GridSpace>>> newSortingGrid;
  //newSortingGrid.resize(grid_y,std::vector<GridSpace>(grid_x));
  newSortingGrid.resize(grid_z,std::vector<std::vector<GridSpace> >(grid_y,std::vector<GridSpace>(grid_x)));
  sortingGrid = newSortingGrid;
  grid.gridSpaces = newSortingGrid;
  grid.gridDimensions = glm::vec3(grid_x,grid_y,grid_z);
  printf("build new Grid with dimension %d , %d \n",grid_y,grid_x);
}

#include <ctime>

//int SampleExample::getCubeSideHash()

json SampleExample::fillJsonWithAllResults(json js)
{
  std::string s1 = "(" + std::to_string(0) + "," + std::to_string(0) + "," + std::to_string(0) + ")";
  CubeSideStorage* cubeside= getCubeSideElements(CubeUp,&grid.gridSpaces[0][0][0]);
for(int i = 0; i < grid.gridDimensions.x; i++)
  {
    for(int j = 0; j < grid.gridDimensions.y; j++)
    {
      for(int k = 0; k < grid.gridDimensions.z; k++)
      {
        std::string s1 = "(" + std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k) + ")";
        //CubeSideStorage* cubeside= getCubeSideElements(CubeBack,&grid.gridSpaces[k][j][i]);
        CubeSideStorage* cubeside= getCubeSideElements(CubeUp,&grid.gridSpaces[k][j][i]);

        if(cubeside->storedElements.empty())
        {
          js["Observations"][s1]["top"] = 1;
        } else {
          for(TimingObject timing : cubeside->storedElements)
          {
            
          js["Observations"][s1]["top"][std::to_string(timing.hashCode)] = timing.fps;
          }
        }

        cubeside= getCubeSideElements(CubeDown,&grid.gridSpaces[k][j][i]);

        if(cubeside->storedElements.empty())
        {
          js["Observations"][s1]["bottom"] = 1;
        } else {
          for(TimingObject timing : cubeside->storedElements)
          {
            
            js["Observations"][s1]["bottom"][std::to_string(timing.hashCode)] = timing.fps;
          }
        }

          cubeside= getCubeSideElements(CubeLeft,&grid.gridSpaces[k][j][i]);

          if(cubeside->storedElements.empty())
          {
            js["Observations"][s1]["left"] = 1;
          }
          else {
            for(TimingObject timing : cubeside->storedElements)
            {
              
              js["Observations"][s1]["left"][std::to_string(timing.hashCode)] = timing.fps;
            }
          }

          cubeside= getCubeSideElements(CubeRight,&grid.gridSpaces[k][j][i]);

          if(cubeside->storedElements.empty())
          {
            js["Observations"][s1]["right"] = 1;
          }
          else {
            for(TimingObject timing : cubeside->storedElements)
            {
              
              js["Observations"][s1]["right"][std::to_string(timing.hashCode)] = timing.fps;
            }
          }

          cubeside= getCubeSideElements(CubeFront,&grid.gridSpaces[k][j][i]);

          if(cubeside->storedElements.empty())
          {
            js["Observations"][s1]["front"] = 1;
          }
          else {
            for(TimingObject timing : cubeside->storedElements)
            {
              
              js["Observations"][s1]["front"][std::to_string(timing.hashCode)] = timing.fps;
            }
          }


          cubeside= getCubeSideElements(CubeBack,&grid.gridSpaces[k][j][i]);
          if(cubeside->storedElements.empty())
          {
            js["Observations"][s1]["back"] = 1;
          }
          else {
            for(TimingObject timing : cubeside->storedElements)
            {
              
              js["Observations"][s1]["back"][std::to_string(timing.hashCode)] = timing.fps;
            }
          }
      }
    }
  }

  return js;
}

json SampleExample::fillJsonWithBestResult(json js)
{
  std::string s1 = "(" + std::to_string(0) + "," + std::to_string(0) + "," + std::to_string(0) + ")";
  CubeSideStorage* cubeside= getCubeSideElements(CubeUp,&grid.gridSpaces[0][0][0]);
for(int i = 0; i < grid.gridDimensions.x; i++)
  {
    for(int j = 0; j < grid.gridDimensions.y; j++)
    {
      for(int k = 0; k < grid.gridDimensions.z; k++)
      {
        std::string s1 = "(" + std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k) + ")";
        //CubeSideStorage* cubeside= getCubeSideElements(CubeBack,&grid.gridSpaces[k][j][i]);
        CubeSideStorage* cubeside= getCubeSideElements(CubeUp,&grid.gridSpaces[k][j][i]);

        if(cubeside->storedElements.empty())
        {
          js["Observations"][s1]["top"] = 1;
        } else {
          float fastestTime = std::numeric_limits<float>::min();
          int fastestParameters = 0;
          for(TimingObject timing : cubeside->storedElements)
          {
            if(timing.fps > fastestTime)
            {
              fastestTime = timing.fps;
              fastestParameters = timing.hashCode;
            }
          }
          js["Observations"][s1]["top"] = {fastestParameters,fastestTime};
        }

        cubeside= getCubeSideElements(CubeDown,&grid.gridSpaces[k][j][i]);

        if(cubeside->storedElements.empty())
        {
          js["Observations"][s1]["bottom"] = 1;
        } else {
          float fastestTime = std::numeric_limits<float>::min();
          int fastestParameters = 0;
          for(TimingObject timing : cubeside->storedElements)
          {
            if(timing.fps > fastestTime)
            {
              fastestTime = timing.fps;
              fastestParameters = timing.hashCode;
            }
          }
          js["Observations"][s1]["bottom"] = {fastestParameters,fastestTime};
        }

          cubeside= getCubeSideElements(CubeLeft,&grid.gridSpaces[k][j][i]);

          if(cubeside->storedElements.empty())
          {
            js["Observations"][s1]["left"] = 1;
          }
          else {
            float fastestTime = std::numeric_limits<float>::min();
            int fastestParameters = 0;
            for(TimingObject timing : cubeside->storedElements)
            {
              if(timing.fps > fastestTime)
              {
                fastestTime = timing.fps;
                fastestParameters = timing.hashCode;
              }
            }
            js["Observations"][s1]["left"] = {fastestParameters,fastestTime};
          }

          cubeside= getCubeSideElements(CubeRight,&grid.gridSpaces[k][j][i]);

          if(cubeside->storedElements.empty())
          {
            js["Observations"][s1]["right"] = 1;
          }
          else {
            float fastestTime = std::numeric_limits<float>::min();
            int fastestParameters = 0;
            for(TimingObject timing : cubeside->storedElements)
            {
              if(timing.fps > fastestTime)
              {
                fastestTime = timing.fps;
                fastestParameters = timing.hashCode;
              }
            }
            js["Observations"][s1]["right"] = {fastestParameters,fastestTime};
          }

          cubeside= getCubeSideElements(CubeFront,&grid.gridSpaces[k][j][i]);

          if(cubeside->storedElements.empty())
          {
            js["Observations"][s1]["front"] = 1;
          }
          else {
            float fastestTime = std::numeric_limits<float>::min();
            int fastestParameters = 0;
            for(TimingObject timing : cubeside->storedElements)
            {
              if(timing.fps > fastestTime)
              {
                fastestTime = timing.fps;
                fastestParameters = timing.hashCode;
              }
            }
            js["Observations"][s1]["front"] = {fastestParameters,fastestTime};
          }


          cubeside= getCubeSideElements(CubeBack,&grid.gridSpaces[k][j][i]);
          if(cubeside->storedElements.empty())
          {
            js["Observations"][s1]["back"] = 1;
          }
          else {
            float fastestTime = std::numeric_limits<float>::min();
            int fastestParameters = 0;
            for(TimingObject timing : cubeside->storedElements)
            {
              if(timing.fps > fastestTime)
              {
                fastestTime = timing.fps;
                fastestParameters = timing.hashCode;
              }
            }
            js["Observations"][s1]["back"] = {fastestParameters,fastestTime};
          }
      }
    }
  }

  return js;
}
void SampleExample::SaveSortingGrid()
{


  json j2;
  j2["Grid Dimensions (x,y,z)"] = {grid.gridDimensions.x,grid.gridDimensions.y,grid.gridDimensions.z};

  j2 = fillJsonWithAllResults(j2);
  std::string s = j2.dump(4);

  time_t timestamp = time(&timestamp);
  struct tm * datetime = localtime(&timestamp);
  //printf("%2d_%2d__%2d_%2d_%2d\n",datetime->tm_mday,datetime->tm_mon,datetime->tm_hour,datetime->tm_min,datetime->tm_sec);

  char buffer [80];
  strftime(buffer,80,"%d_%m-%H_%M_%S",datetime);
  //printf(buffer);
  std::string begin = "C:/Users/Frederik/Key_Inference/Sorting_Grid_Results/";
  std::string filename = std::string(buffer);
  std::string end = ".json";
  std::string fullFileName =begin + filename + end;
  std::fstream fs;
  std::ofstream outstream;

  outstream.open(fullFileName, std::fstream::out | std::fstream::app);

  outstream << s;

  outstream.close();
  
  printf("saved to file\n");
  printf("with File name: ");
  printf(fullFileName.c_str());
  
}




//CubeSideStorage SampleExa

CubeSideStorage* SampleExample::getCubeSideElements(CubeSide side,GridSpace* currentGrid)
{
  if(side == CubeSide::CubeBack)
  {
    return &currentGrid->cube.backElements;
  } else if(side == CubeSide::CubeDown)
  {
    return &currentGrid->cube.downElements;
  } else if(side == CubeSide::CubeFront)
  {
    return &currentGrid->cube.frontElements;
  } else if(side == CubeSide::CubeLeft)
  {
    return &currentGrid->cube.leftElements;
  } else if(side == CubeSide::CubeRight)
  {
    return &currentGrid->cube.rightElements;
  } else//Cube up
  {
    return &currentGrid->cube.upElements;
  }
}


glm::vec3 SampleExample::calculateGridSpaceCenter(glm::vec3 gridspace)
{
  glm::vec3 result{0.0,0.0,0.0};
  nvh::GltfScene& scene =  m_scene.getScene();

  glm::vec3 scene_min = scene.m_dimensions.min;
  glm::vec3 size = scene.m_dimensions.size;
  glm::vec3 stepX = glm::vec3(1.0,0.0,0.0) *  (size.x/grid_x);
  glm::vec3 stepY = glm::vec3(0.0,1.0,0.0) * (size.y/grid_y);
  glm::vec3 stepZ = glm::vec3(0.0,0.0,1.0) * (size.z/grid_z);
  result = scene_min;
  result += stepX * gridspace.x + stepY*gridspace.y + stepZ*gridspace.z;
  result += stepX * 0.5f + stepY* 0.5f + stepZ * 0.5f;

  return result;
}