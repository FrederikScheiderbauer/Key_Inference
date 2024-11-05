#pragma once
#include <vector>
#include <random>
#include <limits>
#include "glm/glm.hpp"
#include "shaders/host_device.h"
#include "rtx_pipeline.hpp"
#include <unordered_map>
#include "json.hpp"

  struct TimingObject
  {
    int hashCode;
    int frames;
    float fps;
    int totalCycles;
  };

  struct CubeSideStorage
  {
    std::vector<TimingObject> storedElements;
    PipelineStorage bestPipeline;
    float bestpipelineFPS = 0.0f;
  };
  struct TimingCube
  {
    CubeSideStorage upElements;
    CubeSideStorage downElements;
    CubeSideStorage leftElements;
    CubeSideStorage rightElements;
    CubeSideStorage frontElements;
    CubeSideStorage backElements;
  };


  struct GridSpace
{
  TimingCube cube;
  float adaptiveGridLearningRate = 1.0f;
  GridCube bestKeyCube;
  float BestPipelineFPS = std::numeric_limits<float>::min();
  PipelineStorage bestPipeline;
};

struct Grid
{
  std::vector<std::vector<std::vector<GridSpace>>> gridSpaces;
  glm::vec3 gridDimensions;
};

//SortingParameters mostRecentParameters;




/*
class SortingGrid1
{
private:


    glm::vec3 gridDimensions1;
    Grid1 sortingGrid1;

    int gridX = 2;
    int gridY = 2;
    int gridZ = 2;


    void buildSortingGrid1();
public:
    SortingGrid1();
    ~SortingGrid1();


    Grid1 getSortingGrid(){return sortingGrid1;};
};

SortingGrid1::SortingGrid1()
{
}

SortingGrid1::~SortingGrid1()
{
}

*/

SortingParameters createSortingParameters1();
SortingParameters morphSortingParameters(SortingParameters parameters);
bool parametersLegalCheck1(SortingParameters parameters);

void storeSortingGrid1();

