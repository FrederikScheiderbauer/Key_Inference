#pragma once
#include <vector>
#include <random>
#include <limits>
#include "glm/glm.hpp"
#include "shaders/host_device.h"
#include "rtx_pipeline.hpp"
#include <unordered_map>

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
  };
  struct TimingCube
  {
    std::vector<TimingObject> up;
    std::vector<TimingObject> down;
    std::vector<TimingObject> left;
    std::vector<TimingObject> right;
    std::vector<TimingObject> front;
    std::vector<TimingObject> back;
  };


  struct GridSpace
{
  std::vector<TimingObject> observedData;
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


