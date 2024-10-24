#pragma once
#include <vector>
#include <random>
#include "glm/glm.hpp"
#include "shaders/host_device.h"

  struct TimingObject
  {
    int hashCode;
    int frames;
    float fps;
    int totalCycles;
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
};

struct Grid
{
  std::vector<std::vector<std::vector<GridSpace>>> gridSpaces;
  glm::vec3 gridDimensions;
};




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
bool parametersLegalCheck1(SortingParameters parameters);


