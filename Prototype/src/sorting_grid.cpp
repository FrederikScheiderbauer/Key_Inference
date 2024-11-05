#include "sorting_grid.hpp"
#include <random>
#include <fstream>

using json = nlohmann::json;

bool parametersLegalCheck1(SortingParameters parameters)
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
  if(!(parameters.noSort || parameters.hitObject ||parameters.estimatedEndpoint || parameters.isFinished ||parameters.rayDirection || parameters.rayOrigin || parameters.rayOrigin || parameters.realEndpoint))
  {
    return false;
  }
  
  return true;
}
SortingParameters createSortingParameters1()
{
  std::random_device device;
  std::mt19937 e2(device());
  SortingParameters result;
  bool isLegal = false;
  std::uniform_int_distribution<std::mt19937::result_type> dist32(1,32);
  std::uniform_int_distribution<std::mt19937::result_type> distBool(0,1);

  while(!isLegal)
  {

    //random number of coherence Bits

    
    result.numCoherenceBitsTotal = dist32(e2);
    result.sortAfterASTraversal = distBool(e2);
    result.estimatedEndpoint = distBool(e2);
    result.realEndpoint = distBool(e2);
    result.noSort = distBool(e2);
    result.hitObject = distBool(e2);
    result.rayDirection = distBool(e2);
    result.rayOrigin =  distBool(e2);
    result.isFinished = distBool(e2);
    
    
    



    isLegal = parametersLegalCheck1(result);
  }
  return result;
}



SortingParameters morphSortingParameters(SortingParameters parameters)
{
  SortingParameters result;

  std::random_device device;
  std::mt19937 e2(device());
  bool isLegal = false;
  std::uniform_int_distribution<std::mt19937::result_type> dist32(1,32);
  std::uniform_int_distribution<std::mt19937::result_type> distBool(0,1);
  std::uniform_real_distribution<> distRand(0.0,1.0);

  


  while(!isLegal)
  {
    result = parameters;
    //random number of coherence Bits

    result.numCoherenceBitsTotal = dist32(e2);

    for(int i = 0; i < 1; i++)
    {
      float randVal = distRand(e2);

      if(randVal < 1.0/8.0)
      {
        result.sortAfterASTraversal &= 1;
        continue;
      }
      if(randVal < 2.0/8.0)
      {
        result.estimatedEndpoint &= 1;
        continue;
      }
      if(randVal < 3.0/8.0)
      {
        result.realEndpoint &= 1;
        continue;
      }
      if(randVal < 4.0/8.0)
      {
        result.noSort &= 1;
        continue;
      }
      if(randVal < 5.0/8.0)
      {
        result.hitObject &= 1;
        continue;
      }
      if(randVal < 6.0/8.0)
      {
        result.rayDirection &= 1;
        continue;
      }

      if(randVal < 7.0/8.0)
      {
        result.rayOrigin &= 1;
        continue;
      }
      result.isFinished &= 1;
    }

  

    isLegal = parametersLegalCheck1(result);
  }
  return result; 

}

void storeSortingGrid1()
{
  json j = {
  {"pi", 3.141},
  {"happy", true},
  {"name", "Niels"},
  {"nothing", nullptr},
  {"answer", {
    {"everything", 42}
  }},
  {"list", {1, 0, 2}},
  {"object", {
    {"currency", "USD"},
    {"value", 42.99}
  }}
};

std::string s = j.dump();

}
/*
void SortingGrid::buildSortingGrid()
{
  Grid newGrid;
  std::vector<std::vector<std::vector<GridSpace>>> newGridSpaces;
  //newSortingGrid.resize(grid_y,std::vector<GridSpace>(grid_x));
  newGridSpaces.resize(gridZ,std::vector<std::vector<GridSpace>>(gridY,std::vector<GridSpace>(gridX)));
  newGrid.gridSpaces = newGridSpaces;
  newGrid.gridDimensions = glm::vec3(gridX,gridY,gridZ);
  sortingGrid = newGrid;
  printf("build new Grid with dimension (%d , %d, %d) \n",gridZ,gridY,gridX);
}

*/