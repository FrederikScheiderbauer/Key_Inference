#include "sorting_grid.hpp"
#include <random>

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