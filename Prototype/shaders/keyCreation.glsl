#ifndef KEYCREATION_GLSL
#define KEYCREATION_GLSL

#include "host_device.h"
#include "globals.glsl"

/*
  vec3 SceneMax;
  vec3 SceneMin;
  */
float computeLargestSceneExtent()
{
    float largestSceneExtent = 0;
    largestSceneExtent = max(largestSceneExtent,abs(rtxState.SceneMax.x - rtxState.SceneMin.x));
    largestSceneExtent = max(largestSceneExtent,abs(rtxState.SceneMax.y - rtxState.SceneMin.y));
    largestSceneExtent = max(largestSceneExtent,abs(rtxState.SceneMax.z - rtxState.SceneMin.z));

    return largestSceneExtent;
}

uint SortingKeyOrigin(vec3 origin)
{
// Point for Morton codes.
        vec3 a = (origin.xyz - vec3(0)) / vec3(1);
        vec3 ia = a * 8388607.0f; //23b/dim

        uint64_t mortonCode = 0;
        for (int i = 22; i >= 2; --i) {
            mortonCode |= uint64_t(((floatBitsToInt(ia.x) >> i) & 1)) << (3 * i - 3); // max 63
            mortonCode |=  uint64_t(((floatBitsToInt(ia.y) >> i) & 1)) << (3 * i - 4); // max 62
            mortonCode |=  uint64_t(((floatBitsToInt(ia.z) >> i) & 1)) << (3 * i - 5); // max 61
        }
        mortonCode |=  uint64_t((floatBitsToInt(ia.x) & 1)); // max 0
        uint result = uint(mortonCode >> 32);
        return result;
}

//Reis et al.
//sorting Key creation considering Origin of Ray and their Direction, by Reis[2017]
//produces 32 bit encoding
uint SortingKeyReis(vec3 origin, vec3 direction)
{
        // Point for Morton codes.
        vec3 a = (origin.xyz - vec3(0)) / vec3(1);
        vec3 nd = normalize(direction);
        vec3 b;
        b.x = atan(nd.y, nd.x) / (2.0f * M_PI) + 0.5f;
        b.y = acos(nd.z) / M_PI;

        vec3 ia = a * 255.0f; //8b/dim
        vec3 ib = b * 255.0f;  //8b/dim

        uint64_t mortonCode = 0;

        for (int i = 7; i >= 1; --i) {
            mortonCode |=  uint64_t(((floatBitsToInt(ia.x) >> i) & 1)) << (3 * i + 10); // max 31
            mortonCode |=  uint64_t(((floatBitsToInt(ia.y) >> i) & 1)) << (3 * i + 9); // max 30
            mortonCode |=  uint64_t(((floatBitsToInt(ia.z) >> i) & 1)) << (3 * i + 8); // max 29
        }
        mortonCode |=  uint64_t((floatBitsToInt(ia.x) & 1)) << (10); // max 10

        for (int i = 7; i >= 3; --i) {
            mortonCode |=  uint64_t(((floatBitsToInt(ib.x) >> i) & 1)) << (2 * i - 5); // max 9
            mortonCode |=  uint64_t(((floatBitsToInt(ib.y) >> i) & 1)) << (2 * i - 6); // max 8
        }
        uint result = uint(mortonCode >> 32);
        // Output key.
        return result;
}

//Costa et al.: Direction-Origin
uint SortingKeyCosta(vec3 origin, vec3 direction)
{
        // Point for Morton codes.
        vec3 a = (origin.xyz - vec3(0)) / vec3(1);
        vec3 nd = normalize(direction);
        vec3 b;
        b.x = atan(nd.y, nd.x) / (2.0f * M_PI) + 0.5f;
        b.y = acos(nd.z) / M_PI;

        vec3 ia = a * 8191.0f; //13b/dim
        vec3 ib = b * 8191.0f;  //13b/dim

        uint64_t mortonCode = 0;

        for (int i = 12; i >= 9; --i) {
            mortonCode |= uint64_t(((floatBitsToInt(ib.x) >> i) & 1)) << (2 * i +39); // max 9
            mortonCode |= uint64_t(((floatBitsToInt(ib.y) >> i) & 1)) << (2 * i +38); // max 8
        }

        for (int i = 7; i >= 1; --i) {
            
            mortonCode |= uint64_t(((floatBitsToInt(ia.x) >> i) & 1)) << (3 * i + 19); // max 31
            mortonCode |= uint64_t(((floatBitsToInt(ia.y) >> i) & 1)) << (3 * i + 18); // max 30
            mortonCode |= uint64_t(((floatBitsToInt(ia.z) >> i) & 1)) << (3 * i + 17); // max 29

        }

        uint result = uint(mortonCode >> 32);
        // Output key.
        return result;
}

//Origin Direction interleaved

uint SortingKeyAila(vec3 origin, vec3 direction)
{
    // Point for Morton codes.
    vec3 a = (origin.xyz - vec3(0)) / vec3(1);
    vec3 b = (normalize(direction) +1.0f) *0.5f;

    vec3 ia = a * 8191.0f; //13b/dim
    vec3 ib = b * 8191.0f;  //13b/dim
    
    uint64_t mortonCode = 0;

    mortonCode |= uint64_t(((floatBitsToInt(ia.x) >> 12) & 1)) << (63); // max 63
    mortonCode |= uint64_t(((floatBitsToInt(ia.y) >> 12) & 1)) << (62); // max 62
    mortonCode |= uint64_t(((floatBitsToInt(ia.z) >> 12) & 1)) << (61); // max 61

    mortonCode |= uint64_t(((floatBitsToInt(ia.x)>> 11) & 1)) << (60); // max 60
    mortonCode |= uint64_t(((floatBitsToInt(ia.y) >> 11) & 1)) << (59); // max 59
    mortonCode |= uint64_t(((floatBitsToInt(ia.z) >> 11) & 1)) << (58); // max 58

    mortonCode |= uint64_t(((floatBitsToInt(ia.x) >> 10) & 1)) << (57); // max 57
    mortonCode |= uint64_t(((floatBitsToInt(ia.y) >> 10) & 1)) << (56); // max 56
    mortonCode |= uint64_t(((floatBitsToInt(ia.z) >> 10) & 1)) << (55); // max 55

    for (int i = 9; i >= 1; --i) {
        mortonCode |= uint64_t(((floatBitsToInt(ia.x) >> i) & 1)) << (6 * i + 0); // max 54
        mortonCode |= uint64_t(((floatBitsToInt(ia.y) >> i) & 1)) << (6 * i - 1); // max 53
        mortonCode |= uint64_t(((floatBitsToInt(ia.z) >> i) & 1)) << (6 * i - 2); // max 52
    }

    for (int i = 12; i >= 4; --i) {
        mortonCode |= uint64_t(((floatBitsToInt(ib.x) >> i) & 1)) << (6 * i - 21); // max 51
        mortonCode |= uint64_t(((floatBitsToInt(ib.y) >> i) & 1)) << (6 * i - 22); // max 50
        mortonCode |= uint64_t(((floatBitsToInt(ib.z) >> i) & 1)) << (6 * i - 23); // max 49
    }
    uint result = uint(mortonCode>> 32);
    return result;
}


uint SortingKeyTwoPoint(vec3 origin, vec3 direction,float rayLength)
{
    vec3 a = origin;
    vec3 b = origin + direction * rayLength;

    //scale a and b to if necessary

    a = (a - vec3(0)) / vec3(1);
    b = (b - vec3(0)) / vec3(1);

    //calculate Morton Codes

    vec3 ia =  a * 32767.0f; //15b/dim
    vec3 ib = b * 32767.0f;  //15b/dim

    uint64_t mortonCode = 0;

    for (int i = 14; i >= 4; --i) {
        mortonCode |= uint64_t(((floatBitsToInt(ia.x) >> i) & 1)) << (6 * i - 21); // max 63
        mortonCode |= uint64_t(((floatBitsToInt(ia.y) >> i) & 1)) << (6 * i - 22); // max 62
        mortonCode |= uint64_t(((floatBitsToInt(ia.z) >> i) & 1)) << (6 * i - 23); // max 61
    }

    for (int i = 14; i >= 5; --i) {
        mortonCode |= uint64_t(((floatBitsToInt(ib.x) >> i) & 1)) << (6 * i - 24); // max 60
        mortonCode |= uint64_t(((floatBitsToInt(ib.y) >> i) & 1)) << (6 * i - 25); // max 59
        mortonCode |= uint64_t(((floatBitsToInt(ib.z) >> i) & 1)) << (6 * i - 26); // max 58
    }

    mortonCode |= uint64_t(floatBitsToInt(ib.x) & 1);

    uint result = uint(mortonCode >> 32);

    return result;
}


uint SortingKeyEndPointEstimationHard(vec3 origin, vec3 direction)
{
    float sceneExtent = computeLargestSceneExtent();
    //float estimatedRayLength = 0.5 * rtxState.maxSceneExtent; //just hardcoded estimate atm
    float estimatedRayLength = 0.2 * sceneExtent; //just hardcoded estimate atm
    vec3 a = origin;
    vec3 b = origin + estimatedRayLength * direction;


    //scale a and b to if necessary

    a = (a - vec3(0)) / vec3(1);
    b = (b - vec3(0)) / vec3(1);

    //calculate Morton Codes

    vec3 ia =  a * 32767.0f; //15b/dim
    vec3 ib = b * 32767.0f;  //15b/dim

    uint64_t mortonCode = 0;

    for (int i = 14; i >= 4; --i) {
        mortonCode |= uint64_t(((floatBitsToInt(ib.x) >> i) & 1)) << (6 * i - 21); // max 63
        mortonCode |= uint64_t(((floatBitsToInt(ib.y) >> i) & 1)) << (6 * i - 22); // max 62
        mortonCode |= uint64_t(((floatBitsToInt(ib.z) >> i) & 1)) << (6 * i - 23); // max 61
    }

    for (int i = 14; i >= 5; --i) {
        mortonCode |= uint64_t(((floatBitsToInt(ib.x) >> i) & 1)) << (6 * i - 24); // max 60
        mortonCode |= uint64_t(((floatBitsToInt(ib.y) >> i) & 1)) << (6 * i - 25); // max 59
        mortonCode |= uint64_t(((floatBitsToInt(ib.z) >> i) & 1)) << (6 * i - 26); // max 58
    }

    mortonCode |= uint64_t(floatBitsToInt(ib.x) & 1);
    uint result = uint(mortonCode >> 32);
    return result;
}

uint SortingKeyEndPointEstimationAdaptive(vec3 origin, vec3 direction,float RayLengthLastPass)
{
    float estimatedRayLength;
    if(RayLengthLastPass == INFINITY)
    {
        float estimatedRayLength = 0.2 *computeLargestSceneExtent();
    }
    else 
    {
        float estimatedRayLength = 0.5* RayLengthLastPass;
    }
     //just hardcoded estimate atm
    vec3 a = origin;
    vec3 b = origin + estimatedRayLength * direction;


    //scale a and b to if necessary

    a = (a - vec3(0)) / vec3(1);
    b = (b - vec3(0)) / vec3(1);

    //calculate Morton Codes

    vec3 ia =  a * 32767.0f; //15b/dim
    vec3 ib = b * 32767.0f;  //15b/dim

    uint64_t mortonCode = 0;

    for (int i = 14; i >= 4; --i) {
        mortonCode |= uint64_t(((floatBitsToInt(ib.x) >> i) & 1)) << (6 * i - 21); // max 63
        mortonCode |= uint64_t(((floatBitsToInt(ib.y) >> i) & 1)) << (6 * i - 22); // max 62
        mortonCode |= uint64_t(((floatBitsToInt(ib.z) >> i) & 1)) << (6 * i - 23); // max 61
    }

    for (int i = 14; i >= 5; --i) {
        mortonCode |= uint64_t(((floatBitsToInt(ib.x) >> i) & 1)) << (6 * i - 24); // max 60
        mortonCode |= uint64_t(((floatBitsToInt(ib.y) >> i) & 1)) << (6 * i - 25); // max 59
        mortonCode |= uint64_t(((floatBitsToInt(ib.z) >> i) & 1)) << (6 * i - 26); // max 58
    }

    mortonCode |= uint64_t(floatBitsToInt(ib.x) & 1);
    uint result = uint(mortonCode >> 32);

    return result;
}








uint createSortingKey(uint sortingMode,PtPayload prd, Ray ray)
{
    uint code;
    if(sortingMode == eOrigin)
    {
        code = SortingKeyOrigin(ray.origin.xyz);
    }
    if(sortingMode == eReis)
    {
        code = SortingKeyReis(ray.origin.xyz,ray.direction.xyz);
    }
    if(sortingMode == eCosta)
    {
        code = SortingKeyCosta(ray.origin.xyz,ray.direction.xyz);
    }
    if(sortingMode == eAila)
    {
        code = SortingKeyAila(ray.origin.xyz,ray.direction.xyz);
    }
    if(sortingMode == eTwoPoint)
    {
        code = SortingKeyTwoPoint(ray.origin.xyz,ray.direction.xyz, prd.hitT);
    }
    if(sortingMode == eEndPointEst)
    {
        code = SortingKeyEndPointEstimationHard(ray.origin.xyz, ray.direction.xyz);
    }
    if(sortingMode == eEndEstAdaptive)
    {
        code = SortingKeyEndPointEstimationAdaptive(ray.origin.xyz, ray.direction.xyz, prd.hitT);
    }
    return code;
}


uint createSortingKeyFromParameters(Ray ray, SortingParameters parameters)
{
    uint resultCode = 0;
    uint originCode = 0;
    uint directionCode = 0;
    uint estEndCode = 0;
    uint realEndCode = 0;

    if(parameters.rayOrigin)
    {
        resultCode = SortingKeyOrigin(ray.origin.xyz);
    }
    if(parameters.rayDirection)
    {
        resultCode = (SortingKeyCosta(ray.origin.xyz,ray.direction.xyz) >> 24);
    }
    if(parameters.estimatedEndpoint)
    {
        resultCode = SortingKeyEndPointEstimationAdaptive(ray.origin.xyz, ray.direction.xyz, prd.hitT);
    }
    if(parameters.realEndpoint)
    {
        resultCode = SortingKeyTwoPoint(ray.origin.xyz,ray.direction.xyz, prd.hitT);
    }


    if(parameters.isFinished)
    {
        resultCode = resultCode & (prd.depth < (rtxState.maxDepth-1) ? 1 : 0);
    }

    return resultCode;


}

uint createSortingKeyFromSpecialization(Ray ray)
{
    uint resultCode = 0;
    uint originCode = 0;
    uint directionCode = 0;
    uint estEndCode = 0;
    uint realEndCode = 0;

    if(RAYORIGIN)
    {
        originCode = SortingKeyOrigin(ray.origin.xyz);
    }
    if(RAYDIRECTION)
    {
        directionCode = (SortingKeyCosta(ray.origin.xyz,ray.direction.xyz) >> 24);
    }
    if(ESTENDPOINT)
    {
        estEndCode = SortingKeyEndPointEstimationAdaptive(ray.origin.xyz, ray.direction.xyz, prd.hitT);
    }
    if(REALENDPOINT)
    {
        realEndCode = SortingKeyTwoPoint(ray.origin.xyz,ray.direction.xyz, prd.hitT);
    }


    if(ISFINISHED)
    {
        resultCode = resultCode & (prd.depth < (rtxState.maxDepth-1) ? 1 : 0);
    }

    return resultCode;


}





#endif