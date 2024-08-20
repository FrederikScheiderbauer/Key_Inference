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

        uint mortonCode = 0;
        for (int i = 22; i >= 2; --i) {
            mortonCode |= ((floatBitsToInt(ia.x) >> i) & 1) << (3 * i - 3); // max 63
            mortonCode |= ((floatBitsToInt(ia.y) >> i) & 1) << (3 * i - 4); // max 62
            mortonCode |= ((floatBitsToInt(ia.z) >> i) & 1) << (3 * i - 5); // max 61
        }
        mortonCode |= (floatBitsToInt(ia.x) & 1); // max 0

        return mortonCode;
}
uint SortingKeyOrigin64(vec3 origin)
{
// Point for Morton codes.
        vec3 a = (origin.xyz - vec3(0)) / vec3(1);
        vec3 ia = a * 8388607.0f; //23b/dim

        uint mortonCode = 0;
        for (int i = 22; i >= 2; --i) {
            mortonCode |= ((floatBitsToInt(ia.x) >> i) & 1) << (3 * i - 3); // max 63
            mortonCode |= ((floatBitsToInt(ia.y) >> i) & 1) << (3 * i - 4); // max 62
            mortonCode |= ((floatBitsToInt(ia.z) >> i) & 1) << (3 * i - 5); // max 61
        }
        mortonCode |= (floatBitsToInt(ia.x) & 1); // max 0

        return mortonCode;
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

        uint mortonCode = 0;

        for (int i = 7; i >= 1; --i) {
            mortonCode |= ((floatBitsToInt(ia.x) >> i) & 1) << (3 * i + 10); // max 31
            mortonCode |= ((floatBitsToInt(ia.y) >> i) & 1) << (3 * i + 9); // max 30
            mortonCode |= ((floatBitsToInt(ia.z) >> i) & 1) << (3 * i + 8); // max 29
        }
        mortonCode |= (floatBitsToInt(ia.x) & 1) << (10); // max 10

        for (int i = 7; i >= 3; --i) {
            mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (2 * i - 5); // max 9
            mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (2 * i - 6); // max 8
        }

        // Output key.
        return mortonCode;
}
//same as above, but produces 64 bit encoding
uint SortingKeyReis64(vec3 origin, vec3 direction)
{
        // Point for Morton codes.
        vec3 a = (origin.xyz - vec3(0)) / vec3(1);
        vec3 nd = normalize(direction);
        vec3 b;
        b.x = atan(nd.y, nd.x) / (2.0f * M_PI) + 0.5f;
        b.y = acos(nd.z) / M_PI;

        vec3 ia = a * 2097151.0f; //8b/dim
        vec3 ib = b * 2097151.0f;  //8b/dim

        uint mortonCode = 0;

        for (int i = 20; i >= 14; --i) {
            mortonCode |= ((floatBitsToInt(ia.x) >> i) & 1) << (3 * i + 3); // max 63
            mortonCode |= ((floatBitsToInt(ia.y) >> i) & 1) << (3 * i + 2); // max 62
            mortonCode |= ((floatBitsToInt(ia.z) >> i) & 1) << (3 * i + 1); // max 61
        }
        mortonCode |= ((floatBitsToInt(ia.x)>>13) & 1) << (42); // max 42

        for (int i = 7; i >= 3; --i) {
            mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (2 * i + 1); // max 41
            mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (2 * i ); // max 40
        }

        // Output key.
        return mortonCode;
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

        uint mortonCode = 0;

        for (int i = 12; i >= 9; --i) {
            mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (2 * i +39); // max 9
            mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (2 * i +38); // max 8
        }

        for (int i = 7; i >= 1; --i) {
            mortonCode |= ((floatBitsToInt(ia.x) >> i) & 1) << (3 * i + 19); // max 31
            mortonCode |= ((floatBitsToInt(ia.y) >> i) & 1) << (3 * i + 18); // max 30
            mortonCode |= ((floatBitsToInt(ia.z) >> i) & 1) << (3 * i + 17); // max 29
        }

        // Output key.
        return mortonCode;
}

//as above, but oroduces 64 bit encoding
uint SortingKeyCosta64(vec3 origin, vec3 direction)
{
        // Point for Morton codes.
        vec3 a = (origin.xyz - vec3(0)) / vec3(1);
        vec3 nd = normalize(direction);
        vec3 b;
        b.x = atan(nd.y, nd.x) / (2.0f * M_PI) + 0.5f;
        b.y = acos(nd.z) / M_PI;

        vec3 ia = a * 32767.0f; //13b/dim
        vec3 ib = b * 32767.0f;  //13b/dim

        uint mortonCode = 0;

        for (int i = 14; i >= 11; --i) {
            mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (2 * i +35); // max 9
            mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (2 * i +34); // max 8
        }

        for (int i = 14; i >= 0; --i) {
            mortonCode |= ((floatBitsToInt(ia.x) >> i) & 1) << (3 * i + 13); // max 31
            mortonCode |= ((floatBitsToInt(ia.y) >> i) & 1) << (3 * i + 12); // max 30
            mortonCode |= ((floatBitsToInt(ia.z) >> i) & 1) << (3 * i + 11); // max 29
        }

        // Output key.
        return mortonCode;
}

//Origin Direction interleaved

uint SortingKeyAila(vec3 origin, vec3 direction)
{
    // Point for Morton codes.
    vec3 a = (origin.xyz - vec3(0)) / vec3(1);
    vec3 b = (normalize(direction) +1.0f) *0.5f;

    vec3 ia = a * 8191.0f; //13b/dim
    vec3 ib = b * 8191.0f;  //13b/dim
    
    uint mortonCode = 0;

    mortonCode |= ((floatBitsToInt(ia.x) >> 12) & 1) << (63); // max 63
    mortonCode |= ((floatBitsToInt(ia.y) >> 12) & 1) << (62); // max 62
    mortonCode |= ((floatBitsToInt(ia.z) >> 12) & 1) << (61); // max 61

    mortonCode |= ((floatBitsToInt(ia.x)>> 11) & 1) << (60); // max 60
    mortonCode |= ((floatBitsToInt(ia.y) >> 11) & 1) << (59); // max 59
    mortonCode |= ((floatBitsToInt(ia.z) >> 11) & 1) << (58); // max 58

    mortonCode |= ((floatBitsToInt(ia.x) >> 10) & 1) << (57); // max 57
    mortonCode |= ((floatBitsToInt(ia.y) >> 10) & 1) << (56); // max 56
    mortonCode |= ((floatBitsToInt(ia.z) >> 10) & 1) << (55); // max 55

    for (int i = 9; i >= 1; --i) {
        mortonCode |= ((floatBitsToInt(ia.x) >> i) & 1) << (6 * i + 0); // max 54
        mortonCode |= ((floatBitsToInt(ia.y) >> i) & 1) << (6 * i - 1); // max 53
        mortonCode |= ((floatBitsToInt(ia.z) >> i) & 1) << (6 * i - 2); // max 52
    }

    for (int i = 12; i >= 4; --i) {
        mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (6 * i - 21); // max 51
        mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (6 * i - 22); // max 50
        mortonCode |= ((floatBitsToInt(ib.z) >> i) & 1) << (6 * i - 23); // max 49
    }

    return mortonCode >> 32;
}

//Origin Direction interleaved
//as above, but produces 64 bit encoding
uint SortingKeyAila64(vec3 origin, vec3 direction)
{
    // Point for Morton codes.
    vec3 a = (origin.xyz - vec3(0)) / vec3(1);
    vec3 b = (normalize(direction) +1.0f) *0.5f;

    vec3 ia = a * 8191.0f; //13b/dim
    vec3 ib = b * 8191.0f;  //13b/dim
    
    uint mortonCode = 0;

    mortonCode |= ((floatBitsToInt(ia.x) >> 12) & 1) << (63); // max 63
    mortonCode |= ((floatBitsToInt(ia.y) >> 12) & 1) << (62); // max 62
    mortonCode |= ((floatBitsToInt(ia.z) >> 12) & 1) << (61); // max 61

    mortonCode |= ((floatBitsToInt(ia.x)>> 11) & 1) << (60); // max 60
    mortonCode |= ((floatBitsToInt(ia.y) >> 11) & 1) << (59); // max 59
    mortonCode |= ((floatBitsToInt(ia.z) >> 11) & 1) << (58); // max 58

    mortonCode |= ((floatBitsToInt(ia.x) >> 10) & 1) << (57); // max 57
    mortonCode |= ((floatBitsToInt(ia.y) >> 10) & 1) << (56); // max 56
    mortonCode |= ((floatBitsToInt(ia.z) >> 10) & 1) << (55); // max 55

    for (int i = 9; i >= 1; --i) {
        mortonCode |= ((floatBitsToInt(ia.x) >> i) & 1) << (6 * i + 0); // max 54
        mortonCode |= ((floatBitsToInt(ia.y) >> i) & 1) << (6 * i - 1); // max 53
        mortonCode |= ((floatBitsToInt(ia.z) >> i) & 1) << (6 * i - 2); // max 52
    }

    for (int i = 12; i >= 4; --i) {
        mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (6 * i - 21); // max 51
        mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (6 * i - 22); // max 50
        mortonCode |= ((floatBitsToInt(ib.z) >> i) & 1) << (6 * i - 23); // max 49
    }

    return mortonCode;
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

    uint mortonCode = 0;

    for (int i = 14; i >= 4; --i) {
        mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (6 * i - 21); // max 63
        mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (6 * i - 22); // max 62
        mortonCode |= ((floatBitsToInt(ib.z) >> i) & 1) << (6 * i - 23); // max 61
    }

    for (int i = 14; i >= 5; --i) {
        mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (6 * i - 24); // max 60
        mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (6 * i - 25); // max 59
        mortonCode |= ((floatBitsToInt(ib.z) >> i) & 1) << (6 * i - 26); // max 58
    }

    mortonCode |= floatBitsToInt(ib.x) & 1;

    return mortonCode;
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

    uint mortonCode = 0;

    for (int i = 14; i >= 4; --i) {
        mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (6 * i - 21); // max 63
        mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (6 * i - 22); // max 62
        mortonCode |= ((floatBitsToInt(ib.z) >> i) & 1) << (6 * i - 23); // max 61
    }

    for (int i = 14; i >= 5; --i) {
        mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (6 * i - 24); // max 60
        mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (6 * i - 25); // max 59
        mortonCode |= ((floatBitsToInt(ib.z) >> i) & 1) << (6 * i - 26); // max 58
    }

    mortonCode |= floatBitsToInt(ib.x) & 1;

    return mortonCode;
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

    uint mortonCode = 0;

    for (int i = 14; i >= 4; --i) {
        mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (6 * i - 21); // max 63
        mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (6 * i - 22); // max 62
        mortonCode |= ((floatBitsToInt(ib.z) >> i) & 1) << (6 * i - 23); // max 61
    }

    for (int i = 14; i >= 5; --i) {
        mortonCode |= ((floatBitsToInt(ib.x) >> i) & 1) << (6 * i - 24); // max 60
        mortonCode |= ((floatBitsToInt(ib.y) >> i) & 1) << (6 * i - 25); // max 59
        mortonCode |= ((floatBitsToInt(ib.z) >> i) & 1) << (6 * i - 26); // max 58
    }

    mortonCode |= floatBitsToInt(ib.x) & 1;

    return mortonCode;
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


#endif