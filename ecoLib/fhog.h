#pragma once
#include <stdio.h>

#include <opencv2/core/types_c.h>

namespace eco
{
	// DataType: STRUCT featureMap
	// FEATURE MAP DESCRIPTION
	//   Rectangular map (sizeX x sizeY),
	//   every cell stores feature vector (dimension = numFeatures)
	// map             - matrix of feature vectors
	//                   to set and get feature vectors (i,j)
	//                   used formula map[(j * sizeX + i) * p + k], where
	//                   k - component of feature vector in cell (i, j)
	typedef struct
	{
		int sizeX;
		int sizeY;
		int numFeatures;
		float *map;
	} CvLSVMFeatureMapCaskade;

#include "float.h"

#define PI CV_PI
constexpr auto EPS = 0.000001;
#define F_MAX FLT_MAX
#define F_MIN -FLT_MAX

	// The number of elements in bin
	// The number of sectors in gradient histogram building
constexpr auto NUM_SECTOR = 9;

// The number of levels in image resize procedure
// We need Lambda levels to resize image twice
constexpr auto LAMBDA = 10;

// Block size. Used in feature pyramid building procedure
constexpr auto SIDE_LENGTH = 8;

constexpr auto VAL_OF_TRUNCATE = 0.2f;

//modified from "_lsvm_error.h"
constexpr auto LATENT_SVM_OK = 0;
constexpr auto LATENT_SVM_MEM_NULL = 2;
constexpr auto DISTANCE_TRANSFORM_OK = 1;
constexpr auto DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR = -1;
constexpr auto DISTANCE_TRANSFORM_ERROR = -2;
constexpr auto DISTANCE_TRANSFORM_EQUAL_POINTS = -3;
constexpr auto LATENT_SVM_GET_FEATURE_PYRAMID_FAILED = -4;
constexpr auto LATENT_SVM_SEARCH_OBJECT_FAILED = -5;
constexpr auto LATENT_SVM_FAILED_SUPERPOSITION = -6;
constexpr auto FILTER_OUT_OF_BOUNDARIES = -7;
constexpr auto LATENT_SVM_TBB_SCHEDULE_CREATION_FAILED = -8;
constexpr auto LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT = -9;
constexpr auto FFT_OK = 2;
constexpr auto FFT_ERROR = -10;
constexpr auto LSVM_PARSER_FILE_NOT_FOUND = -11;
	int getFeatureMaps(const IplImage* image, const int k, CvLSVMFeatureMapCaskade** map);
	int normalizeAndTruncate(CvLSVMFeatureMapCaskade *map, const float alfa);

	int PCAFeatureMaps(CvLSVMFeatureMapCaskade *map);

	int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj,
		const int sizeX, const int sizeY, const int p);

	int freeFeatureMapObject(CvLSVMFeatureMapCaskade **obj);
} // namespace eco
