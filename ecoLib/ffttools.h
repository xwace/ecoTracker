#pragma once
#include <opencv2/imgproc/imgproc.hpp>

namespace eco
{
	cv::Mat dft(const cv::Mat img_org, const bool backwards = false);
	cv::Mat fftshift(const cv::Mat img_org,
		const bool rowshift = true,
		const bool colshift = true,
		const bool reverse = 0);

	cv::Mat real(const cv::Mat img);
	cv::Mat imag(const cv::Mat img);
	cv::Mat magnitude(const cv::Mat img);
	cv::Mat complexDotMultiplication(const cv::Mat &a, const cv::Mat &b);
	cv::Mat complexDotMultiplicationCPU(const cv::Mat &a, const cv::Mat &b);

	cv::Mat complexDotDivision(const cv::Mat a, const cv::Mat b);
	cv::Mat complexMatrixMultiplication(const cv::Mat &a, const cv::Mat &b);
	cv::Mat complexConvolution(const cv::Mat a_input,
		const cv::Mat b_input,
		const bool valid = 0);

	cv::Mat real2complex(const cv::Mat &x);
	cv::Mat mat_conj(const cv::Mat &org);
	float mat_sum_f(const cv::Mat &org);
	double mat_sum_d(const cv::Mat &org);

	inline bool SizeCompare(cv::Size &a, cv::Size &b)
	{
		return a.height < b.height;
	}

	inline void rot90(cv::Mat &matImage, int rotflag)
	{
		if (rotflag == 1)
		{
			cv::transpose(matImage, matImage);
			cv::flip(matImage, matImage, 1); // flip around y-axis
		}
		else if (rotflag == 2)
		{
			cv::transpose(matImage, matImage);
			cv::flip(matImage, matImage, 0); // flip around x-axis
		}
		else if (rotflag == 3)
		{
			cv::flip(matImage, matImage, -1); // flip around both axis
		}
		else if (rotflag != 0) // 0: keep the same
		{
			assert(0 && "error: unknown rotation flag!");
		}
	}
}
