#pragma once
#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace eco
{
	typedef std::vector<std::vector<cv::Mat>> ECO_FEATS;
	typedef cv::Vec<float, 2> COMPLEX; // represent a complex number;

	// hog parameters cofiguration =========================================
	struct HogParameters
	{
		int cell_size = 6;
		int compressed_dim = 10; // Compressed dimensionality of each output layer (ECO Paper Table 1)
		int nOrients = 9;
		size_t nDim = 31; // Original dimension of feature
		float penalty = 0;
	};
	struct HogFeatures
	{
		HogParameters fparams;
		cv::Size img_input_sz;  // input sample size
		cv::Size img_sample_sz; // the size of sample
		cv::Size data_sz_block0;
	};

	//---------------------------
	struct CnParameters // only used for Color image
	{
		std::string tablename = "look_tables/CNnorm.txt";
		float table[32768][10];
		int cell_size = 4;
		int compressed_dim = 3;
		size_t nDim = 10;
		float penalty = 0;
	};
	struct CnFeatures
	{
		CnParameters fparams;
		cv::Size img_input_sz;
		cv::Size img_sample_sz;
		cv::Size data_sz_block0;
	};

	// Cojugate Gradient Options Structure =====================================
	struct CgOpts
	{
		bool debug;
		bool CG_use_FR;
		float tol;
		bool CG_standard_alpha;
		float init_forget_factor;
		int maxit;
	};

	// Parameters set exactly the same as 'testing_ECO_HC.m'====================
	struct EcoParameters
	{
		// Features
		bool useDeepFeature = false;
		bool useHogFeature = true;
		bool useColorspaceFeature = false;// not implemented yet
		bool useCnFeature = true;
		bool useIcFeature = false;

		HogFeatures hog_features;
		//ColorspaceFeatures colorspace_feature;
		CnFeatures cn_features;
		//IcFeatures ic_features;

		// extra parameters
		CgOpts CG_opts;
		float max_score_threshhold = 0.2;

		// Global feature parameters1s
		int normalize_power = 2;
		bool normalize_size = true;
		bool normalize_dim = true;

		// img sample parameters
		std::string search_area_shape = "square"; // The shape of the samples
		float search_area_scale = 5.0;		 // The scaling of the target size to get the search area  缩放目标大小以获得搜索区域
		int min_image_sample_size = 22500;   // Minimum area of image samples, 200x200
		int max_image_sample_size = 40000;   // Maximum area of image samples, 250x250

		// Detection parameters
		int refinement_iterations = 1; // Number of iterations used to refine the resulting position in a frame 用于优化帧中结果位置的迭代次数
		int newton_iterations = 5;	 // The number of Newton iterations used for optimizing the detection score 用于优化检测分数的牛顿迭代次数
		bool clamp_position = false;   // Clamp the target position to be inside the image 将目标位置夹紧在图像内部

		// Learning parameters
		float output_sigma_factor = 1.0f / 16.0f; // Label function sigma
		float learning_rate = 0.009; // Learning rate
		size_t nSamples = 30; // Maximum number of stored training samples
		std::string sample_replace_strategy = "lowest_prior"; // Which sample to replace when the memory is full
		bool lt_size = 0; // The size of the long - term memory(where all samples have equal weight) 长期记忆的大小（所有样本的权重相等）
		int train_gap = 5; // The number of intermediate frames with no training(0 corresponds to training every frame) 没有训练的中间帧数（0对应于每帧训练）
		int skip_after_frame = 10; // After which frame number the sparse update scheme should start(1 is directly)  在该帧编号之后，稀疏更新方案应开始（直接为1）
		bool use_detection_sample = true; // Use the sample that was extracted at the detection stage also for learning  使用在检测阶段提取的样本也用于学习

		// Factorized convolution parameters
		bool use_projection_matrix = true;	// Use projection matrix, i.e. use the factorized convolution formulation 使用投影矩阵，即使用因子化卷积公式
		bool update_projection_matrix = true; // Whether the projection matrix should be optimized or not 是否应优化投影矩阵
		std::string proj_init_method = "pca"; // Method for initializing the projection matrix
		float projection_reg = 1e-7; // Regularization paremeter of the projection matrix (lambda)  投影矩阵的正则化参数（lambda）

		// Generative sample space model parameters
		bool use_sample_merge = true; // Use the generative sample space model to merge samples 使用生成样本空间模型合并样本
		std::string sample_merge_type = "Merge"; // Strategy for updating the samples  更新样本的策略
		std::string distance_matrix_update_type = "exact"; // Strategy for updating the distance matrix 更新距离矩阵的策略

		// Conjugate Gradient parameters
		int CG_iter = 5; // The number of Conjugate Gradient iterations in each update after the first frame 第一帧后每次更新中的共轭梯度迭代次数
		int init_CG_iter = 10 * 15; // The total number of Conjugate Gradient iterations used in the first frame 第一帧中使用的共轭梯度迭代的总数
		int init_GN_iter = 10; // The number of Gauss-Newton iterations used in the first frame(only if the projection matrix is updated) 第一帧中使用的高斯-牛顿迭代次数（仅当投影矩阵更新时）
		bool CG_use_FR = false; // Use the Fletcher-Reeves(true) or Polak-Ribiere(false) formula in the Conjugate Gradient  在共轭梯度中使用Fletcher-Reeves（真）或Polak-Ribiere（假）公式
		bool CG_standard_alpha = true;  // Use the standard formula for computing the step length in Conjugate Gradient 使用标准公式计算共轭梯度中的步长
		int CG_forgetting_rate = 50; // Forgetting rate of the last conjugate direction 最后共轭方向的遗忘率
		float precond_data_param = 0.75; // Weight of the data term in the preconditioner 预处理器中数据项的权重
		float precond_reg_param = 0.25; // Weight of the regularization term in the preconditioner 预处理器中正则化项的权重
		int precond_proj_param = 40; // Weight of the projection matrix part in the preconditioner

		// Regularization window parameters
		bool use_reg_window = true; // Use spatial regularization or not  是否使用空间正则化
		double reg_window_min = 1e-4; // The minimum value of the regularization window
		double reg_window_edge = 10e-3; // The impact of the spatial regularization  空间正则化的影响
		size_t reg_window_power = 2; // The degree of the polynomial to use(e.g. 2 is a quadratic window)  使用多项式的次数（例如，2是二次窗）
		float reg_sparsity_threshold = 0.05; // A relative threshold of which DFT coefficients that should be set to zero DFT系数应设置为零的相对阈值

		// Interpolation parameters
		std::string interpolation_method = "bicubic"; // The kind of interpolation kernel  插值核的类型
		float interpolation_bicubic_a = -0.75;   // The parameter for the bicubic interpolation kernel 双三次插值核的参数
		bool interpolation_centering = true; // Center the kernel at the feature sample  将内核放在特征样本的中心
		bool interpolation_windowing = false; // Do additional windowing on the Fourier coefficients of the kernel 对核的傅立叶系数进行额外的加窗

		// Scale parameters for the translation model  转换模型的比例参数
		// Only used if: use_scale_filter = false
		size_t number_of_scales = 7; // Number of scales to run the detector 运行探测器的刻度数
		float scale_step = 1.01f; // The scale factor
		float min_scale_factor;
		float max_scale_factor;

		// Scale filter parameters
		// Only used if: use_scale_filter = true
		bool use_scale_filter = true; // Use the fDSST scale filter or not (for speed) 是否使用fDSST刻度过滤器（用于速度）
		float scale_sigma_factor = 1.0f / 16.0f; // Scale label function sigma
		float scale_learning_rate = 0.025;		 // Scale filter learning rate
		int number_of_scales_filter = 17;		 // Number of scales
		int number_of_interp_scales = 33;		 // Number of interpolated scales
		float scale_model_factor = 1.0;			 // Scaling of the scale model
		float scale_step_filter = 1.02;			 // The scale factor for the scale filter
		float scale_model_max_area = 32 * 16;	 // Maximume area for the scale sample patch 比例样本块的最大面积
		std::string scale_feature = "HOG4";			 // Features for the scale filter (only HOG4 supported)  尺度过滤器的功能（仅支持HOG4）
		int s_num_compressed_dim = 17;	         // Number of compressed feature dimensions in the scale filter  比例过滤器中的压缩特征尺寸数
		float lambda = 1e-2;					 // Scale filter regularization  尺度滤波器正则化
		float do_poly_interp = true;			 // Do 2nd order polynomial interpolation to obtain more accurate scale  进行二阶多项式插值以获得更精确的比例
		cv::Size scale_model_sz;

		bool debug = false; // to show heatmap or not

	// GPU
		bool use_gpu = false; // whether Caffe use gpu or not
		int gpu_id = 0;
	};
}
