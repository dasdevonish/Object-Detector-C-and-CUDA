#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <dirent.h>
#include <svm.h>

#define PI   3.14159265358979323846
#define cellSize 8
#define bin_size 20
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

char *HOG_function_train(IplImage *input, int mode);
double *HOG_function_predict(IplImage *input, int mode);
void create_dataset_in_svm_format(char **path[], int num_of_svm_classes, int mode);
int count_files_in_directory(char *path);
void apply_HOG_descriptor(int num_of_folders, char *path, int HOG_label, int mode);
void write_label(int HOG_label);
char *merge_path_and_file_name(char *s1, char *s2);
void train_svm(void);
void run_object_detector(int stepSize, int windowSize, int mode, double scale);
IplImage* img_resize(IplImage* src_img, int new_width, int new_height);
void sliding_window(IplImage *input, IplImage **input_all_images, int stepSize, int windowSize, int pyramid_iteration_count, short **position_window_x, short **position_window_y);
void classify_svm(void);
void store_bounding_box_points(int windowSize, double scale, int pyramid_num, int position_window_x, int position_window_y, CvPoint *pt1, CvPoint *pt2, int *p_prediction_count);
int my_SVM_prediction(double *HOG_features, struct svm_model* model);

double my_svm_predict(struct svm_model* model, struct svm_node *x);

double my_svm_predict_values(struct svm_model *model, struct svm_node *x, double* dec_values);

double my_k_function(struct svm_node *x, struct svm_node *y, struct svm_parameter param);

void draw_bounding_box(IplImage *output, CvPoint *pt1, CvPoint *pt2, int prediction_count);
void enterGPU(IplImage **input, int mode, struct svm_model* model, int total_num_of_windows, int *window_array, int *p_prediction_count, int windowSize, double scale, short **position_window_x, short **position_window_y, CvPoint *pt1, CvPoint *pt2);


void create_dataset_in_svm_format(char **path[], int num_of_folders, int mode)
{
	printf("Creating a dataset in the SVM format...\n");
	/*Count number of files in directory and assign HOG label per folder*/
	int HOG_label = 1;
	int number_of_files_per_directory[2] = { 0 };
	int total_files_in_directory = 0;
	int i;
	for (i = 0; i < num_of_folders; i++)
	{
		number_of_files_per_directory[i] = count_files_in_directory(*path[i]);
		total_files_in_directory += number_of_files_per_directory[i];
		if (i == (num_of_folders - 1)) 
		{
			HOG_label = -1;
		}
		/*Apply the HOG descriptor per folder*/
		printf("Opening Folder %u...\n", i + 1);
		printf("%u images found in folder...\nApplying HOG algorithm to images\n", number_of_files_per_directory[i]);
		apply_HOG_descriptor(num_of_folders, *path[i], HOG_label, mode);
	}
	//printf("%d", total_files_in_directory);
}
int count_files_in_directory(char *path)
{
	int file_count = 0;
	DIR *p_dir;
	struct dirent *entry;
	p_dir = opendir(path);
	if (!p_dir) 
	{
		printf("opendir() failed! Does it exist?\n");
		return 1;
	}
	while ((entry = readdir(p_dir)) != NULL)
	{
		if (entry->d_type == DT_REG) /* If the entry is a regular file */
			file_count++;
	}
	closedir(p_dir);
	return file_count;
}
void apply_HOG_descriptor(int num_of_folders, char *path, int HOG_label, int mode)
{
	DIR *p_dir;
	struct dirent *entry;
	p_dir = opendir(path);
	if (p_dir)
	{
		/*Iterate through folder*/
		while ((entry = readdir(p_dir)) != NULL)
		{
			if (entry->d_type == DT_REG) /* If the entry is a regular file */
			{
				write_label(HOG_label);
				char *p_path_name = merge_path_and_file_name(path, entry->d_name); 
				IplImage *input = cvLoadImage(p_path_name, CV_LOAD_IMAGE_GRAYSCALE);	
				free(p_path_name);

				if (!input)
				{
					printf("Image can NOT Load!!!\n");
					exit (1);
				}
				//Apply HOG descriptor per image*/
				HOG_function_train(input, mode);

				cvReleaseImage(&input);
				//cvWaitKey(0);
				//cvDestroyAllWindows();
				
			}		
		}
		closedir(p_dir);
	}
}
void write_label(int HOG_label)
{
	FILE *fptr;
	fptr = fopen("data\\SVM_Format\\ObjectDetectionTrain.svm", "a");
	if (fptr == NULL)
	{
		printf("Error!");
		exit(1);
	}
	fprintf(fptr, "%d", HOG_label);
	//printf("%d", HOG_label);
	fclose(fptr);

}
char *merge_path_and_file_name(char *s1, char *s3)
{
	char *s2 = "\\";
	size_t n1 = strlen(s1);
	size_t n2 = strlen(s2);
	char *p = (char *)malloc(n1 + n2 + strlen(s3) + 1);
	if (p)
	{
		strcpy(p, s1);
		strcpy(p + n1, s2);
		strcpy(p + n1 + n2, s3);
	}
	return p;

}
char *HOG_function_train(IplImage *input, int mode)
{
	int i;
	int j;
	int k;

	/*Create 4 images and initialise to 0*/
	IplImage *dx = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);
	IplImage *dy = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);
	IplImage *dxy = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);
	IplImage *theta = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);
	uchar *pdx = (uchar*)dx->imageData;
	uchar *pdy = (uchar*)dy->imageData;
	uchar *pdxy = (uchar*)dxy->imageData;
	uchar *ptheta = (uchar*)theta->imageData;

	for (i = 0; i < input->height; i++)
		for (j = 0; j < input->width; j++)
		{
			pdx[i*input->widthStep + j] = 0;
			pdy[i*input->widthStep + j] = 0;
			pdxy[i*input->widthStep + j] = 0;
			ptheta[i*input->widthStep + j] = 0;
		}
	/*Find gradient in x and y direction*/
	cvSobel(input, dx, 1, 0, 3);
	cvSobel(input, dy, 0, 1, 3);

	/*Calculate magnitudes and angles*/
	for (i = 0; i < input->height; i++)
		for (j = 0; j < input->width; j++)
		{
			pdxy[i*input->widthStep + j] = (uchar)sqrt((pdx[i*input->widthStep + j] * pdx[i*input->widthStep + j]) + (pdy[i*input->widthStep + j] * pdy[i*input->widthStep + j]));
			ptheta[i*input->widthStep + j] = (uchar)(atan2(pdy[i*input->widthStep + j], pdx[i*input->widthStep + j]) * (180 / PI));
			if (ptheta[i*input->widthStep + j] < 0)
				ptheta[i*input->widthStep + j] = ptheta[i*input->widthStep + j] + 180;
			//may use round function instead of casting directly to int
		}



	/*Calculate the number of cells rows and columns for image*/
	int row = input->height;
	int cell_counti = (int)floor(row / cellSize);
	int col = input->width;
	int cell_countj = (int)floor(col / cellSize);

	/*Declare variables that dictate start and end of each cell*/
	/*start_i, end_i, start_j, end_j are absolute pixel postion of the entire image*/
	/*i - start_i and j - start_j are the relative positive, i.e position will range 0 - 7 for temporary structure.*/
	int start_i, end_i, start_j, end_j;
	double angleratio = 0;

	/*3D histogram array and initialise to zero*/
	double* histogram = (double*)malloc(cell_counti * cell_countj * 9 * sizeof(double));
	if (histogram == NULL)
	{
		printf("Out of memory");
		exit(0);
	}
	for (i = 0; i < cell_counti; i++)
		for (j = 0; j < cell_countj; j++)
			for (k = 0; k < 9; k++)
				*(histogram + i * cell_countj * 9 + j * 9 + k) = 0;

	/*Iterate through each cell group*/
	int cell_i;
	int cell_j;
	for (cell_i = 0; cell_i < cell_counti; cell_i++)
	{
		for (cell_j = 0; cell_j < cell_countj; cell_j++)
		{
			/*Set absolute pixel locations from cell group*/
			start_i = (cell_i)*cellSize;
			end_i = (cell_i + 1)*cellSize - 1;
			start_j = (cell_j)*cellSize;
			end_j = (cell_j + 1)*cellSize - 1;

			/*Orientation Binning*/
			for (i = start_i; i <= end_i; i++)
			{
				for (j = start_j; j <= end_j; j++)
				{
					angleratio = ptheta[i*input->widthStep + j] / (double)bin_size;
					/*Separate integer and decimal part*/
					int angleint = (int)angleratio;
					double angledec = angleratio - angleint;
					/*Put magnitude in appropriate bin*/
					if (angleint > 7)
					{
						*(histogram + cell_i * cell_countj * 9 + cell_j * 9 + 0) += pdxy[i*input->widthStep + j] * (angledec);
					}
					else
					{
						*(histogram + cell_i * cell_countj * 9 + cell_j * 9 + angleint + 1) += pdxy[i*input->widthStep + j] * (angledec);
					}
					*(histogram + cell_i * cell_countj * 9 + cell_j * 9 + angleint) += pdxy[i*input->widthStep + j] * (1 - angledec);
				}
			}
		}
	}

	int cell_block = 4;
	int block_counti = cell_counti - 1;
	int block_countj = cell_countj - 1;
	int fstart = 0;

	/*Create array to store entire HOG features vector*/
	double* HOG_features = (double*)malloc(block_counti * block_countj * 36 * sizeof(double));
	char *HOG_features_char = NULL;
	if (HOG_features == NULL)
	{
		printf("Out of memory");
		exit(0);
	}
	if (mode == 0)
		;
	else
	{
		HOG_features_char = (char *)malloc(block_counti * block_countj * 36 * 2 * sizeof(double));
		if (HOG_features_char == NULL)
		{
			printf("Out of memory");
			exit(0);
		}
	}

	for (i = 0; i < block_counti*block_countj * 36; i++)
		*(HOG_features + i) = 0;

	/*Iterate through all the blocks*/
	int block_i;
	int block_j;
	for (block_i = 0; block_i < block_counti; block_i++)
	{
		for (block_j = 0; block_j < block_countj; block_j++)
		{
			double len_block = 0;
			/*Create 3D dynamic array to store single block features*/
			double* histo_block = (double*)malloc(36 * sizeof(double));
			if (histo_block == NULL)
			{
				printf("Out of memory");
				exit(0);
			}
			for (i = 0; i < 36; i++)
				*(histo_block + i) = 0;

			/*Create array to store normalized single block features*/
			double* norm_block = (double*)malloc(36 * sizeof(double));
			if (norm_block == NULL)
			{
				printf("Out of memory");
				exit(0);
			}
			for (i = 0; i < 36; i++)
				*(norm_block + i) = 0;

			/*Iterate through each single block*/
			for (i = 0; i < 2; i++)
			{
				for (j = 0; j < 2; j++)
				{
					int cell_i = block_i + i;
					int cell_j = block_j + j;
					int cell_b_count = 2 * i + j;
					/*Copy cell features to block features*/
					int ii;
					for (ii = 0; ii < 9; ii++)
					{
						*(histo_block + (ii + cell_b_count * 9)) = *(histogram + cell_i * cell_countj * 9 + cell_j * 9 + ii);
						len_block += *(histogram + cell_i * cell_countj * 9 + cell_j * 9 + ii) * *(histogram + cell_i * cell_countj * 9 + cell_j * 9 + ii);
					}
				}
			}
			/*Calculate the normalized block*/
			/*There is a problem is len_block is equal to zero. Program gives invalid values*/
			len_block = sqrt(len_block);
			if (len_block != 0)
			{
				for (i = 0; i < 36; i++)
					*(norm_block + i) = *(histo_block + i) / len_block;
				int jj;
				for (jj = 0; jj < 36; jj++)
				{
					*(HOG_features + jj + fstart) = *(norm_block + jj);
				}

			}
			fstart += 36;

			free(histo_block);
			free(norm_block);
		}
	}
	free(histogram);

	/*Open appropriate file*/
	if (mode == 0)
	{
		FILE *fptr;
		fptr = fopen("data\\SVM_Format\\ObjectDetectionTrain.svm", "a");
		if (fptr == NULL)
		{
			printf("Error!");
			exit(1);
		}
		/*Write to HOG features to a file*/
		int feature_number = 0;
		for (i = 0; i < (36 * block_counti * block_countj); i++)
		{

			fprintf(fptr, " %d:%f", feature_number + 1, *(HOG_features + feature_number));
			//printf(" %d:%f", feature_number + 1 , *(HOG_features + feature_number));
			feature_number++;
		}
		fprintf(fptr, "\n");
		//printf("\n");
		fclose(fptr);

	}
	else
	{
		sprintf(HOG_features_char, "0");
		/*Store HOG features to a string*/
		int feature_number = 0;
		/*size of double is 8 bytes, size of char is 1 byte. 1 double holds 8 characters*/
		char *buffer = (char *)malloc(2 * sizeof(double));
		//free(buffer);
		for (i = 0; i < (36 * block_counti * block_countj); i++)
		{
			sprintf(buffer, " %d:%f", feature_number + 1, *(HOG_features + feature_number));
			strcat(HOG_features_char, buffer);
			feature_number++;
		}
		free(buffer);
		//free(HOG_features_char);
	}
	/*Release memory*/
	free(HOG_features);

	cvReleaseImage(&dx);
	cvReleaseImage(&dy);
	cvReleaseImage(&dxy);
	cvReleaseImage(&theta);

	return HOG_features_char;
}

double *HOG_function_predict(IplImage *input, int mode)
{
	//double* exit_point = (double*)malloc(1 * sizeof(double));
	//*exit_point = ((double)(1));
	

	int i;
	int j;
	int k;
	uchar *pinput = (uchar*)input->imageData;

	uchar *pdxy = (uchar*)malloc(input->width * input->height * sizeof(uchar));
	uchar *ptheta = (uchar*)malloc(input->width * input->height * sizeof(uchar));

	for (i = 0; i < input->height; i++)
		for (j = 0; j < input->width; j++)
		{
			pdxy[i*input->widthStep + j] = 0;
			ptheta[i*input->widthStep + j] = 0;
		}

	float kernelx[3][3] = { {-1, 0, 1},
						   {-2, 0, 2},
						   {-1, 0, 1} };
	float kernely[3][3] = { {-1, -2, -1},
							{0,  0,  0},
							{1,  2,  1} };

	short mag_x = 0;
	short mag_y = 0;
	short x, y;
	

	for (y = 1; y < input->height - 1; y++)
	{
		for (x = 1; x < input->width - 1; x++)
		{
			mag_x = (short)((kernelx[0][0] * pinput[((y - 1)*input->widthStep) + (x - 1)])
				+ (kernelx[0][1] * pinput[((y - 1)*input->widthStep) + (x)])
				+ (kernelx[0][2] * pinput[((y - 1)*input->widthStep) + (x + 1)])
				+ (kernelx[1][0] * pinput[((y)*input->widthStep) + (x - 1)])
				+ (kernelx[1][1] * pinput[((y)*input->widthStep) + (x)])
				+ (kernelx[1][2] * pinput[((y)*input->widthStep) + (x + 1)])
				+ (kernelx[2][0] * pinput[((y + 1)*input->widthStep) + (x - 1)])
				+ (kernelx[2][1] * pinput[((y + 1)*input->widthStep) + (x)])
				+ (kernelx[2][2] * pinput[((y + 1)*input->widthStep) + (x + 1)]));

			mag_y = (short)((kernely[0][0] * pinput[((y - 1)*input->widthStep) + (x - 1)])
				+ (kernely[0][1] * pinput[((y - 1)*input->widthStep) + (x)])
				+ (kernely[0][2] * pinput[((y - 1)*input->widthStep) + (x + 1)])
				+ (kernely[1][0] * pinput[((y)*input->widthStep) + (x - 1)])
				+ (kernely[1][1] * pinput[((y)*input->widthStep) + (x)])
				+ (kernely[1][2] * pinput[((y)*input->widthStep) + (x + 1)])
				+ (kernely[2][0] * pinput[((y + 1)*input->widthStep) + (x - 1)])
				+ (kernely[2][1] * pinput[((y + 1)*input->widthStep) + (x)])
				+ (kernely[2][2] * pinput[((y + 1)*input->widthStep) + (x + 1)]));

			if (mag_x < 0)
				mag_x = 0;
			if (mag_y < 0)
				mag_y = 0;

			short val = (short)(sqrt((float)((mag_x * mag_x) + (mag_y * mag_y))));
			pdxy[y*input->widthStep + x] = (uchar)val;

			short theta_val = (short)(atan2((float)mag_y, (float)mag_x) * (180 / PI));
			if (theta_val < 0)
				theta_val = theta_val + 180;
			ptheta[y*input->widthStep + x] = (uchar)theta_val;;

		}
	}

	/*Calculate the number of cells rows and columns for image*/
	int row = input->height;
	int cell_counti = (int)floor((float)row / (float)cellSize);
	int col = input->width;
	int cell_countj = (int)floor((float)col / (float)cellSize);

	/*Declare variables that dictate start and end of each cell*/
	/*start_i, end_i, start_j, end_j are absolute pixel postion of the entire image*/
	/*i - start_i and j - start_j are the relative positive, i.e position will range 0 - 7 for temporary structure.*/
	int start_i, end_i, start_j, end_j;
	double angleratio = 0;

	/*3D histogram array and initialise to zero*/
	double* histogram = (double*)malloc(cell_counti * cell_countj * 9 * sizeof(double));
	if (histogram == NULL)
	{
		printf("Out of memory");
		return NULL;
	}
	for (i = 0; i < cell_counti; i++)
		for (j = 0; j < cell_countj; j++)
			for (k = 0; k < 9; k++)
				*(histogram + i * cell_countj * 9 + j * 9 + k) = 0;

	/*Iterate through each cell group*/
	int cell_i;
	int cell_j;
	for (cell_i = 0; cell_i < cell_counti; cell_i++)
	{
		for (cell_j = 0; cell_j < cell_countj; cell_j++)
		{
			/*Set absolute pixel locations from cell group*/
			start_i = (cell_i)*cellSize;
			end_i = (cell_i + 1)*cellSize - 1;
			start_j = (cell_j)*cellSize;
			end_j = (cell_j + 1)*cellSize - 1;

			/*Orientation Binning*/
			for (i = start_i; i <= end_i; i++)
			{
				for (j = start_j; j <= end_j; j++)
				{
					angleratio = ptheta[i*input->widthStep + j] / (double)bin_size;
					/*Separate integer and decimal part*/
					int angleint = (int)angleratio;
					double angledec = angleratio - angleint;
					/*Put magnitude in appropriate bin*/
					if (angleint > 7)
					{
						*(histogram + cell_i * cell_countj * 9 + cell_j * 9 + 0) += pdxy[i*input->widthStep + j] * (angledec);
					}
					else
					{
						*(histogram + cell_i * cell_countj * 9 + cell_j * 9 + angleint + 1) += pdxy[i*input->widthStep + j] * (angledec);
					}
					*(histogram + cell_i * cell_countj * 9 + cell_j * 9 + angleint) += pdxy[i*input->widthStep + j] * (1 - angledec);
				}
			}
		}
	}

	int cell_block = 4;
	int block_counti = cell_counti - 1;
	int block_countj = cell_countj - 1;
	int fstart = 0;

	/*Create array to store entire HOG features vector*/
	double* HOG_features = (double*)malloc(block_counti * block_countj * 36 * sizeof(double));
	//char *HOG_features_char = NULL;
	if (HOG_features == NULL)
	{
		printf("Out of memory");
		return NULL;
	}

	for (i = 0; i < block_counti*block_countj * 36; i++)
		*(HOG_features + i) = 0;

	/*Iterate through all the blocks*/
	int block_i;
	int block_j;
	for (block_i = 0; block_i < block_counti; block_i++)
	{
		for (block_j = 0; block_j < block_countj; block_j++)
		{
			double len_block = 0;
			/*Create 3D dynamic array to store single block features*/
			double* histo_block = (double*)malloc(36 * sizeof(double));
			if (histo_block == NULL)
			{
				printf("Out of memory");
				return NULL;
			}
			for (i = 0; i < 36; i++)
				*(histo_block + i) = 0;

			/*Create array to store normalized single block features*/
			double* norm_block = (double*)malloc(36 * sizeof(double));
			if (norm_block == NULL)
			{
				printf("Out of memory");
				return NULL;
			}
			for (i = 0; i < 36; i++)
				*(norm_block + i) = 0;

			/*Iterate through each single block*/
			for (i = 0; i < 2; i++)
			{
				for (j = 0; j < 2; j++)
				{
					int cell_i = block_i + i;
					int cell_j = block_j + j;
					int cell_b_count = 2 * i + j;
					/*Copy cell features to block features*/
					int ii;
					for (ii = 0; ii < 9; ii++)
					{
						*(histo_block + (ii + cell_b_count * 9)) = *(histogram + cell_i * cell_countj * 9 + cell_j * 9 + ii);
						len_block += *(histogram + cell_i * cell_countj * 9 + cell_j * 9 + ii) * *(histogram + cell_i * cell_countj * 9 + cell_j * 9 + ii);
					}
				}
			}
			/*Calculate the normalized block*/
			/*There is a problem is len_block is equal to zero. Program gives invalid values*/
			len_block = sqrt(len_block);
			if (len_block != 0)
			{
				for (i = 0; i < 36; i++)
				{
					*(norm_block + i) = *(histo_block + i) / len_block;
				}
					
				int jj;
				for (jj = 0; jj < 36; jj++)
				{
					*(HOG_features + jj + fstart) = *(norm_block + jj);
				}

			}
			fstart += 36;

			free(histo_block);
			free(norm_block);
		}
	}
	/*Free memory*/
	free(histogram);
	free(pdxy);
	free(ptheta);

	return HOG_features;
}

void train_svm(void)
{
	printf("Dataset file with HOG features created! \nGenerating SVM Model from dataset...\n");
	system("External_Libraries\\libsvm\\windows\\svm-train.exe data\\SVM_Format\\ObjectDetectionTrain.svm data\\Models\\ObjectDetectionModel");
	printf("Model successfully created!\n");
	return;
}
void run_object_detector(int stepSize, int windowSize, int mode, double scale)
{
	/*Search directory for multiple test images*/
	printf("Searching directory for test images...\n");
	DIR *p_dir;
	struct dirent *entry;
	p_dir = opendir("data\\Objects\\test_images");
	if (p_dir)
	{
		while ((entry = readdir(p_dir)) != NULL)
		{
			if (entry->d_type == DT_REG)
			{
				/*Get image name and load image*/
				//write_label(HOG_label);
				char *p_path_name = merge_path_and_file_name("data\\Objects\\test_images", entry->d_name);
				IplImage *input = cvLoadImage(p_path_name, CV_LOAD_IMAGE_GRAYSCALE);
				IplImage *output = cvLoadImage(p_path_name, CV_LOAD_IMAGE_COLOR);
				free(p_path_name);
				if (!input)
				{
					printf("Image can NOT Load!!!\n");
					return;
				}

				printf("Image found!\n"
					"Creating multiple sized copies of image\n");

				/*Create multiple images of different sizes*/
				int max_num_of_pyramid = 5;
				int pyramid_iteration = 0;
				int num_of_windows_to_compute = 0;
				int total_num_of_windows = 0;

				IplImage **input_all_images; //Pointer to an array, which holds different sized IplImages
				IplImage *input_scaled;
				int *window_array;
				short **position_window_x;
				short **position_window_y;
				input_all_images = (IplImage **)malloc(max_num_of_pyramid * sizeof(IplImage *)); //Points to an array of different IplImages
				input_scaled = (IplImage *)malloc(max_num_of_pyramid * sizeof(IplImage));
				window_array = (int *)malloc(max_num_of_pyramid * sizeof(int));
				position_window_x = (short **)malloc(max_num_of_pyramid * sizeof(short *));
				position_window_y = (short **)malloc(max_num_of_pyramid * sizeof(short *));

				CvPoint *pt1; //bounding box locations
				CvPoint *pt2;
				int prediction_count = 0;
				int max_prediction_count = 100;
				int *p_prediction_count;
				p_prediction_count = &prediction_count;
				int *p_max_prediction_count;
				p_max_prediction_count = &max_prediction_count;

				pt1 = (CvPoint *)malloc(max_prediction_count * sizeof(CvPoint));
				pt2 = (CvPoint *)malloc(max_prediction_count * sizeof(CvPoint));

				/*Initialise image for first iteration*/
				input_scaled[pyramid_iteration] = *input; //Set input_scale to hold original input image
				printf("Copy %u created!\n", pyramid_iteration + 1);

				IplImage *input_buffer;
				while (1)
				{
					input_buffer = &input_scaled[pyramid_iteration]; //Set input buffer to hold 1st scale input image
					num_of_windows_to_compute = (((input_buffer->width - windowSize) / stepSize) + 1) * (((input_buffer->height - windowSize) / stepSize) + 1); //Compute number of windows for this scale
					printf("Approximately %u windows to compute!\n", num_of_windows_to_compute);
					window_array[pyramid_iteration] = num_of_windows_to_compute;
					total_num_of_windows += num_of_windows_to_compute;
					input_all_images[pyramid_iteration] = (IplImage *)malloc(num_of_windows_to_compute * sizeof(IplImage)); //Allocate space for current array of different IplImages
					position_window_x[pyramid_iteration] = (short *)malloc(num_of_windows_to_compute * sizeof(short));
					position_window_y[pyramid_iteration] = (short *)malloc(num_of_windows_to_compute * sizeof(short));


					if ((input_buffer->width / scale < 64) || (input_buffer->height / scale < 64) || scale == 1) //If the next iteration is too small, break out of loop
						break;
					else
					{
						if (pyramid_iteration >= max_num_of_pyramid - 1)	//If more space is need
						{
							max_num_of_pyramid *= 2;
							input_scaled = (IplImage *)realloc(input_scaled, max_num_of_pyramid * sizeof(IplImage));
							window_array = (int *)realloc(window_array, max_num_of_pyramid * sizeof(int));
							input_all_images = (IplImage **)realloc(input_all_images, max_num_of_pyramid * sizeof(IplImage *)); //Increase array of IplImages
							position_window_x = (short **)realloc(position_window_x, max_num_of_pyramid * sizeof(short *));
							position_window_y = (short **)realloc(position_window_y, max_num_of_pyramid * sizeof(short *));
						}
						pyramid_iteration++;
						input_scaled[pyramid_iteration] = *(img_resize(input_buffer, (int)(input_buffer->width / scale), (int)(input_buffer->height / scale))); //Resize input image and store it at another location
						printf("Copy %u created!\n", pyramid_iteration + 1);
					}
				}

				/*Run sliding window for images*/
				int i;
				for (i = 0; i <= pyramid_iteration; i++)
				{
					sliding_window(&input_scaled[i], input_all_images, stepSize, windowSize, i, position_window_x, position_window_y);
				}

				/*Load SVM Model once before sliding window*/
				struct svm_model* model;
				if ((model = svm_load_model("data\\Models\\ObjectDetectionModel")) == 0)
				{
					printf("can't open model file %s\n", "ObjectDetectionModel");
					exit(1);
				}

				enterGPU(input_all_images, mode, model, total_num_of_windows, window_array, p_prediction_count, windowSize, scale, position_window_x, position_window_y, pt1, pt2);


				printf("Prediction count: %u", prediction_count);
				draw_bounding_box(output, pt1, pt2, prediction_count);


				printf("All bounding boxes drawn\n");
				printf("Displaying Image!\n");
				cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
				cvShowImage("result", output);
				cvWaitKey(0);
				cvDestroyAllWindows();

				/*Free memory*/
				for (i = 0; i <= pyramid_iteration; i++)
				{
					free(input_all_images[i]);
					free(position_window_x[i]);
					free(position_window_y[i]);
				}
				free(input_scaled);
				free(input_all_images);
				free(window_array);
				free(pt1);
				free(pt2);
				free(position_window_x);
				free(position_window_y);

				svm_free_and_destroy_model(&model);

				cvReleaseImage(&input);
				cvReleaseImage(&output);
			}
		}
		closedir(p_dir);
	}
	return;
}

IplImage* img_resize(IplImage* src_img, int new_width, int new_height)
{
	IplImage* des_img;
	des_img = cvCreateImage(cvSize(new_width, new_height), src_img->depth, src_img->nChannels);
	cvResize(src_img, des_img, CV_INTER_LINEAR);

	return des_img;
}
void sliding_window(IplImage *input, IplImage **input_all_images, int stepSize, int windowSize, int pyramid_iteration_count, short **position_window_x, short **position_window_y)
{
	int y;
	int x;
	int i = 0;
	for (y = 0; y <= (input->height - windowSize); y += stepSize)
	{
		for (x = 0; x <= (input->width - windowSize); x += stepSize)
		{
			/*Create sample test image*/
			cvSetImageROI(input, cvRect(x, y, windowSize, windowSize));
			//IplImage *tmp = cvCreateImage(cvGetSize(input), input->depth, input->nChannels);
			//cvCopy(input, tmp, NULL);
			
			////input = cvCloneImage(tmp);
			input_all_images[pyramid_iteration_count][i] = *(cvCreateImage(cvGetSize(input), input->depth, input->nChannels)); 
			cvCopy(input, &input_all_images[pyramid_iteration_count][i], NULL);
			cvResetImageROI(input);
			
			/*Store corresponding x and y values*/
			position_window_x[pyramid_iteration_count][i] = x;
			position_window_y[pyramid_iteration_count][i] = y;
			

			i++;
		}
	}
	return;
}

int my_SVM_prediction(double *HOG_features, struct svm_model* model)
{ 
	struct svm_node *x;
	int max_nr_attr = 2000;

	/*Initialise variable x that holds*/
	x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));

	/*Declare variables that will be used*/
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	//---------------------------------------------------------------------

	int i = 0;
	int resultant_class;

	double target_label, predict_label;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

	/*Extract label from HOG feature string*/
	target_label = (double)0;
	
	
	/*Create structure compatible for SVM library*/
	while (1)
	{
		if (i >= 1764)
		{
			break;
		}

		x[i].index = i + 1;
		inst_max_index = x[i].index;
		x[i].value = HOG_features[i];
		++i;
	}
	

	x[i].index = -1;

	predict_label = my_svm_predict(model, x);
	resultant_class = (int)(predict_label);

	/*Accuracy code*/
	if (predict_label == target_label)
		++correct;
	error += (predict_label - target_label)*(predict_label - target_label);
	sump += predict_label;
	sumt += target_label;
	sumpp += predict_label * predict_label;
	sumtt += target_label * target_label;
	sumpt += predict_label * target_label;
	++total;

	/*Free Memory*/
	free(x);

	return resultant_class;
}

double my_svm_predict(struct svm_model* model, struct svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;

	dec_values = (double *)malloc(((nr_class*(nr_class - 1) / 2)) * sizeof(double));
	
	double pred_result = my_svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double my_svm_predict_values(struct svm_model *model, struct svm_node *x, double* dec_values)
{
	int i;
	int nr_class = model->nr_class;
	int l = model->l;

	double *kvalue = (double *)malloc((l) * sizeof(double));

	for (i = 0; i < l; i++)
	{
		kvalue[i] = my_k_function(x, model->SV[i], model->param);
	}
		

	int *start = (int *)malloc((nr_class) * sizeof(int));
	start[0] = 0;
	for (i = 1; i < nr_class; i++)
		start[i] = start[i - 1] + model->nSV[i - 1];

	int *vote = (int *)malloc((nr_class) * sizeof(int));
	for (i = 0; i < nr_class; i++)
		vote[i] = 0;

	int p = 0;
	for (i = 0; i < nr_class; i++)
		for (int j = i + 1; j < nr_class; j++)
		{
			double sum = 0;
			int si = start[i];
			int sj = start[j];
			int ci = model->nSV[i];
			int cj = model->nSV[j];

			int k;
			double *coef1 = model->sv_coef[j - 1];
			double *coef2 = model->sv_coef[i];
			for (k = 0; k < ci; k++)
				sum += coef1[si + k] * kvalue[si + k];
			for (k = 0; k < cj; k++)
				sum += coef2[sj + k] * kvalue[sj + k];
			sum -= model->rho[p];
			dec_values[p] = sum;

			if (dec_values[p] > 0)
				++vote[i];
			else
				++vote[j];
			p++;
		}

	int vote_max_idx = 0;
	for (i = 1; i < nr_class; i++)
		if (vote[i] > vote[vote_max_idx])
			vote_max_idx = i;

	free(kvalue);
	free(start);
	free(vote);
	return model->label[vote_max_idx];
}

double my_k_function(struct svm_node *x, struct svm_node *y, struct svm_parameter param)
{
	double sum = 0;
	
	while (x->index != -1 && y->index != -1)
	{
		if (x->index == y->index)
		{
			double d = x->value - y->value;
			sum += d * d;
			++x;
			++y;
		}
		else
		{
			if (x->index > y->index)
			{
				sum += y->value * y->value;
				++y;
			}
			else
			{
				sum += x->value * x->value;
				++x;
			}
		}
	}

	while (x->index != -1)
	{
		sum += x->value * x->value;
		++x;
	}

	while (y->index != -1)
	{
		sum += y->value * y->value;
		++y;
	}
	return exp(-param.gamma*sum);
}

void classify_svm(void)
{
	printf("HOG features for test image calculated and written in SVM format to a file!\nRunning SVM software...\n");
	system("External_Libraries\\libsvm\\unix\\svm-predict data\\SVM_Format\\ObjectDetectionTest.svm data\\Models\\ObjectDetectionModel data\\Models\\ObjectDetectionPredictions");
	printf("Successful! Predictions output in a file!\n");
	return;
}
void store_bounding_box_points(int windowSize, double scale,  int pyramid_num, int position_window_x, int position_window_y, CvPoint *pt1, CvPoint *pt2, int *p_prediction_count)
{
	int prediction_count = *p_prediction_count;
	/*Calculate scaling value*/
	double scale_multiple = 1;
	if (pyramid_num > 0)
	{
		int j;
		for (j = 0; j < pyramid_num; j++)
		{
			scale_multiple = scale_multiple * scale;
		}
	}

	pt1[prediction_count].x = (int)(position_window_x * (scale_multiple));
	pt1[prediction_count].y = (int)(position_window_y * (scale_multiple));
	pt2[prediction_count].x = (int)((position_window_x + windowSize) * (scale_multiple));
	pt2[prediction_count].y = (int)((position_window_y + windowSize) * (scale_multiple));
	
	printf("p1.x: %u, p1.y: %u\n", pt1[prediction_count].x, pt1[prediction_count].y);
	//printf("p2.x: %u, p2.y: %u\n", pt2[prediction_count].x, pt2[prediction_count].y);

	//printf("A bounding box drawn...\n");
}

void draw_bounding_box(IplImage *output, CvPoint *pt1, CvPoint *pt2, int prediction_count)
{
	int i;
	for (i = 1; i <= prediction_count; i++)
		cvRectangle(output, pt1[i], pt2[i], CV_RGB(255, 0, 0), 2, 8, 0);
}


void enterGPU(IplImage **input, int mode, struct svm_model* model, int total_num_of_windows, int *window_array, int *p_prediction_count, int windowSize, double scale, short **position_window_x, short **position_window_y, CvPoint *pt1, CvPoint *pt2)
{

	int i; //test
	for (i = 0; i < total_num_of_windows; i++)
	{
		/*If threads valid*/
		if (i < total_num_of_windows)
		{
			int first_num = 0;
			int second_num = 0;
			int upper_bounds = window_array[first_num];
			int lower_bounds = 0;

			/*Extract first number for 2D image array*/
			while (1)
			{
				if (i < upper_bounds)
					break;
				else
				{
					first_num++;
					upper_bounds += window_array[first_num];
				}
			}

			/*Extract second number for 2D image array*/
			lower_bounds = (upper_bounds - window_array[first_num]) + 1;
			second_num = i - (lower_bounds - 1); //(lower_bounds - 1) window number to index

			IplImage *image_structure = &input[first_num][second_num];

			/*Get HOG features*/
			double *HOG_features = HOG_function_predict(image_structure, mode);

			/*Run prediction algorithm*/
			int prediction_num = my_SVM_prediction(HOG_features, model);
			free(HOG_features);

			/*Draw bounding box*/
			if (prediction_num > 0)
			{
				(*p_prediction_count)++; //The value of ptr is incremented
				store_bounding_box_points(windowSize, scale, first_num, position_window_x[first_num][second_num], position_window_y[first_num][second_num], pt1, pt2, p_prediction_count);
			}

		}
	}
}


int main()
{
	//sizeof(IplImage);
	/*Default Inputs*/
	int stepSize = 32;	//Number of pixels going to skip over in x and y direction default 640
	int windowSize = 64; //the width and height of sliding window
	int mode = 3; //Mode
	double scale = 1.2; //Pyramid Image Scale
	//Max prediction count

	int num_of_folders = 2;
	char **path[2];
	char *path1 = "data\\Objects\\positive_examples";
	path[0] = &path1;
	char *path2 = "data\\Objects\\negative_examples";
	path[1] = &path2;

	printf("Welcome to an object detection program\n"
		"Enter the mode you want to enter\n"
		"0 - Extract HOG features and Train Model\n"
		"1 - Train model with previously extracted features\n"
		"2 - Test model with labelled images for SVM metrics\n"
		"3 - Real time predictions\n");
	//mode = getchar() - '0';
	
	/*Timer start*/
	clock_t t;
	t = clock();

	if (mode == 0)
	{
		remove("data\\SVM_Format\\ObjectDetectionTrain.svm");
		/*Create a dataset in SVM format*/
		create_dataset_in_svm_format(path, num_of_folders, mode);
		/*Train SVM machine*/
		train_svm();
	}
	else if (mode == 1)
	{
		train_svm();
	}
	else if (mode == 2)
	{
		printf("Sorry, feature not implemented fully as yet!");
	}
	else if (mode == 3)
	{
		/*Specify sizes for algorithm*/
		printf("Enter the step size of sliding window algorithm\n");
		//stepSize = getchar() - '0';
		printf("Enter the changing scales of sliding window algorithm\n");
		//scale = getchar() - '0';
		/*Test Image*/
		run_object_detector(stepSize, windowSize, mode, scale);
	}
	else
	{
		printf("Invalid input... Closing\n");
	}

	/*Timer end*/
	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds 
	printf("Program took %f seconds to execute \n", time_taken);

	system("pause");
	return 0;
}
	
