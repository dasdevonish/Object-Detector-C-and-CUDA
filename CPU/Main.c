#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <stdio.h>
#include <dirent.h>
#include <math.h>
#include <opencv2/core.hpp>
#include <string.h>

#define PI   3.14159265358979323846
#define cellSize 8
#define bin_size 20

void HOG_function(IplImage *input, int mode);
void create_dataset_in_svm_format(int **path[], int num_of_svm_classes, int mode);
int count_files_in_directory(char *path);
void apply_HOG_descriptor(int num_of_folders, char *path, int HOG_label, int mode);
void write_label(int HOG_label);
char *merge_path_and_file_name(char *s1, char *s2);
void train_svm(void);
void run_object_detector(int stepSize, int windowSize, int mode, int scale);
IplImage* img_resize(IplImage* src_img, int new_width, int new_height);
void sliding_window(IplImage *input, int stepSize, int windowSize, int mode, IplImage *output, int scale, int pyramid_iteration);
void classify_svm(void);


void create_dataset_in_svm_format(int **path[], int num_of_folders, int mode)
{
	printf("Creating a dataset in the SVM format...\n");
	/*Count number of files in directory and assign HOG label per folder*/
	int HOG_label = 1;
	int number_of_files_per_directory[2] = { 0 };
	int total_files_in_directory = 0;
	for (int i = 0; i < num_of_folders; i++)
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
				///*Open appropriate file*/
				//FILE *fptr;
				//fptr = fopen("data\\SVM_Format\\ObjectDetectionTrain.svm", "a");
				///*Write to name features to a file*/
				//fprintf(fptr, "%s\n", p_path_name);
				//fclose(fptr);
				free(p_path_name);
				if (!input)
				{
					printf("Image can NOT Load!!!\n");
					return 1;
				}
				/*Apply HOG descriptor per image*/
				HOG_function(input, mode);

				cvWaitKey(0);
				cvDestroyAllWindows();
				
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
void HOG_function(IplImage *input, int mode) 
{
	/*Create 4 images and initialise to 0*/
	IplImage *dx = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);
	IplImage *dy = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);
	IplImage *dxy = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);
	IplImage *theta = cvCreateImage(cvSize(input->width, input->height), input->depth, input->nChannels);
	uchar *pdx = (uchar*)dx->imageData;
	uchar *pdy = (uchar*)dy->imageData;
	uchar *pdxy = (uchar*)dxy->imageData;
	uchar *ptheta = (uchar*)theta->imageData;
	for (int i = 0; i < input->height; i++)
		for (int j = 0; j < input->width; j++)
		{
			pdx[i*input->widthStep + j] = 0;
			pdy[i*input->widthStep + j] = 0;
			pdxy[i*input->widthStep + j] = 0;
			ptheta[i*input->widthStep + j] = 0;
		}
	/*Find gradient in x and y direction*/
	cvSobel(input, dx, 1, 0, 1);
	cvSobel(input, dy, 0, 1, 1);

	/*Calculate magnitudes and angles*/
	for (int i = 0; i < input->height; i++)
		for (int j = 0; j < input->width; j++)
		{
			pdxy[i*input->widthStep + j] = sqrt((pdx[i*input->widthStep + j] * pdx[i*input->widthStep + j]) + (pdy[i*input->widthStep + j] * pdy[i*input->widthStep + j]));
			ptheta[i*input->widthStep + j] = atan2(pdy[i*input->widthStep + j], pdx[i*input->widthStep + j]) * (180 / PI);
			if (ptheta[i*input->widthStep + j] < 0)
				ptheta[i*input->widthStep + j] = ptheta[i*input->widthStep + j] + 180;
			//may use round function instead of casting directly to int
		}

	/*Calculate the number of cells rows and columns for image*/
	int row = input->height;
	int cell_counti = floor(row / cellSize);
	int col = input->width;
	int cell_countj = floor(col / cellSize);

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
	for (int i = 0; i < cell_counti; i++)
		for (int j = 0; j < cell_countj; j++)
			for (int k = 0; k < 9; k++)
				*(histogram + i * cell_countj * 9 + j * 9 + k) = 0;

	/*Iterate through each cell group*/
	for (int cell_i = 0; cell_i < cell_counti; cell_i++)
	{
		for (int cell_j = 0; cell_j < cell_countj; cell_j++)
		{
			/*Set absolute pixel locations from cell group*/
			start_i = (cell_i)*cellSize;
			end_i = (cell_i + 1)*cellSize - 1;
			start_j = (cell_j)*cellSize;
			end_j = (cell_j + 1)*cellSize - 1;

			/*Orientation Binning*/
			for (int i = start_i; i <= end_i; i++)
			{
				for (int j = start_j; j <= end_j; j++)
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
	if (HOG_features == NULL)
	{
		printf("Out of memory");
		exit(0);
	}
	for (int i = 0; i < block_counti*block_countj * 36; i++)
		*(HOG_features + i) = 0;

	/*Iterate through all the blocks*/
	for (int block_i = 0; block_i < block_counti; block_i++)
	{
		for (int block_j = 0; block_j < block_countj; block_j++)
		{
			double len_block = 0;
			/*Create 3D dynamic array to store single block features*/
			double* histo_block = (double*)malloc(36 * sizeof(double));
			if (histo_block == NULL)
			{
				printf("Out of memory");
				exit(0);
			}
			for (int i = 0; i < 36; i++)
				*(histo_block + i) = 0;

			/*Create array to store normalized single block features*/
			double* norm_block = (double*)malloc(36 * sizeof(double));
			if (norm_block == NULL)
			{
				printf("Out of memory");
				exit(0);
			}
			for (int i = 0; i < 36; i++)
				*(norm_block + i) = 0;

			/*Iterate through each single block*/
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					int cell_i = block_i + i;
					int cell_j = block_j + j;
					int cell_b_count = 2 * i + j;
					/*Copy cell features to block features*/
					for (int ii = 0; ii < 9; ii++)
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
				for (int i = 0; i < 36; i++)
					*(norm_block + i) = *(histo_block + i) / len_block;

				for (int jj = 0; jj < 36; jj++)
					*(HOG_features + jj + fstart) = *(norm_block + jj);
			}
			fstart += 36;

			/*
			{
				if (isnan(*(norm_block + jj)))
					*(HOG_features + jj + fstart) = *(HOG_features + jj + fstart);
				else
					*(HOG_features + jj + fstart) = *(norm_block + jj);
			}
			*/	
			

			free(histo_block);
			free(norm_block);
		}
	}
	free(histogram);
	
	/*Open appropriate file*/
	FILE *fptr;
	if (mode == 0)
	{
		fptr = fopen("data\\SVM_Format\\ObjectDetectionTrain.svm", "a");
	}
	else
	{
		fptr = fopen("data\\SVM_Format\\ObjectDetectionTest.svm", "a");
		fprintf(fptr, "0");
	}
	/*Write to HOG features to a file*/
	int feature_number = 0;
	for (int i = 0; i < (36 * block_counti * block_countj); i++)
	{
		if (fptr == NULL)
		{
			printf("Error!");
			exit(1);
		}
		fprintf(fptr, " %d:%f", feature_number + 1 , *(HOG_features + feature_number));
		//printf(" %d:%f", feature_number + 1 , *(HOG_features + feature_number));
		feature_number++;
	}
	fprintf(fptr, "\n");
	//printf("\n");
	fclose(fptr);
	free(HOG_features);
}
void train_svm(void)
{
	printf("Dataset file with HOG features created! \nGenerating SVM Model from dataset...\n");
	system(" SVMlight\\svm_learn.exe data\\SVM_Format\\ObjectDetectionTrain.svm data\\Models\\ObjectDetectionModel ");
	printf("Model successfully created!\n");
	return;
}
void run_object_detector(int stepSize, int windowSize, int mode, int scale)
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
				//write_label(HOG_label);
				char *p_path_name = merge_path_and_file_name("data\\Objects\\test_images", entry->d_name);
				IplImage *input = cvLoadImage(p_path_name, CV_LOAD_IMAGE_GRAYSCALE);
				IplImage *output = cvLoadImage(p_path_name, CV_LOAD_IMAGE_COLOR);
				free(p_path_name);
				if (!input)
				{
					printf("Image can NOT Load!!!\n");
					return 1;
				}
				/*Pyramid function*/
				int pyramid_iteration = 0;
				while (1)
				{
					/*Display approximate number of possible images*/
					int num_of_windows_to_compute = (((input->width - windowSize) / stepSize) + 1) * (((input->height - windowSize) / stepSize) + 1);
					printf("Image found! Approximately %u windows to compute!\n", num_of_windows_to_compute);

					/*Run slidow window*/
					printf("Running sliding window algorithm...\n");
					sliding_window(input, stepSize, windowSize, mode, output, scale, pyramid_iteration);
					if ((input->width / scale < 64) || (input->height / scale < 64))
						break;
					else
					{
						pyramid_iteration++;
						printf("Resizing Image in Image Pyramid for different sizes...\n");
						input = img_resize(input, input->width / scale, input->height / scale);
						remove("data\\SVM_Format\\ObjectDetectionTest.svm");
					}
				}
				printf("All bounding boxes drawn\n");
				printf("Displaying Image!\n");
				cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
				cvShowImage("result", output);
				cvWaitKey(0);
				cvDestroyAllWindows();
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
void sliding_window(IplImage *input, int stepSize, int windowSize, int mode, IplImage *output, int scale, int pyramid_iteration)
{
	int num_of_windows_x_max = 0;
	/*Slide through whole test image*/
	for (int y = 0; y <= (input->height - windowSize); y += stepSize)
	{
		int num_of_windows_x = 0;
		for (int x = 0; x <= (input->width - windowSize); x += stepSize)
		{
			/*Create sample test image*/
			cvSetImageROI(input, cvRect(x, y, windowSize, windowSize));
			IplImage *tmp = cvCreateImage(cvGetSize(input), input->depth, input->nChannels);
			cvCopy(input, tmp, NULL);
			cvResetImageROI(input);
			////input = cvCloneImage(tmp);

			/*Output HOG feature vectors to a file*/
			HOG_function(tmp, mode);

			/*Store window number in x direction*/
			num_of_windows_x++;
			if (num_of_windows_x_max < num_of_windows_x)
			{
				num_of_windows_x_max = num_of_windows_x;
			}
		}
	}
	/*Run prediction software*/
	classify_svm();

	/*Read line and count line number*/
	printf("Reading predictions file and drawing bounding box in correct locations.\n");
	FILE *fp;
	float prediction_num;
	int line_count = 0;
	fp = fopen("data\\Models\\ObjectDetectionPredictions", "r");
	while (fscanf(fp, "%f", &prediction_num) == 1)
	{
		line_count++;
		/*Calculate corresponding prediction number to window position*/
		if (prediction_num > 0)
		{
			int position_window_col = (line_count % num_of_windows_x_max) - 1;
			int position_window_row = (line_count / num_of_windows_x_max);
			if (position_window_col == -1)
			{
				position_window_col = num_of_windows_x_max - 1;
				position_window_row = position_window_row - 1;
			}
			/*Calculate absolute window position*/
			int position_window_x = position_window_col * stepSize;
			int position_window_y = position_window_row * stepSize;

			/*Calculate scaling value*/
			int scale_multiple = 1;
			if (pyramid_iteration > 0)
			{
				for (int i = 0; i < pyramid_iteration; i++)
				{
					scale_multiple = scale_multiple * scale;
				}
			}
			/*Create bounding box*/
			CvPoint pt1, pt2;
			pt1.x = position_window_x * (scale_multiple);
			pt1.y = position_window_y * (scale_multiple);
			pt2.x = (position_window_x + windowSize) * (scale_multiple);
			pt2.y = (position_window_y + windowSize) * (scale_multiple);
			//cvResetImageROI(input);
			cvRectangle(output, pt1, pt2, CV_RGB(255, 0, 0), 2, 8, 0);
		}	
	}
	printf("Some bounding boxes drawn...\n");
	fclose(fp);
	return;
}
void classify_svm(void)
{
	printf("HOG features for test image calculated and written in SVM format to a file!\nRunning SVM software...\n");
	system(" SVMlight\\svm_classify.exe data\\SVM_Format\\ObjectDetectionTest.svm data\\Models\\ObjectDetectionModel data\\Models\\ObjectDetectionPredictions ");
	printf("Successful! Predictions output in a file!\n");
	return;
}


int main()
{
	/*Inputs*/
	int stepSize = 640;	//Number of pixels going to skip over in x and y direction
	int windowSize = 64; //the width and height of sliding window
	int mode = 0; //Mode
	int scale = 10; //Pyramid Image Scale

	int num_of_folders = 2;
	int **path[2];
	char *path1 = "data\\Objects\\positive_examples";
	path[0] = &path1;
	char *path2 = "data\\Objects\\negative_examples";
	path[1] = &path2;
	
	if (mode == 0)
	{
		remove("data\\SVM_Format\\ObjectDetectionTrain.svm");
		/*Create a dataset in SVM format*/
		create_dataset_in_svm_format(&path, num_of_folders, mode);
		/*Train SVM machine*/
		train_svm();
	}
	if (mode == 1)
	{
		remove("data\\SVM_Format\\ObjectDetectionTest.svm");
		/*Test Image*/
		run_object_detector(stepSize, windowSize, mode, scale);
	}
	system("pause");
	return 0;
}
	