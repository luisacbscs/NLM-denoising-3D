#include <iostream>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <string>
#include <regex>
#include <math.h>
#include <omp.h>	// Configuration Properties > C/C++ > Language > Open MP Support: Yes (/openmp)

using namespace std;
using namespace itk;

// ARGUMENTS: 0 - .EXE, 1 - PATH TO IMAGE + FILE NAME, 2 - PATH TO OUTPUT IMAGE + FILENAME,
//  3 - SIGMA (NOISE), 4 - H, 5 - KERNEL SIZE, 6 - NEIGHBOURHOOD MAX DISTANCE
int main(int argc, char **argv) {

	// FILE HANDLING
	string inputFilename;
	string outputFilename = "output.nii";

	// NLM PARAMETERS
	float sigma = 0.01;
	float h = 0.15;
	int kernelSize = 3;
	int neighbRadius = 3;	// DISTANCE TO THE CENTER PIXEL; NEIGHBOURHOOD WILL BE SIZED (2*neighbDistance+1)(2*neighbDistance+1)

	if (argc == 1) {

		std::cout << "Introduce file path: ";
		cin >> inputFilename;

		std::cout << "Introduce output file path: ";
		cin >> outputFilename;

		char define;
		std::cout << "Define NLM denoising parameters? (Y/N)? ";
		cin >> define;
		if (define == 'Y') {
			std::cout << "Introduce noise standard deviation estimate: ";
			cin >> sigma;
			std::cout << "Introduce decay factor h: ";
			cin >> h;
			std::cout << "Introduce kernel size: ";
			cin >> kernelSize;
			std::cout << "Introduce max distance for denoising (denoising radius in vx): ";
			cin >> neighbRadius;
		}

	} else if (argc == 6) {
		
		inputFilename = argv[1];
		sigma = atof(argv[2]);
		h = atof(argv[3]);		
		kernelSize = atoi(argv[4]);
		neighbRadius = atoi(argv[5]);

		// OUTPUT FILENAME
		string suffix = ".nii";
		stringstream hss, kernelSizess, neighbSizess;
		hss << h;
		kernelSizess << kernelSize;
		neighbSizess << 2 * neighbRadius + 1;
		string newSuffix = "_" + hss.str() + "_" + kernelSizess.str() + "_" + neighbSizess.str() + ".nii";
		outputFilename = inputFilename;
		outputFilename = outputFilename.replace(inputFilename.find(suffix), suffix.length(), newSuffix);

	}
	else if (argc == 7) {
		inputFilename = argv[1];
		outputFilename = argv[2];
		sigma = atof(argv[3]);
		h = atof(argv[4]);
		kernelSize = atoi(argv[5]);
		neighbRadius = atoi(argv[6]);
	}
	else {
		std::cout << "Insufficient parameters for NLM denoising. Please introduce:"
			<< "\n - File path + file name\n - Output file path + file name (optional)\n - Noise standard deviation estimate\n"
			<< " - Decay factor h\n - Kernel size\n - Neighbourhood radius (maximum distance from center)\n";
		return EXIT_FAILURE;
	}

	int kernelStep = int((kernelSize - 1) / 2);
	int neighbSize = 2 * neighbRadius + 1;
	int neighbCenterSize = 2 * (neighbRadius - kernelStep) + 1;

	// IMAGE OBJECT CONSTANTS
	using PixelType = float;
	const int Dimension = 3;

	// IMAGE CLASS INSTANTIATION
	using ImageType = Image<PixelType, Dimension>;

	// READER CLASS INSTANTIATION
	using ReaderType = ImageFileReader<ImageType>;

	// READER OBJECT
	ReaderType::Pointer reader = ReaderType::New();
	//reader->SetFileName(argv[1]);
	//cout << inputFilename.c_str() << endl;
	reader->SetFileName(inputFilename.c_str());

	try
	{
		reader->Update();
	}
	catch (const itk::ExceptionObject& error)
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
	

	// IMAGE OBJECT
	ImageType::Pointer image = reader->GetOutput();
	ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();

	std::cout << "Input file: " << inputFilename.c_str() << endl;
	std::cout << "Image size: " << size << "\n";
	
	std::cout << "NLM parameters:\n - Noise standard deviation: " << sigma << "\n - h = " << h << "\n - Kernel Size: " << kernelSize << "x"
		<< kernelSize << "x" << kernelSize << "\n - Neighbourhood size: " << neighbSize << "x" << neighbSize << "x" << neighbSize << endl;

	// ITERATE OVER THE IMAGE
	int frameStart = kernelStep;
	int frameEndx = size[0] - kernelStep;
	int frameEndy = size[1] - kernelStep;
	int frameEndz = size[2] - kernelStep;
	// Using OpenMP to parallelize the iteration
	#pragma omp parallel for 
	for (int i = 0; i < size[0]; ++i) {
		for (int j = 0; j < size[1]; ++j) {
			for (int k = 0; k < size[2]; ++k) {

				ImageType::IndexType index;	// index must be created inside the loop for the parallelization to be possible

				// DEALING WITH BORDERS: REFLECT STRATEGY
				if (i < 0) {
					index[0] = abs(i) - 1;
				}
				else if (i > (size[0] - 1)) {
					index[0] = size[0] - (i - (size[0] - 1));
				}
				else {
					index[0] = i;
				}

				if (j < 0) {
					index[1] = abs(j) - 1;
				}
				else if (j > (size[1] - 1)) {
					index[1] = size[1] - (j - (size[1] - 1));
				}
				else {
					index[1] = j;
				}

				if (k < 0) {
					index[2] = abs(k) - 1;
				}
				else if (k > (size[2] - 1)) {
					index[2] = size[2] - (k - (size[2] - 1));
				}
				else {
					index[2] = k;
				}

				// STORE THE VALUES OF THE CENTER KERNEL IN AN ARRAY
				float* centerKernel = new float[kernelSize * kernelSize * kernelSize];
				// GET THE MEAN OF THE KERNEL OF THE CENTER PIXEL
				float centerSum = 0.0;
				int centerCounter = 0;

				// ITERATE OVER THE CENTER KERNEL
				for (int ci = i - kernelStep; ci <= i + kernelStep; ++ci) {
					for (int cj = j - kernelStep; cj <= j + kernelStep; ++cj) {
						for (int ck = k - kernelStep; ck <= k + kernelStep; ++ck) {

							ImageType::IndexType cind;

							// DEALING WITH BORDERS: REFLECT STRATEGY
							if (ci < 0) {
								cind[0] = abs(ci) - 1;
							}
							else if (ci > (size[0] - 1)) {
								cind[0] = size[0] - (ci - (size[0] - 1));
							}
							else {
								cind[0] = ci;
							}

							if (cj < 0) {
								cind[1] = abs(cj) - 1;
							}
							else if (cj > (size[1] - 1)) {
								cind[1] = size[1] - (cj - (size[1] - 1));
							}
							else {
								cind[1] = cj;
							}

							if (ck < 0) {
								cind[2] = abs(ck) - 1;
							}
							else if (ck > (size[2] - 1)) {
								cind[2] = size[2] - (ck - (size[2] - 1));
							}
							else {
								cind[2] = ck;
							}

							centerCounter++;
							centerSum = centerSum + (float)image->GetPixel(cind);
							centerKernel[(ci - i + kernelStep) * kernelSize * kernelSize +
								(cj - j + kernelStep) * kernelSize + (ck - k + kernelStep)]
								= (float)image->GetPixel(cind);
							// The memory position for a given index (i, j, k) of a 3D array of size M*N*D will be given by:
							// i * N * N + j * D + k. Thus, to acess the respective value: Arr[i * N * N + j * D + k].

						}
					}
				}

				float centerMean;
				centerMean = centerSum / centerCounter;

				// VARIABLES TO STORE THE SUM OF THE WEIGHTS AND THE WEIGHTED PIXELS VALUES
				float weightNormConst = 0.0;
				float weightedPxSum = 0.0;

				// ARRAY THAT SAVES THE NEIGHBOURHOOD KERNELS' MEANS (IN THE CORRESPONDING CENTER PIXEL)
				float* neighbVals = new float[neighbCenterSize * neighbCenterSize * neighbCenterSize]; // must have the size of the neighbourhood 
				// excluding the borders: 2*(neighbRadius-kernelStep)+1 in each direction = number of kernels in the neighbourhood
				float pxValue = 0.0;
				int pxCounter = 0;

				// ITERATE OVER THE NEIGHBOURHOOD	
				for (int ii = index[0] - neighbRadius + kernelStep; ii <= index[0] + neighbRadius - kernelStep; ++ii) {
					for (int jj = index[1] - neighbRadius + kernelStep; jj <= index[1] + neighbRadius - kernelStep; ++jj) {
						for (int kk = index[2] - neighbRadius + kernelStep; kk <= index[2] + neighbRadius - kernelStep; ++kk) {

							ImageType::IndexType indexNeighb;

							// DEALING WITH BORDERS: REFLECT STRATEGY
							if (ii < 0) {
								indexNeighb[0] = abs(ii) - 1;
							}
							else if (ii > (size[0] - 1)) {
								indexNeighb[0] = size[0] - (ii - (size[0] - 1));
							}
							else {
								indexNeighb[0] = ii;
							}

							if (jj < 0) {
								indexNeighb[1] = abs(jj) - 1;
							}
							else if (jj > (size[1] - 1)) {
								indexNeighb[1] = size[1] - (jj - (size[1] - 1));
							}
							else {
								indexNeighb[1] = jj;
							}

							if (kk < 0) {
								indexNeighb[2] = abs(kk) - 1;
							}
							else if (kk > (size[2] - 1)) {
								indexNeighb[2] = size[2] - (kk - (size[2] - 1));
							}
							else {
								indexNeighb[2] = kk;
							}

							// STORE THE VALUES OF EACH OF THE NEIGHBOURHOOD'S KERNELS IN AN ARRAY
							float* neighbKernel = new float[kernelSize * kernelSize * kernelSize];
							// VARIABLES TO GET THE KERNEL'S MEAN
							float kernelSum = 0.0;
							int counter = 0;
							// ITERATE OVER THE KERNEL
							//ImageType::IndexType indexaux;
							//for (indexaux[0] = indexNeighb[0] - kernelStep; indexaux[0] <= indexNeighb[0] + kernelStep; ++indexaux[0]) {				
								//for (indexaux[1] = indexNeighb[1] - kernelStep; indexaux[1] <= indexNeighb[1] + kernelStep; ++indexaux[1]) {
									//for (indexaux[2] = indexNeighb[2] - kernelStep; indexaux[2] <= indexNeighb[2] + kernelStep; ++indexaux[2]) {
							
							for (int iii = ii - kernelStep; iii <= ii + kernelStep; ++iii) {
								for (int jjj = jj - kernelStep; jjj <= jj + kernelStep; ++jjj) {
									for (int kkk = kk - kernelStep; kkk <= kk + kernelStep; ++kkk) {
										//cout << indexaux[0] << ", " << indexaux[1] << ", " << indexaux[2] << " --> ";

										ImageType::IndexType indexaux;
										
										// DEALING WITH BORDERS: REFLECT STRATEGY
										if (iii < 0) {
											indexaux[0] = abs(iii) - 1;
										}
										else if (iii > (size[0] - 1)) {
											indexaux[0] = size[0] - (iii - (size[0] - 1));
										}
										else {
											indexaux[0] = iii;
										}

										if (jjj < 0) {
											indexaux[1] = abs(jjj) - 1;
										}
										else if (jjj > (size[1] - 1)) {
											indexaux[1] = size[1] - (jjj - (size[1] - 1));
										}
										else {
											indexaux[1] = jjj;
										}

										if (kkk < 0) {
											indexaux[2] = abs(kkk) - 1;
										}
										else if (kkk > (size[2] - 1)) {
											indexaux[2] = size[2] - (kkk - (size[2] - 1));
										}
										else {
											indexaux[2] = kkk;
										}

										//cout << indexaux[0] << ", " << indexaux[1] << ", " << indexaux[2] << endl;
										
										counter++; // COUNTS THE NUMBER OF PIXELS IN KERNEL (e.g. 3x3 kernel = 9 pixels)
										kernelSum = kernelSum + (float)image->GetPixel(indexaux);
										neighbKernel[(iii - ii + kernelStep) * kernelSize * kernelSize
											+ (jjj - jj + kernelStep) * kernelSize +
											(kkk - kk + kernelStep)] = (float)image->GetPixel(indexaux);

									}
								}
							}

							// VARIABLE TO STORE THE SUM OF THE SQUARED ERRORS BETWEEN THE NEIGHBOURHOOD KERNEL'S PIXELS AND THE CENTER KERNEL'S PIXELS
							float squaredErrorsSum = 0.0;

							for (int x = 0; x < kernelSize; x++) {
								for (int y = 0; y < kernelSize; y++) {
									for (int z = 0; z < kernelSize; z++) {

										squaredErrorsSum = squaredErrorsSum +
											pow((neighbKernel[x * kernelSize * kernelSize + y * kernelSize + z] 
												- centerKernel[x * kernelSize * kernelSize + y * kernelSize + z]), 2);

									}
								}
							}

							// VARIABLE TO STORE THE MSE BETWEEN EACH NEIGHBOURHOOD KERNEL AND THE CENTER KERNEL
							float kernelMSE;
							kernelMSE = squaredErrorsSum / counter;

							// CALCULATING THE WEIGHTS
							float weight = 0.0;
							if ((kernelMSE - (2 * pow(sigma, 2))) < 0) {
								weight = 1;
							}
							else {
								weight = exp(-((kernelMSE - (2 * pow(sigma, 2))) / pow(h, 2)));
							}

							// WEIGHT NORMALIZING CONSTANT = SUM OF ALL WEIGHTS
							weightNormConst = weightNormConst + weight;

							// SUM OF THE PRODUCTS BETWEEN EACH NEIGHBOUHOOD KERNEL'S CENTER PIXEL VALUE AND ITS WEIGHT
							weightedPxSum = weightedPxSum + (float)image->GetPixel(indexNeighb) * weight;

							// MEAN OF THE CURRENT NEIGHBOURHOOD KERNEL 
							float kernelMean;
							kernelMean = kernelSum / counter;

							// SAVING THE CURRENT NEIGHBOURHOOD KERNEL'S MEAN (OF THE CORRESPONDING CENTER PIXEL)
							neighbVals[(ii - i + neighbRadius - kernelStep) * neighbCenterSize * neighbCenterSize +
								(jj - j + neighbRadius - kernelStep) * neighbCenterSize + 
								(kk - k + neighbRadius - kernelStep)] = kernelMean;

							delete[] neighbKernel;
						}
					}
				}

				// ITERATE OVER THE ARRAY THAT STORED THE NEIGHBOURHOOD KERNELS' MEANS
				for (int x = 0; x < neighbCenterSize; x++) {
					for (int y = 0; y < neighbCenterSize; y++) {
						for (int z = 0; z < neighbCenterSize; z++) {

							if ((centerMean * (1 - h)) < neighbVals[x * neighbCenterSize * neighbCenterSize +
								y * neighbCenterSize + z] &&
								neighbVals[x * neighbCenterSize * neighbCenterSize + y * neighbCenterSize + z] <
								(centerMean * (1 + h)) &&
								isnan(neighbVals[x * neighbCenterSize * neighbCenterSize + y * neighbCenterSize + z]) == 0) {
								pxCounter++;
								pxValue = pxValue + neighbVals[x * neighbCenterSize * neighbCenterSize + y * neighbCenterSize + z];
							}
						}
					}
				}

				// NEW VALUE FOR THE PIXEL (AFTER THE DENOISING)
				float weightedMean;
				weightedMean = weightedPxSum / weightNormConst;

				// NEW VALUE FOR THE PIXEL (AFTER THE DENOISING)
				float newPx;
				newPx = pxValue / pxCounter;

				if (!isnan(newPx)) {
					image->SetPixel(index, float(weightedMean));
				}

				delete[] centerKernel;
				delete[] neighbVals;
			}
		}
	}

	
	using WriterType = itk::ImageFileWriter<ImageType>;
	auto writer = WriterType::New();

	std::cout << "Output file: " << outputFilename.c_str() << endl;
	writer->SetFileName(outputFilename.c_str());
	writer->SetInput(image);

	try
	{
		writer->Update();
	}
	catch (const itk::ExceptionObject& error)
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}

	return 0;

}
