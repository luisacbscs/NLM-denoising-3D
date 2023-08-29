#include <iostream>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <string>
#include <regex>
#include <math.h>
#include <omp.h>

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

		cout << "Introduce file path: ";
		cin >> inputFilename;

		cout << "Introduce output file path: ";
		cin >> outputFilename;

		char define;
		cout << "Define NLM denoising parameters? (Y/N)? ";
		cin >> define;
		if (define == 'Y') {
			cout << "Introduce noise standard deviation estimate: ";
			cin >> sigma;
			cout << "Introduce decay factor h: ";
			cin >> h;
			cout << "Introduce kernel size: ";
			cin >> kernelSize;
			cout << "Introduce max distance for denoising (denoising radius in vx): ";
			cin >> neighbRadius;
		}

	} else if (argc == 6) {
		
		inputFilename = argv[1];

		// OUTPUT FILENAME
		string suffix = ".nii";
		stringstream hss, kernelSizess, neighbSizess;
		hss << h;
		kernelSizess << kernelSize;
		neighbSizess << 2 * neighbRadius + 1;
		string newSuffix = "_" + hss.str() + "_" + kernelSizess.str() + "_" + neighbSizess.str() + ".nii";
		outputFilename = inputFilename.replace(inputFilename.find(suffix), suffix.length(), newSuffix);

		sigma = atof(argv[2]);
		h = atof(argv[3]);		
		kernelSize = atoi(argv[4]);
		neighbRadius = atoi(argv[5]);

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
		cout << "Insufficient parameters for NLM denoising. Please introduce:"
			<< "\n - File path + file name\n - Output file path + file name (optional)\n - Noise standard deviation estimate\n "
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
	reader->SetFileName(inputFilename.c_str());
	reader->Update();

	// IMAGE OBJECT
	ImageType::Pointer image = reader->GetOutput();
	ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();

	cout << "Image size: " << size << "\n";
	
	cout << "NLM parameters:\n - Noise standard deviation: " << sigma << "\n - h = " << h << "\n - Kernel Size: " << kernelSize << "x" 
		<< kernelSize << "x" << kernelSize << "\n - Neighbourhood size: " << neighbSize << "x" << neighbSize << "x" << neighbSize << endl;

	// ITERATE OVER THE IMAGE
	int frameStart = neighbRadius;
	int frameEndx = size[0] - neighbRadius;
	int frameEndy = size[1] - neighbRadius;
	int frameEndz = size[2] - neighbRadius;
	// Using OpenMP to parallelize the iteration
	#pragma omp parallel for
	for (int i = frameStart; i < frameEndx; ++i) {
		for (int j = frameStart; j < frameEndy; ++j) {
			for (int k = frameStart; k < frameEndz; ++k) {

				ImageType::IndexType index;	// index must be created inside the loop for the parallelization to be possible
				index[0] = i;
				index[1] = j;
				index[2] = k;

				// STORE THE VALUES OF THE CENTER KERNEL IN AN ARRAY
				float* centerKernel = new float[kernelSize * kernelSize * kernelSize];
				// GET THE MEAN OF THE KERNEL OF THE CENTER PIXEL
				float centerSum = 0.0;
				int centerCounter = 0;

				// ITERATE OVER THE CENTER KERNEL
				ImageType::IndexType cind;
				for (cind[0] = index[0] - kernelStep; cind[0] <= index[0] + kernelStep; ++cind[0]) {
					for (cind[1] = index[1] - kernelStep; cind[1] <= index[1] + kernelStep; ++cind[1]) {
						for (cind[2] = index[2] - kernelStep; cind[2] <= index[2] + kernelStep; ++cind[2]) {

							centerCounter++;
							centerSum = centerSum + (float)image->GetPixel(cind);
							centerKernel[(cind[0] - index[0] + kernelStep) * kernelSize * kernelSize +
								(cind[1] - index[1] + kernelStep) * kernelSize + (cind[2] - index[2] + kernelStep)]
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
				ImageType::IndexType indexNeighb;
				for (indexNeighb[0] = index[0] - neighbRadius + kernelStep; indexNeighb[0] <= index[0] + neighbRadius - kernelStep;
					++indexNeighb[0]) {
					for (indexNeighb[1] = index[1] - neighbRadius + kernelStep; indexNeighb[1] <= index[1] + neighbRadius - kernelStep;
						++indexNeighb[1]) {
						for (indexNeighb[2] = index[2] - neighbRadius + kernelStep; indexNeighb[2] <= index[2] + neighbRadius - kernelStep;
							++indexNeighb[2]) {

							// STORE THE VALUES OF EACH OF THE NEIGHBOURHOOD'S KERNELS IN AN ARRAY
							float* neighbKernel = new float[kernelSize * kernelSize * kernelSize];
							// VARIABLES TO GET THE KERNEL'S MEAN
							float kernelSum = 0.0;
							int counter = 0;
							// ITERATE OVER THE KERNEL
							ImageType::IndexType indexaux;
							for (indexaux[0] = indexNeighb[0] - kernelStep; indexaux[0] <= indexNeighb[0] + kernelStep; ++indexaux[0]) {
								for (indexaux[1] = indexNeighb[1] - kernelStep; indexaux[1] <= indexNeighb[1] + kernelStep; ++indexaux[1]) {
									for (indexaux[2] = indexNeighb[2] - kernelStep; indexaux[2] <= indexNeighb[2] + kernelStep; ++indexaux[2]) {

										counter++; // COUNTS THE NUMBER OF PIXELS IN KERNEL (e.g. 3x3 kernel = 9 pixels)
										kernelSum = kernelSum + (float)image->GetPixel(indexaux);
										neighbKernel[(indexaux[0] - indexNeighb[0] + kernelStep) * kernelSize * kernelSize
											+ (indexaux[1] - indexNeighb[1] + kernelStep) * kernelSize +
											(indexaux[2] - indexNeighb[2] + kernelStep)] = (float)image->GetPixel(indexaux);

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
							neighbVals[(indexNeighb[0] - index[0] + neighbRadius - kernelStep) * neighbCenterSize * neighbCenterSize +
								(indexNeighb[1] - index[1] + neighbRadius - kernelStep) * neighbCenterSize + 
								(indexNeighb[2] - index[2] + neighbRadius - kernelStep)] = kernelMean;

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
