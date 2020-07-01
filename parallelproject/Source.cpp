//summary

//The digital image enhancement tool takes a pgm image as an input and an intensity histogram is calculated
//by looking at the values of the pixels from the input. The cumulative histogram is then calculated
//and normalised by the tool to the maximum value of 255. A look-up table is then created using the histogram
//and is used by the tool to assign an intensity value to boost the contrast of the input image as reflected
//in the output.

//The assignment was developed on a Windows 10 PC using an Intel M3-6Y30 with HD 515 Graphics and the installation
//followed the basic instructions and uses the graphics processor.
//References: The development of this tool was heavily assisted by, and much of the kernel and general structure is taken from:
//https://github.com/gcielniak/OpenCL-Tutorials and other materials in the lectures.
//Other resources used in the development:
//http://www.cplusplus.com/doc/tutorial/
//https://github.com/HandsOnOpenCL/Exercises-Solutions
//https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf
//https://software.intel.com/en-us/articles/using-opencl-20-atomics

#include <iostream>
#include <vector>
#include "CImg.h"
#include "../include/Utils.h"

using namespace cimg_library;
using namespace std;

auto main(int argc, char** argv)-> int
{
	//set platform, device, and mode
	const auto platform_id = 0;
	const auto device_id = 0;
	const auto mode_id = 0;

	std::cout << "Running on: " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) <<
		std::endl;
	//load the test file
	string inputFilename = "test.pgm";

	//unsigned char to process input properly. display input image
	CImg<unsigned char> inputImage(inputFilename.c_str());
	CImgDisplay display(inputImage, "Image");

	{
		cout << "Pixel Amount: " << inputImage.height() * inputImage.width() << endl;
	}
	{
		cout << "Image Width: " << inputImage.width() << ", Image Height: " << inputImage.height() << endl;
	}
	//get 'context', where the platform and device are selected
	const cl::Context context = GetContext(platform_id, device_id);

	//create the queue to push the commands to
	const cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

	//add the kernel file
	cl::Program::Sources sources;
	AddSources(sources, "kernel.cl");

	//load the kernel with the device that will be used
	cl::Program program(context, sources);

	//build the program with error handling
	try
	{
		program.build();
	}
	catch (const cl::Error& err) //error handling
	{
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(
			context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(
			context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
			context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		throw;
	}

	{
		//generate an intensity histogram bin
		std::vector<int> histoBin(256);

		//get bin size
		const auto histoBinSize = sizeof(int) * histoBin.size();

		//buffer for the histogram and images
		cl::Buffer histoBuffer(context, CL_MEM_READ_WRITE, histoBinSize);
		cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, inputImage.size());
		cl::Buffer outputImageBuffer(context, CL_MEM_READ_ONLY, inputFilename.size());
		cl::Buffer histoBufferScan(context, CL_MEM_READ_WRITE, histoBinSize);

		//display bin sizes
		std::cout << "Bin Size: " << histoBin.size() << std::endl;
		std::cout << "Bin Size in Bytes: " << histoBinSize << std::endl;

		//write input image's data to the memory
		int bin_size = histoBin.size();
		if (bin_size == 256)
		{
			queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, inputImage.size(), &inputImage.data()[0]);
			queue.enqueueFillBuffer(histoBuffer, 0, 0, histoBinSize);
		}

		//load the histogram from the kernel
		cl::Kernel histoKernel = cl::Kernel(program, "histogram");

		//arguments for image buffer as input and histogram as output
		histoKernel.setArg(0, inputImageBuffer);
		histoKernel.setArg(1, histoBuffer);

		//kernel execution
		cl::Event profile;
		queue.enqueueNDRangeKernel(histoKernel, cl::NullRange, cl::NDRange(inputImage.size()),
			cl::NullRange, nullptr, &profile);

		{
			//kernel execution time
			std::cout << "Kernel execution time:" << profile.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				profile.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		}
		//move histogram from memory to the vector created earlier
		queue.enqueueReadBuffer(histoBuffer, CL_TRUE, 0, histoBinSize, &histoBin[0]);

		// generate a cumulative histogram
		std::vector<int> cumulativeHistoBin(histoBin.size());

		//get bin size
		auto cumulativeHistoSize = sizeof(int) * cumulativeHistoBin.size();

		//buffer for the cumulative histogram
		cl::Buffer cumulativeHistoBuffer(context, CL_MEM_READ_WRITE, cumulativeHistoSize);
		cl::Buffer cumulativeHistoBufferScan(context, CL_MEM_READ_WRITE, cumulativeHistoSize);

		// use the histogram buffer to write to memory
		if (bin_size == 256)
		{
			queue.enqueueWriteBuffer(histoBuffer, CL_TRUE, 0, histoBinSize, &histoBin[0]);
			queue.enqueueFillBuffer(cumulativeHistoBuffer, 0, 0, cumulativeHistoSize);
		}

		//load from the kernel
		cl::Kernel cumulativeHistoKernel = cl::Kernel(program, "scan_add_atomic");
		
		// arguments for the buffers
		cumulativeHistoKernel.setArg(0, histoBuffer); 
		cumulativeHistoKernel.setArg(1, cumulativeHistoBuffer);

		// kernel execution
		queue.enqueueNDRangeKernel(cumulativeHistoKernel, cl::NullRange, cl::NDRange(histoBin.size()), cl::NDRange(256));

		//move the result
		queue.enqueueReadBuffer(cumulativeHistoBuffer, CL_TRUE, 0, cumulativeHistoSize, &cumulativeHistoBin[0]);

		//normalise the cumulative histogram
		std::vector<int> normalisationBin(cumulativeHistoBin.size());
		
		//calculate normalisation bin size
		const auto normalisationBinSize = sizeof(int) * normalisationBin.size();

		//buffer for the normalised histogram
		cl::Buffer normalisationBuffer(context, CL_MEM_READ_WRITE, normalisationBinSize);

		//use the buffer to write the data to memory
		if (bin_size == 256)
		{
			queue.enqueueWriteBuffer(cumulativeHistoBuffer, CL_TRUE, 0, cumulativeHistoSize, &cumulativeHistoBin[0]);
			queue.enqueueFillBuffer(normalisationBuffer, 0, 0, normalisationBinSize);
		}

		//load the normalisation from the kernel
		cl::Kernel normalisationKernel = cl::Kernel(program, "normalisationBins");
		normalisationKernel.setArg(0, cumulativeHistoBuffer); 
		normalisationKernel.setArg(1, normalisationBuffer);

		// execution of the normalisation
		queue.enqueueNDRangeKernel(normalisationKernel, cl::NullRange, cl::NDRange(cumulativeHistoBin.size()),
		                           cl::NDRange(256));

		//move the normalisation result
		queue.enqueueReadBuffer(normalisationBuffer, CL_TRUE, 0, normalisationBinSize, &normalisationBin[0]);

		//create LUT LookUp Table for back-projection with vector and buffer to store values
		vector<unsigned char> outputImage(inputImage.size());
		cl::Buffer outputImageLUTBuffer(context, CL_MEM_READ_WRITE, inputImage.size());

		// write to the buffer
		if (bin_size == 256)
		{
			queue.enqueueWriteBuffer(normalisationBuffer, CL_TRUE, 0, normalisationBinSize, &normalisationBin[0]);
		}

		//load the LUT from the kernel and load the image data stored in the buffers
		cl::Kernel kernelLut = cl::Kernel(program, "lut");
		kernelLut.setArg(0, inputImageBuffer);
		kernelLut.setArg(1, outputImageLUTBuffer);
		kernelLut.setArg(2, normalisationBuffer); 

		//execute
		queue.enqueueNDRangeKernel(kernelLut, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange);

		//copying the data
		queue.enqueueReadBuffer(outputImageLUTBuffer, CL_TRUE, 0, outputImage.size(), &outputImage[0]);

		//output the image
		CImg<unsigned char> output_image(outputImage.data(), inputImage.width(), inputImage.height(),
			inputImage.depth(), inputImage.spectrum());
		CImgDisplay outputImageDisplay(output_image, "Output");

		//make the program window remain open
		while (!display.is_closed() && !outputImageDisplay.is_closed() && !display.is_keyESC() && !outputImageDisplay.
			is_keyESC())
		{
			display.wait(1);
			display.wait(1);
		}
		return 0;
	}
}