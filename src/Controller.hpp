/*
 * Author: yulia
 * Date: 18/11/2016
 */
#ifndef CONTROLLER_HPP_
#define CONTROLLER_HPP_


#include <ctime>
#include <vector>
#include <iostream>
#include <string>
#include <dirent.h>
#include <libgen.h>

#include <sys/types.h>
#include <sys/stat.h>

#include "INIReader.h" //parser for INI config files

#include <boost/filesystem/convenience.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options.hpp>

#include "CentralAttentionalMap.hpp"
#include "Environment.hpp"
#include "PeripheralAttentionalMap.hpp"
#include "ConspicuityMap.hpp"
#include "FixationHistoryMap.hpp"
#include "PriorityMap.hpp"
#include "Eye.hpp"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace boost::program_options;

class Controller {


private:
	Environment* env;
	Eye* eye;
	FixationHistoryMap* fixHistMap;
	PeripheralAttentionalMap* periphMap;
	CentralAttentionalMap* centralMap;
	ConspicuityMap* conspMap;
	PriorityMap* priorityMap;

	int maxNumFixations, numSubjects, curSubject, partNumFixations, blendingStrategy;
	string configFilename, inputFilename, inputDirname, saveDir, screenshotDir, fixDir, fixLogName;
	bool configFileOptSet, saveDirOptSet, saveScreenOptSet, saveFixOptSet, inputDirOptSet, inputImgOptSet, overwriteOptSet, displayFixOptSet;
	bool pix2degOptSet, viewDistOptSet, inputSizeDegOptSet, nextFixAsMax;
	double periphGain, centerGain, iorSizeDeg, iorDecayRate, nextFixThresh;
	double periphAttFieldSizeDeg, centralAttFieldSizeDeg;
	double pix2deg, viewDist, inputSizeDeg;
	int paddingR, paddingG, paddingB;
	Scalar paddingRGB;
	string BUSaliencyAlgorithm, AIMBasis;
	string imgBasename; //input filename without extension
	vector<string> imageList;
	string imgName;

public:
	Controller();
	bool init(string imgFilename);
	void run();
	void resetOptions();
	bool update(string imgFilename);
	void parseCommandLineOptions(int argc, char *argv[]);
	void printParameters();
	void parseConfigFile();
	string getSaveDir();
	string getScreenshotDir();
	void getImageList();
	void runSTAR_FC();
	void runSingleInput();
	void runBatchInput();
	bool resultsDirExists(string imgFilename);
	void makeResultsDir(string imgFilename);

};

#endif
