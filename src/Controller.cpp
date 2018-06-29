/*
 * Author: yulia
 * Date: 18/11/2016
 */

#include "Controller.hpp"

Controller::Controller(){
	env = NULL;
	resetOptions();
}

void Controller::resetOptions() {
	configFileOptSet = false;
	saveDirOptSet = false;
	saveScreenOptSet = false;
	saveFixOptSet = false;
	inputDirOptSet = false;
	inputImgOptSet = false;
	overwriteOptSet = false;
	pix2degOptSet = false;
	viewDistOptSet = false;
	inputSizeDegOptSet = false;
	displayFixOptSet = false;
}

void Controller::printParameters() {
	cout << "-------------------STAR-FC parameters------------------------" << endl;
	cout << "Input parameters" << endl;
	if (inputDirOptSet)
		cout << "\tInput directory:\t\t\t" << inputDirname << endl;
	else
		cout << "\tInput image:\t\t\t" << inputFilename << endl;

	cout << "\nAttention map parameters" << endl;
	cout << "\tPeripheral saliency:\t\t\t" << BUSaliencyAlgorithm << " (" << AIMBasis << ")" << endl;
	cout << "\tCentral saliency:\t\t\tSALICON" << endl;

	cout << "\tPeripheral gain:\t\t\t" << periphGain << endl;

	cout << "\tBlending strategy:\t\t\t";
	switch(blendingStrategy) {
		case 1:	cout << "SAR" << endl;
				break;
		case 2: cout << "MCA" << endl;
				break;
		case 3: cout << "WCA" << endl;
				break;
	}

	if (nextFixAsMax)
		cout << "\tNext fixation selection:\t\tdeterministic (as conspicuity maximum)" << endl;
	else
		cout << "\tNext fixation selection:\t\trandom (from conspicuity map above " << nextFixThresh << " threshold)" << endl;

	//TODO: diameter or radius
	cout << "\tPeripheral attentional field:\t\t" << periphAttFieldSizeDeg << " dva" << endl;
	cout << "\tCentral attentional field:\t\t" << centralAttFieldSizeDeg << " dva" << endl;
	cout << "\tSize of inhibition of return (IoR):\t" << iorSizeDeg << " dva" << endl;
	cout << "\tDecay rate of the IoR:\t\t\t" << iorDecayRate << endl;

	cout << "\nViewing parameters" << endl;
	if (pix2degOptSet)
		cout << "\tPixels per dva:\t\t\t" << pix2deg << endl;
	if (inputSizeDegOptSet)
		cout << "\tSize of the stimuli in dva:\t\t" << inputSizeDeg << endl;
	if (viewDistOptSet)
		cout << "\tViewing distance:\t\t\t" << viewDist << endl;
	cout << "\tMax number of fixations:\t\t" << maxNumFixations << endl;
	cout << "\tNumber of simulated subjects:\t\t" << numSubjects << endl;
	if (paddingR > 0)
		cout << "\tPadding:\t\t\t\t[" << paddingR << " " << paddingG << " " << paddingB << "]" << endl;
	else
		cout << "\tPadding:\t\t\t\t average input image RGB" << endl;

	cout << "\nLog parameters" << endl;
	if (saveDirOptSet) {
		cout << "\tOutput directory:\t\t\t" << saveDir << endl;
		if (overwriteOptSet)
			cout << "\tWARNING: existing results will be overwritten!" << endl;
		else
			cout << "\tWARNING: exsisting results will not be recomputed!" << endl;
	} else {
		cout << "\tNot saving the results" << endl;
	}
}

void Controller::parseCommandLineOptions(int argc, char *argv[]) {

	options_description desc{"Options"};
	desc.add_options()
		("help", "produce this help message")
		("configFile", value<string>(), "path to config file (e.g. config_files/test.ini)")
		("inputDir", value<string>(), "path to input directory with test images")
		("inputImg", value<string>(), "path to input image")
		("outputDir", value<string>(), "path to output directory (will be created if it doesn't exist)")
		("display", bool_switch()->default_value(false), "show image with fixations");
	try {
	  variables_map vm;
      store(parse_command_line(argc, argv, desc), vm);
	  notify(vm);
	  	if (vm.count("help")) {
	  		cout << desc << endl;
		}
		if (vm.count("configFile")) {
			configFileOptSet = true;
			configFilename = vm["configFile"].as<string>();
		}
		if (vm.count("inputDir")) {
			inputDirOptSet = true;
			inputDirname = vm["inputDir"].as<string>();
		}
		if (vm.count("inputImg")) {
			inputImgOptSet = true;
			inputFilename = vm["inputImg"].as<string>();
		}
		if (vm.count("outputDir")) {
			saveDirOptSet = true;
			saveDir = vm["outputDir"].as<string>();
		}

		displayFixOptSet = vm["display"].as<bool>();

	} catch (const exception &ex) {
		cerr << ex.what() << endl;
	}

	if (configFileOptSet) {
		parseConfigFile();
	} else {
		std::cout << desc << '\n';
		exit(-1);
	}

	if (inputImgOptSet && inputDirOptSet) {
		cout << "\033[1;31mERROR: conflicting options provided. Cannot run with both input and batch in the .ini file.\033[0m" << endl;
		std::cout << desc << '\n';
		exit(-1);
	}

	if (!inputImgOptSet && !inputDirOptSet) {
		cout << "\033[1;31mERROR: provide input image (--inputImg) or input directory (--inputDir).\033[0m" << endl;
		std::cout << desc << '\n';
		exit(-1);
	}

	if (pix2degOptSet && inputSizeDegOptSet) {
		cout << "\033[1;31mERROR: conflicting options (pix2deg and inputSizeDeg) provided. Only one should be present in the .ini file!\033[0m" << endl;
		std::cout << desc << '\n';
		exit(-1);
	}

	if (!viewDistOptSet || (!pix2degOptSet && !inputSizeDegOptSet)) {
		cout << "\033[1;31mERROR: viewing parameters (viewDist, pix2deg, inputSizeDeg) are not set! Specify the viewing distance and either pix2deg or inputSizeDeg in the .ini file.\033[0m" << endl;
		std::cout << desc << '\n';
		exit(-1);
	}

}

void Controller::parseConfigFile() {
	INIReader reader(configFilename);
	if (reader.ParseError() < 0) {
		cout << "ERROR: Cannot load " << configFilename << "!" << endl;
		exit(-1);
	}

	//parse attention_map_params
	BUSaliencyAlgorithm = reader.Get("attention_map_params", "BUSalAlgorithm", "AIM");
	AIMBasis = reader.Get("attention_map_params", "AIMBasis", "21infomax950");
	blendingStrategy = reader.GetInteger("attention_map_params", "blendingStrategy", 1);
	nextFixAsMax = reader.GetBoolean("attention_map_params", "nextFixAsMax", true);
	nextFixThresh = reader.GetReal("attention_map_params", "nextFixThresh", 0.95);
	periphGain = reader.GetReal("attention_map_params", "pgain", 1.15);
	centerGain = reader.GetReal("attention_map_params", "cgain", 1.0);
	periphAttFieldSizeDeg = reader.GetReal("attention_map_params", "pSizeDeg", 9.5);
	centralAttFieldSizeDeg = reader.GetReal("attention_map_params", "cSizeDeg", 9.6);
	iorSizeDeg = reader.GetReal("attention_map_params", "iorSizeDeg", 1.5);
	iorDecayRate = reader.GetReal("attention_map_params", "iorDecayRate", 100);

	//parse viewing params
	pix2deg = reader.GetReal("viewing_params", "pix2deg", -1);
	inputSizeDeg = reader.GetReal("viewing_params", "inputSizeDeg", -1);
	viewDist = reader.GetReal("viewing_params", "viewDist", -1);
	maxNumFixations = reader.GetInteger("viewing_params", "maxNumFixations", 720);
	numSubjects = reader.GetInteger("viewing_params", "numSubjects", 1);
	paddingR = reader.GetInteger("viewing_params", "paddingR", -1);
	paddingG = reader.GetInteger("viewing_params", "paddingG", -1);
	paddingB = reader.GetInteger("viewing_params", "paddingB", -1);

	paddingRGB = Scalar(paddingR, paddingG, paddingB);

	if (pix2deg > 0) {
		pix2degOptSet = true;
	}
	if (inputSizeDeg > 0) {
		inputSizeDegOptSet = true;
	}
	if (viewDist > 0) {
		viewDistOptSet = true;
	}

	//parse log_params

	saveFixOptSet = reader.GetBoolean("log_params", "saveFix", "on");
	saveScreenOptSet = reader.GetBoolean("log_params", "saveScreen", "off");
	fixLogName = reader.Get("log_params", "fixLogName", "fixationList");
	partNumFixations = reader.GetInteger("log_params", "partNumFixations", 0);
	overwriteOptSet = reader.GetBoolean("log_params", "overwrite", false);
}

/**
 * If the input directory is set, iterate through all files (non-recursively) and
 * put all png and jpg images into imageList vector (sorted alphabetically).
 */
void Controller::getImageList() {
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (inputDirname.c_str())) != NULL) {
	  /* find all images in the directory */
	  while ((ent = readdir (dir)) != NULL) {
		  if (strstr(ent->d_name, ".jpg") || strstr(ent->d_name, ".png")) {
			  imageList.push_back(string(ent->d_name));
			  //cout << string(ent->d_name) << endl;
		  }
	  }
	  closedir(dir);
	} else {
	  printf("ERROR: The directory %s does not exist!\n", inputDirname.c_str());
	  exit(-1);
	}
	sort(imageList.begin(), imageList.end());
}

/**
 * Initialize Controller and all maps using the dimensions of the input image.
 * Returns true if the image should be processed (when the image is new or overwrite option is set) and false otherwise
 *
 */
bool Controller::init(string imgFilename) {

	boost::filesystem::path path(imgFilename);
	imgBasename = path.stem().string();

	resultsDirExists(imgFilename);

	inputFilename = imgFilename;
	if (viewDistOptSet && (pix2degOptSet || inputSizeDegOptSet)) {
		env = new Environment(viewDist, inputSizeDeg, pix2deg, paddingRGB);
	} else {
		//env = new Environment();
		cerr << "ERROR: viewing parameters are not set!" << endl;
		exit(-1);
	}
	env->loadStaticStimulus(imgFilename);
	eye = new Eye(env);
	fixHistMap = new FixationHistoryMap(env->getHeight(), env->getWidth(), env->getFullHeight(), env->getFullWidth(),
										iorDecayRate, iorSizeDeg, 1/env->getPix2Deg());
	periphMap = new PeripheralAttentionalMap(env->getHeight(), env->getWidth(), 1/env->getPix2Deg(), periphAttFieldSizeDeg, BUSaliencyAlgorithm, AIMBasis);
	centralMap = new CentralAttentionalMap(env->getHeight(), env->getWidth(), 1/env->getPix2Deg(), centralAttFieldSizeDeg);
#ifndef WITH_SALICON
	centralMap->loadFaceTemplates();
#endif
	conspMap = new ConspicuityMap(env->getHeight(), env->getWidth(), periphGain, centerGain);
	priorityMap = new PriorityMap(env->getHeight(), env->getWidth(), periphGain, centerGain, blendingStrategy, nextFixAsMax, nextFixThresh);

	makeResultsDir(imgFilename);
	return true;
}


/*
 * Make directory for saving the results.
 * Depending on what results are being saved.
 */
void Controller::makeResultsDir(string imgFilename) {
	boost::filesystem::path path(imgFilename);
	imgBasename = path.stem().string();
	boost::system::error_code ec;

	if (saveFixOptSet) {
		fixDir = saveDir + "/fixations/" + imgBasename;
		if (!boost::filesystem::is_directory(fixDir)) {
			if (boost::filesystem::create_directories(fixDir), ec) {
				if (ec) {
					cout << "Could not create new output directory " << "(" << fixDir << "). Will not log fixations." << endl;
					cout << ec << endl;
				}
			}
		}
	}

	if (saveScreenOptSet) {
		screenshotDir = saveDir + "/frames/" + imgBasename;
		if (!boost::filesystem::is_directory(screenshotDir)) {
			if (boost::filesystem::create_directories(screenshotDir, ec)) {
				if (ec) {
					cout << "Could not create directory for screenshots" << "(" << screenshotDir << "). Saving to output/frames." << endl;
					cout << ec << endl;
				}
			}
		}
	}
}


/*
 * Check if the results directory exists and is non-empty (given that the appropriate flags for saving the results are set).
 */
bool Controller::resultsDirExists(string imgFilename) {
    imgName = imgFilename;
	boost::filesystem::path path(imgFilename);
	imgBasename = path.stem().string();

	string tempFixSaveDir = saveDir + "/fixations/" + imgBasename;

	if (saveFixOptSet) {
		if (overwriteOptSet) {
			boost::filesystem::remove_all(tempFixSaveDir); //delete all previous results
			cout << "Overwriting results for " << imgFilename << "!" << endl;
		} else {
			if (boost::filesystem::exists(tempFixSaveDir)) { //check if the results directory exists
																		//and subdirectories for fixations and/or screenshots are non-empty
				if ((saveFixOptSet && !boost::filesystem::is_empty(tempFixSaveDir))) {
			    	cout << "Skipping " << imgFilename << "!" << endl;
			    	return true;
			    }
			}
		}
	}
	return false;
}

/**
 * Update the Environment and reset all maps using the dimensions of the new image.
 * Returns true if the image should be processed (when the image is new or overwrite option is set) and false otherwise
 * TODO: check if loading a new image changes any of the environment parameters and if these
 * changes should be propagated via reset functions
 */
bool Controller::update(string imgFilename) {
	imgName = imgFilename;
	//if the first image in the list was skipped, all structures must be initialized first
	if (!env) {
		return init(imgFilename);
	}

	inputFilename = imgFilename;
	env->loadStaticStimulus(imgFilename);
	eye->reset();
	fixHistMap->reset(env->getHeight(), env->getWidth(), env->getFullHeight(), env->getFullWidth());
	periphMap->reset(env->getHeight(), env->getWidth());
	centralMap->reset(env->getHeight(), env->getWidth());
	conspMap->reset(env->getHeight(), env->getWidth());
	priorityMap->reset(env->getHeight(), env->getWidth());

	makeResultsDir(imgFilename);
	return true;
}

void Controller::run() {

    //run STAR_FC on a single image
    if (inputImgOptSet) {
    	init(inputFilename);
		for (int j = 0; j < numSubjects; j++) {
			curSubject = j;
			if (j > 0)
				update(inputFilename);
			runSTAR_FC();
		}
    } else {
    	//run STAR_FC on all images in the input directory
    	getImageList();
		cout << "Found " << imageList.size() << " images" << endl;
    	string imgPath = inputDirname + "/" + imageList[0];
		bool status;
		bool proceed = !resultsDirExists(imgPath);
    	if (proceed) {
    		status = init(imgPath);
			cout << "initialized " << status << endl;
    		for (int j = 0; j < numSubjects; j++) {
    			curSubject = j;
    			if (curSubject > 0)
    				update(imgPath);

				runSTAR_FC();
    		}
    	}
    	for (int i = 1; i < imageList.size(); i++) {
    		imgPath = inputDirname + "/" + imageList[i];
        	proceed = !resultsDirExists(imgPath);
    		if (proceed) {
    			for (int j = 0; j < numSubjects; j++) {
    				curSubject = j;
    				update(imgPath);
    				runSTAR_FC();
    			}
    		}
    	}
    }

}


string Controller::getSaveDir() {
	return saveDir;
}

string Controller::getScreenshotDir() {
	return screenshotDir;
}


void Controller::runSTAR_FC() {

	Point prevGazeCoords = Point(env->getWidth()/2+1, env->getHeight()/2+1);
	eye->setGazeCoords(prevGazeCoords); //set gaze at center initially
	//fixHistMap->saveFixationCoords(eye->getGazeCoords());

	if (displayFixOptSet) {
		namedWindow("Image with fixations", CV_WINDOW_NORMAL);
		resizeWindow("Image with fixations", env->getWidth()/2,env->getHeight()/2);
	}

	cout << fixHistMap->getNumberOfFixations() << endl;
	while(fixHistMap->getNumberOfFixations() < maxNumFixations) {

		clock_t begin  = clock();
		eye->viewScene();
		clock_t end = clock();
		double elapsed_secs = double(end-begin)/CLOCKS_PER_SEC;

		cout << "Processing fixation " << fixHistMap->getNumberOfFixations() << " for image " << imgName << endl;
		cout << "Foveation took " << elapsed_secs << " sec" << endl;

		//QPixmap eyeViewPixmap = OpenCV_Qt::cvMatToQPixmap(eye->getFoveatedView());
		//saveToImg("debug/foveatedScene.png", eye->getFoveatedView());

		periphMap->computeBUSaliency(eye->getFoveatedView());
		periphMap->computePeriphMap(blendingStrategy==1);

		//QPixmap periphMapPixmap = OpenCV_Qt::cvMatToQPixmap(periphMap->getPeriphMap());

#ifdef WITH_SALICON
		centralMap->centralDetection(eye->getFoveatedView());
		centralMap->maskCentralDetection();
#else
		centralMap->detectFaces(eye->getFoveatedView());
		centralMap->maskFaceDetection();
#endif

		// QPixmap centralMapPixmap = OpenCV_Qt::cvMatToQPixmap(centralMap->getCentralMap());


		conspMap->computeConspicuityMap(periphMap->getPeriphMap(), centralMap->getCentralMap());
		//emit periphMapMaxSet(QString("periphMapMax: %1/%2").arg(periphMap->getPeriphMapMax(), 3, 'f', 2, '0').arg(conspMap->getScaledPeriphMapMax(), 3, 'f', 2, '0'));
		//emit centralMapMaxSet(QString("centralMapMax: %1/%2").arg(centralMap->getCentralMapMax(), 3, 'f', 2, '0').arg(conspMap->getScaledCentralMapMax(), 3, 'f', 2, '0'));


		//QPixmap conspMapPixmap = OpenCV_Qt::cvMatToQPixmap(conspMap->getConspMap());

		//cout << "current eye gaze x=" << eye->getGazeCoords().y << " y=" << eye->getGazeCoords().x << endl;
		priorityMap->computeNextFixationDirection(periphMap, centralMap, fixHistMap->getFixationHistoryMap(eye->getGazeCoords()));

		//QPixmap priorityMapPixmap = OpenCV_Qt::cvMatToQPixmap(priorityMap->getPriorityMapVis());

		prevGazeCoords = eye->getGazeCoords();
		eye->setGazeCoords(priorityMap->getNextFixationDirection());

		//cout << "New eye gaze x=" << eye->getGazeCoords().y << " y=" << eye->getGazeCoords().x << endl;

		fixHistMap->decayFixations();
		fixHistMap->saveFixationCoords(prevGazeCoords);

		//QPixmap fixHistPixmap = OpenCV_Qt::cvMatToQPixmap(fixHistMap->getFixationHistoryMap(eye->getGazeCoords()));

		env->drawFixation(eye->getGazeCoords());

		if (displayFixOptSet) {
			imshow("Image with fixations", env->getSceneWithFixations());
			waitKey(10);
		}
		//environmentPixmap = OpenCV_Qt::cvMatToQPixmap(env->getSceneWithFixations());
		//emit environmentChanged(environmentPixmap);
		//save_pixmap("debug/envWithFixations.png", environmentPixmap);

		//QString fixCountString = QString("Fixation count: %1/%2").arg(fixHistMap->getNumberOfFixations()).arg(maxNumFixations);

		if (saveFixOptSet && partNumFixations > 0) {
			// partial results logging
			if(fixHistMap->getNumberOfFixations()%partNumFixations == 0){
				if (numSubjects > 1) {
					string saveToMat = fixDir + "/" + fixLogName + "_partial_" + to_string(fixHistMap->getNumberOfFixations()) + "_" + to_string(curSubject) + ".mat";
					fixHistMap->dumpFixationsToMat(saveToMat.c_str());
					string saveToImg = fixDir + "/" + fixLogName + "_partial_" + to_string(fixHistMap->getNumberOfFixations()) + "_" + to_string(curSubject) + ".jpg";
					env->saveSceneWithFixations(saveToImg.c_str());

				} else {
					string saveToMat = fixDir + "/" + fixLogName + "_partial_" + to_string(fixHistMap->getNumberOfFixations()); fixHistMap->dumpFixationsToMat(saveToMat.c_str());
					string saveToImg = fixDir + "/" + fixLogName + "_partial_" + to_string(fixHistMap->getNumberOfFixations()) + ".jpg";
					env->saveSceneWithFixations(saveToImg.c_str());
				}
			}
		}
	}

	//final results logging
	if (numSubjects > 1) {
		string saveTo = fixDir + '/' + fixLogName + '_' + to_string(curSubject) + ".mat";
		fixHistMap->dumpFixationsToMat(saveTo.c_str());
	} else {
		string saveTo = fixDir + "/" + fixLogName + ".mat";
		fixHistMap->dumpFixationsToMat(saveTo.c_str());
	}

	if (displayFixOptSet) {
		destroyAllWindows();
	}
}
