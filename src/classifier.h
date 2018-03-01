#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

	vector<string> possible_labels = {"left","keep","right"};


	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  	string predict(vector<double>);
  	
	//Counter for the 3 labels values
	std::vector<double> counter = {0.0, 0.0, 0.0}; //left, keep and right

	// Sum by feature and label
	std::vector<double> sum = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right
	// Mean by feature and label
	std::vector<double> mean = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right

	// Variance by feature and label
	std::vector<double> var = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right
	// Standard deviation by feature and label
	std::vector<double> sd = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right

	// Prior probability
	std::vector<double> pp = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right

	std::vector<string> names = {"s Left", "d Left", "s\' Left", "d\' Left", 
								"s Keep", "d Keep", "s\' Keep", "d\' Keep", 
								"s Right", "d Right", "s\' Right", "d\' Right"};

};

#endif
