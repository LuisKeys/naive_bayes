#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "Eigen/Dense"
#include "classifier.h"

using namespace std;
using Eigen::VectorXd;

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

	// Get mean and deviation
	//Counter for the 3 labels values
	std::vector<double> counter = {0.0, 0.0, 0.0}; //left, keep and right
	std::vector<double> sum = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right
	std::vector<double> mean = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right

	std::vector<double> var = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right

	std::vector<double> sd = {0.0, 0.0, 0.0, 0.0, 
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right

	std::vector<string> names = {"s Left", "d Left", "s\' Left", "d\' Left", 
								"s Keep", "d Keep", "s\' Keep", "d\' Keep", 
								"s Right", "d Right", "s\' Right", "d\' Right"};

	int offset = 4;
	for(int i = 0; i < data.size(); ++i) {
		if(labels[i] == "left"){
			counter[0]++;
			for(int j = 0; j < 4; ++j)
				sum[j] += data[i][j];
		}
		if(labels[i] == "keep"){
			counter[1]++;
			for(int j = 0; j < 4; ++j)
				sum[j + offset] += data[i][j];
		}
		if(labels[i] == "right"){
			counter[2]++;
			for(int j = 0; j < 4; ++j)
				sum[j + 2 * offset] += data[i][j];
		}
	}

	// Mean
	for(int i = 0; i < 3; ++i){
		for(int j = 0; j < 4; ++j){
			mean[j + i * offset] = sum[j + i * offset] / counter[i];
		}
	}

	// standard deviation
	for(int i = 0; i < data.size(); ++i) {
		if(labels[i] == "left") {
			for(int j = 0; j < 4; ++j){
				var[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
			}
		}		

		if(labels[i] == "keep") {
			for(int j = 0; j < 4; ++j){
				var[j + offset] += (data[i][j] - mean[j + offset]) * (data[i][j] - mean[j + offset]);
			}
		}

		if(labels[i] == "keep") {
			for(int j = 0; j < 4; ++j){
				var[j + 2 * offset] += (data[i][j] - mean[j + 2 * offset]) * (data[i][j] - mean[j + 2 * offset]);
			}
		}
	}

	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < 4; ++j) {
			var[j + i * offset] /= counter[i];
			sd[j + i * offset] = sqrt(var[j + i * offset]);
				cout << "Value:" << names[j + i * offset] << " Mean:" << mean[j + i * offset] << " sd:" << sd[j + i * offset] << endl;
		}
	}

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
}

string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/

	return this->possible_labels[1];
}
