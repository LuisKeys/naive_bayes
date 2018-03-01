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
				cout << "Val:" << names[j + i * offset] << " m:" << mean[j + i * offset] 
				<< " sd:" << sd[j + i * offset] 
				<< endl;
		}
	}

	for(int i = 0; i < 3; ++i) {
		pp[i] = counter[i] / data.size();
		cout << " pp:" << pp[i] << endl;
	}
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
