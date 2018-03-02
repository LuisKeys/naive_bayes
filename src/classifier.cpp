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
	int index = 0.0;
	for(int i = 0; i < data.size(); ++i) {
		if(labels[i] == _left){
			counter[0]++;
			for(int j = 0; j < 4; ++j)
				sum[j] += data[i][j];
		}
		if(labels[i] == _keep){
			counter[1]++;
			for(int j = 0; j < 4; ++j)
				sum[j + offset] += data[i][j];
		}
		if(labels[i] == _right){
			counter[2]++;
			for(int j = 0; j < 4; ++j)
				sum[j + 2 * offset] += data[i][j];
		}
	}

	// Mean
	for(int i = 0; i < 3; ++i){
		for(int j = 0; j < 4; ++j){
			index = j + i * offset;
			mean[index] = sum[index] / counter[i];
			// cout << "Val:" << names[index] << " m:" << mean[index] << endl;
		}
	}

	// Standard deviation
	for(int i = 0; i < data.size(); ++i) {
		if(labels[i] == _left) {
			for(int j = 0; j < 4; ++j){
				var[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
			}
		}		

		if(labels[i] == _keep) {
			for(int j = 0; j < 4; ++j){
				index = j + offset;
				var[index] += (data[i][j] - mean[index]) * (data[i][j] - mean[index]);
			}
		}

		if(labels[i] == _right) {
			for(int j = 0; j < 4; ++j){
				index = j + 2 * offset;
				var[index] += (data[i][j] - mean[index]) * (data[i][j] - mean[index]);
			}
		}
	}

	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < 4; ++j) {
			index = j + i * offset;

			var[index] /= counter[i];
			sd[index] = sqrt(var[index]);

			 cout << "Val:" << names[index] << " m:" << mean[index] 
			 << " sd:" << sd[index] << endl;
		}
	}

	for(int i = 0; i < 3; ++i) {
		pp[i] = counter[i] / data.size();
		cout << " pp:" << pp[i] << endl;
	}
}

string GNB::predict(vector<double> sample)
{

	int offset = 4;
	double exp_param = 0.0;
	double _2_x_PI = 2 * _PI;
	double comb_prod = 1.0;
	double max_cp = -1.0;
	double cp = -1.0;
	int max_cp_index = 0;

	// k = conditional probability factor
	std::vector<double> k = {0.0, 0.0, 0.0, 0.0, //s, d, s', d' left
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right

	// p = conditional probability
	std::vector<double> p = {0.0, 0.0, 0.0, 0.0, //s, d, s', d' left
								0.0, 0.0, 0.0, 0.0,  //s, d, s', d' keep
								0.0, 0.0, 0.0, 0.0, }; //s, d, s', d' right

	// Conditional probability
	for(int i = 0; i < 12; ++i){
		k[i] = 1.0 / sqrt(_2_x_PI * var[i]);
	}

	for(int i = 0; i < 3; ++i){
		for(int j = 0; j < 4; ++j){
			int index = j + i * offset;
			exp_param = (sample[j] - mean[index]);
			exp_param *= exp_param;
			exp_param /= (2 * var[index]);
			p[j + i * offset] = k[index] * exp(-exp_param);
		}
	}
	

	for(int i = 0; i < 3; ++i) {
		for(int j = 0; j < 4; ++j){
			int index = j + i * offset;
			comb_prod *= p[index];
		}
		cp = pp[i] * comb_prod;
		comb_prod = 1.0;
		if(max_cp < cp) {
			max_cp = cp;
			max_cp_index = i;
		}
	}

	return this->possible_labels[max_cp_index];
}
