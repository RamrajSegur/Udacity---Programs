/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "helper_functions.h"

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 100;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	//std::vector<Particle> particles;
	//ParticleFilter pf1;
	for (unsigned int i = 0; i < num_particles; i++)
	{
		
		double sample_x, sample_y, sample_theta;
		Particle pf2;
		pf2.id = i;
		pf2.x = dist_x(gen);
		pf2.y = dist_y(gen);
		pf2.theta = dist_theta(gen);
		pf2.weight = 1.0;
		particles.push_back(pf2);
		
	}
	is_initialized = true;
	//std::cout << "Done Initializing" << std::endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	default_random_engine gen;
	//std::cout << "I am at the start of the prediction step" << std::endl;
	for (unsigned int i = 0; i < particles.size(); i++)
	{
		if (fabs(yaw_rate) < 0.0001)
		{
			particles[i].x += velocity*delta_t*cos(particles[i].theta);
			particles[i].y += velocity*delta_t*sin(particles[i].theta);
		}
		else
		{
			particles[i].x += (velocity / yaw_rate)*((sin(particles[i].theta + yaw_rate*delta_t)) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t)));
			particles[i].theta += yaw_rate*delta_t;

		}
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
		particles[i].weight = 1.0;
	}
	//std::cout << "Done Prediction" << std::endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++)
	{
		LandmarkObs obs = observations[i];
		//double min_dist = numeric_limits<double>::max();
		double min_dist = 1000.0;
		int id_min;
		for (unsigned int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];
			double current_distance = dist(obs.x, obs.y, pred.x, pred.y);
			if (current_distance < min_dist)
			{
				min_dist = current_distance;
				id_min = pred.id;
			}
		}
		observations[i].id = id_min;
	}
}

double ParticleFilter::gauss(double p_x, double p_y, double o_x, double o_y, double st_x, double st_y)
{
	double x_term = ((p_x - o_x)*(p_x - o_x)) / (2 * (st_x*st_x));
	double y_term = ((p_y - o_y)*(p_y - o_y)) / (2 * (st_y*st_y));
	double main_term = 1 / (2 * M_PI*st_x*st_y);
	return (main_term*exp(-(x_term + y_term)));

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//std::cout << "I am at start of update function" << std::endl;

	for (unsigned int i = 0; i < num_particles; i++)
	{
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		vector<LandmarkObs> predictions;

		// Selecting the landmarks within the given sensor range and its ID.

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			double map_x = map_landmarks.landmark_list[j].x_f;
			double map_y = map_landmarks.landmark_list[j].y_f;
			int map_id = map_landmarks.landmark_list[j].id_i;
			double distance = sqrt(pow(map_x - p_x, 2) + pow(map_y - p_y, 2));
			if(distance<=sensor_range) // Sensor range seggregation
			{
				predictions.push_back(LandmarkObs{map_id,map_x,map_y});
			}
		}

		vector<LandmarkObs> trans_obs;

		// Transforming the observation 

		for (unsigned int k = 0; k < observations.size(); k++)
		{
			double trans_x = particles[i].x + cos(particles[i].theta)*observations[k].x - sin(particles[i].theta)*observations[k].y;
			double trans_y = particles[i].y + sin(particles[i].theta)*observations[k].x + cos(particles[i].theta)*observations[k].y;
			int obs_id = observations[k].id;
			trans_obs.push_back(LandmarkObs{ obs_id,trans_x,trans_y });
		}
		 
		dataAssociation(predictions, trans_obs); // Finding the associated landmark and get the ID (Stored in trans_obs.id

		particles[i].weight = 1.0; // Weight Initialization

		for (unsigned int l = 0; l < trans_obs.size(); l++)
		{
			double pred_x, pred_y;
			double trans_obs_x = trans_obs[l].x;
			double trans_obs_y = trans_obs[l].y;
			int trans_id = trans_obs[l].id; // This Id is changed and calculated effectively based on its association.

			for (unsigned int m = 0; m < predictions.size(); m++)
			{
				if (predictions[m].id == trans_id) //Checking the correct ID of close landmark
				{
					pred_x = predictions[m].x; // Taking the close landmark values
					pred_y = predictions[m].y;
					
				}
			}
				// Calculating the probability -- weight corresponding to the particular landmark and observation
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			//double prob = gauss(trans_obs_x, trans_obs_y, pred_x, pred_y, std_x, std_y);
			double prob = (1 / (2 * M_PI*std_x*std_y)) * exp(-(pow(pred_x - trans_obs_x, 2) / (2 * pow(std_x, 2)) + (pow(pred_y - trans_obs_y, 2) / (2 * pow(std_y, 2)))));
			particles[i].weight *= prob;
			//cout << "Particles_weight:  " << particles[i].weight << endl;
		}

	}

	//std::cout << "I am at the end of update function" << endl;

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//cout << "I am at the start of resample function" << endl;

	default_random_engine gen;

	vector<Particle> new_particles;

	vector<double> weights;

	double beta=0.0;

	for (unsigned int i = 0; i < particles.size(); i++)
	{
		weights.push_back(particles[i].weight);
	}

	int num_of_particles = particles.size();

	discrete_distribution<int> dist_index(0, num_of_particles - 1);

	double weights_max = *max_element(weights.begin(), weights.end());

	auto index = dist_index(gen);

	//uniform_real_distribution<double> unirealdist(0.0, weights_max);

	normal_distribution<double> dist_weights(0.0, weights_max);
	//cout << "I am in the middle of resample" << endl;
	for (unsigned int i = 0; i < num_of_particles; i++)
	{
		//beta+=2.0*unirealdist(gen);
		beta += 2.0 *dist_weights(gen);
		while (beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_of_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
	//cout << "I am at the end of resample function" << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
