/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include<limits>

#include "helper_functions.h"
#include <iostream>

using std::string;
using std::vector;
using std::normal_distribution;
using std::cout;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100; 


  double std_x, std_y, std_theta;
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // Create a normal (Gaussian) distribution for x, y, theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++)
  {
      Particle p;
      p.id = i;
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
      p.weight = 1.0;
      particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) 
{

  std::normal_distribution<double> normal_distribution(0, 1);

  if(yaw_rate < 0.00001)
  {
    for (Particle& particle : particles) 
    {
      particle.x += velocity * delta_t * std::cos(particle.theta) + normal_distribution(gen) * std_pos[0];
      particle.y += velocity * delta_t * std::sin(particle.theta) + normal_distribution(gen) * std_pos[1];
      particle.theta += normal_distribution(gen) * std_pos[2];
    }
  }
  
  else
  {
    for (Particle& particle : particles) 
    {
      particle.x += (2 * velocity / yaw_rate) * std::sin(0.5 * yaw_rate * delta_t) * std::cos(particle.theta + 0.5 * yaw_rate) + normal_distribution(gen) * std_pos[0];
      particle.y += (2 * velocity / yaw_rate) * std::sin(0.5 * yaw_rate * delta_t) * std::sin(particle.theta + 0.5 * yaw_rate) + normal_distribution(gen) * std_pos[1];
      particle.theta += yaw_rate * delta_t + normal_distribution(gen) * std_pos[2];
    }
  }

}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) 
{
    for (LandmarkObs& observation : observations) {
    double min_error = std::numeric_limits<double>::max();
    LandmarkObs minimumErrorPrediction;
    for (LandmarkObs& prediction : predicted) 
    {
      double error = dist(observation.x, observation.y, prediction.x, prediction.y);
      
      if (error < min_error) 
      {
        min_error = error;
        minimumErrorPrediction = prediction;
      }
    }
    observation.id = minimumErrorPrediction.id;
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {


  //Find all landmarks in the sensor range
  std::vector<LandmarkObs> predictions;
  double noise_x = std_landmark[0];
	double noise_y = std_landmark[1];

  double gaussNorm = 1.0 / 2*M_PI*noise_x*noise_y;

  for(int i =0; i < particles.size(); i++)
  {
      // cout << "particle x: " << x << " particle y:" << y << "\n";
      predictions.clear();
      particles[i].associations.clear();
      particles[i].sense_x.clear();
      particles[i].sense_y.clear();

      for(int k = 0; k < map_landmarks.landmark_list.size(); k++)
      {
        double lm_x = map_landmarks.landmark_list[k].x_f;
        double lm_y = map_landmarks.landmark_list[k].y_f;
        double lm_id = map_landmarks.landmark_list[k].id_i;

        //check the distance
        if(dist(lm_x, lm_y, particles[i].x, particles[i].y) <= sensor_range)
        {
          predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
        }
      }

      // Transform observations to MAP coordinate system
      std::vector<LandmarkObs> map_observations;

      double prediction_x, prediction_y, observation_x, observation_y = 0;


      for(int k = 0; k < observations.size(); k++)
      {
        double x_map = cos(particles[i].theta)*observations[k].x - sin(particles[i].theta)*observations[k].y + particles[i].x;
        double y_map = sin(particles[i].theta)*observations[k].x + cos(particles[i].theta)*observations[k].y + particles[i].y;
        map_observations.push_back(LandmarkObs{observations[k].id, x_map, y_map});
      }
      
      // Associate observations and predictions
      dataAssociation(predictions, map_observations);
      particles[i].weight = 1.0;

      for (int k = 0; k < map_observations.size(); k++)
      {
        observation_x = map_observations[k].x;
        observation_y = map_observations[k].y;
        for(int n =0; n < predictions.size(); n++)
        {
          if(map_observations[k].id == predictions[n].id)
          {
            prediction_x = predictions[n].x;
            prediction_y = predictions[n].y;

            const double x_exp = std::exp(-1 * pow(observation_x - prediction_x, 2) * 0.5 / pow(noise_x, 2));
            const double y_exp = std::exp(-pow(observation_y - prediction_y, 2) * 0.5 / pow(noise_y, 2));
            particles[i].weight *= gaussNorm * x_exp * y_exp;

            particles[i].associations.push_back(predictions[n].id);
            particles[i].sense_x.push_back(predictions[n].x);
            particles[i].sense_y.push_back(predictions[n].y);
            break;
          }
        }
      }
  }
}

void ParticleFilter::resample() {

  // cout << "Resampling...\n";
  std::vector<double> weights(num_particles, 1.0);
  //Update the weights vector
  for(int i = 0; i < num_particles; i++)
  {
    weights[i]= particles[i].weight;
    // cout << "Weight[ "<< i << "] is " << weights[i] << "\n";
  }

  std::discrete_distribution<int> randIndexGenerator(weights.begin(), weights.end());

  std::vector<Particle> sampledParticles(num_particles);

  for(int i = 0; i < num_particles; i++)
  {
    int tempIndex = randIndexGenerator(gen);
    Particle* p = &particles[tempIndex];

    sampledParticles[i] = Particle{p->id, p->x, p->y, p->theta, p->weight, p->associations, p->sense_x, p->sense_y};

  }
  particles = sampledParticles;
}


string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}