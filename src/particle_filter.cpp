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
  
  // yaw_rate = std::abs(yaw_rate) < 0.000001 ? 0 : yaw_rate;
  // const double term = 0.5 * yaw_rate;
  // const double delta_yaw = yaw_rate * delta_t;
  // const double coef = yaw_rate != 0 ? (2 * velocity / yaw_rate) * std::sin(0.5 * delta_yaw) : velocity * delta_t;
  // for (Particle& particle : particles) 
  // {
  //   particle.x += coef * std::cos(particle.theta + term) + normal_distribution(gen) * std_pos[0];
  //   particle.y += coef * std::sin(particle.theta + term) + normal_distribution(gen) * std_pos[1];
  //   particle.theta += delta_yaw + normal_distribution(gen) * std_pos[2];
  // }
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
          // cout << "Found landmark in range. Distance is: " << dist(lm_x, lm_y, particles[i].x, particles[i].y) << "\n";
          predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
        }
      }
      // cout << "Found "<< landmarksInRange.size() << " landmarks in range\n";

      // Transform observations to MAP coordinate system
      std::vector<LandmarkObs> map_observations;

      double prediction_x, prediction_y, observation_x, observation_y = 0;


      for(int k = 0; k < observations.size(); k++)
      {
        double x_map = cos(particles[i].theta)*observations[k].x - sin(particles[i].theta)*observations[k].y + particles[i].x;
        double y_map = sin(particles[i].theta)*observations[k].x + cos(particles[i].theta)*observations[k].y + particles[i].y;
        map_observations.push_back(LandmarkObs{observations[k].id, x_map, y_map});
        // cout << "New coordinates: (" << x_map << "," << y_map << ")\n";
      }
      
      // Associate observations and predictions
      dataAssociation(predictions, map_observations);
      particles[i].weight = 1.0;

      // For each observation find the corresponding landmark and update the weight
      // cout << "I have " << map_observations.size() << " map observations and " << predictions.size() << " predictions \n";
      // for(int m = 0; m < map_observations.size(); m++){
      //   cout << "observation " << m << " is (" << map_observations[m].x << "," << map_observations[m].y << ") \n";
      // }
      // for(int m = 0; m < predictions.size(); m++){
      //   cout << "observation " << m << " is (" << predictions[m].x << "," << predictions[m].y << ") \n";
      // }
      // return;

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

            const double x_exp = std::exp(-pow2(observation_x - prediction_x) * 0.5 / pow2(noise_x));
            const double y_exp = std::exp(-pow2(observation_y - prediction_y) * 0.5 / pow2(noise_y));
            particles[i].weight *= gaussNorm * x_exp * y_exp;
            // cout << "Calculated particle weight is " << particles[i].weight << "\n";
            // std::cout << "obs.x - prd.x= " << observation_x - prediction_x << " obs.y - prd.y= " <<observation_y - prediction_y << "\n";

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
  // cout << "Entering for...\n";
  for(int i = 0; i < num_particles; i++)
  {
    // cout << "Creating p\n";
    int tempIndex = randIndexGenerator(gen);
    // cout << "Genereted number: " << tempIndex << "\n";
    // cout << "This particle has id: " << particles[tempIndex].id << "\n";
    Particle* p = &particles[tempIndex];
    // Particle* p = &particles[0];
    // cout << "Creating new particle\n";
    sampledParticles[i] = Particle{p->id, p->x, p->y, p->theta, p->weight, p->associations, p->sense_x, p->sense_y};

  }
  // cout << "Assigning...\n";
  // particles.clear();
  particles = sampledParticles;
  // cout << "Resampling done\n";
}

// void ParticleFilter::SetAssociations(Particle& particle, 
//                                      const vector<int>& associations, 
//                                      const vector<double>& sense_x, 
//                                      const vector<double>& sense_y) {
//   // particle: the particle to which assign each listed association, 
//   //   and association's (x,y) world coordinates mapping
//   // associations: The landmark id that goes along with each listed association
//   // sense_x: the associations x mapping already converted to world coordinates
//   // sense_y: the associations y mapping already converted to world coordinates
//   particle.associations= associations;
//   particle.sense_x = sense_x;
//   particle.sense_y = sense_y;
// }

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