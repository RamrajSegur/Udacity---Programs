#include "kalman_filter.h"
#include <iostream>
#include <math.h>
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
	//cout << "Prediction" << endl;
	x_ = F_*x_;
	MatrixXd F_T = F_.transpose();
	P_ = F_*P_*F_T + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
	//cout << "Laser Update" << endl;
	int xsize = x_.size();
	MatrixXd I = MatrixXd::Identity(xsize, xsize);
	VectorXd y = z - H_*x_;
	MatrixXd S = H_*P_*H_.transpose() + R_;
	MatrixXd K = P_*H_.transpose()*S.inverse();
	x_ = x_ + K*y;
	P_ = (I - K*H_)*P_.transpose();

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */	
	//cout << "Radar Update" << endl;
	int xsize = x_.size();
	MatrixXd I = MatrixXd::Identity(xsize, xsize);
	double rho = sqrt((x_(0)*x_(0)) + (x_(1)*x_(1)));
	double phi = atan2(x_(1), x_(0));
	double rhodot = ((x_(0)*x_(2)) + (x_(1)*x_(3))) / rho;
	VectorXd H = VectorXd(3);
	H << rho, phi, rhodot;
	VectorXd y = z - H;
	while (y(1) > M_PI)
	{
		y(1) -= 2 * M_PI;
	}
	while (y(1) < -M_PI)
	{
		y(1) += 2 * M_PI;
	}
	MatrixXd S = H_*P_*H_.transpose() + R_;
	MatrixXd K = P_*H_.transpose()*S.inverse();
	x_ = x_ + K*y;
	P_ = (I - K*H_)*P_.transpose();
}
