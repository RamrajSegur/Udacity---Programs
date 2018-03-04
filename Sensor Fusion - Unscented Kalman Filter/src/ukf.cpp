#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ =0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ =0.55;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;
  Xsig_pred_ = MatrixXd(5, 15);
  n_aug_ =7;
  n_x_ = 5;
  lambda_ = 3 - n_aug_;

  time_us_ = 0;
  //Xsig_pred = MatrixXd(5,15);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:
  
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	
	if (!is_initialized_)
	{	
		
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			
			// Convert radar from polar to cartesian coordinates and initialize state.
			float rho = meas_package.raw_measurements_[0]; // range
			float phi = meas_package.raw_measurements_[1]; // bearing
			float rho_dot = meas_package.raw_measurements_[2]; // velocity of rho
			// Coordinates convertion from polar to cartesian
			float px = rho * cos(phi); 
			float py = rho * sin(phi);
			float vx = rho_dot * cos(phi);
			float vy = rho_dot * sin(phi);
			float v  = sqrt(vx * vx + vy * vy);
			x_ << px, py, 0, 0, 0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
		{
			
			x_(0) = meas_package.raw_measurements_[0];
			x_(1) = meas_package.raw_measurements_[1];
		}
		previous_t= meas_package.timestamp_;
		is_initialized_ = true;
		P_ << 1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1; 

		x_(2) = 0;
		x_(3) = 0;
		x_(4) = 0;
		return;
	}
	
	time_us_ = meas_package.timestamp_;
	double delta_t = (time_us_ - previous_t)/1000000.0;

	Prediction(delta_t);

	if (meas_package.sensor_type_ == MeasurementPackage::LASER)
	{
		
		UpdateLidar(meas_package);			
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
	{
		
		UpdateRadar(meas_package);
	}

	previous_t = time_us_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	weights_ = VectorXd(15);
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (int i = 1; i < 15; i++)
	{
		weights_(i) = 1 / (2 * (lambda_ + n_aug_));
	}
	MatrixXd X_sig_aug = MatrixXd(7, 15); //Augmentation Matrix sigma declaration
	VectorXd X_aug = VectorXd(7); // State Augmentation Matrix Declaration
	for (int i = 0; i < 5; i++)
	{
		X_aug(i) = x_(i);
	}

	X_aug(5) = 0;
	X_aug(6) = 0;


	MatrixXd P_aug = MatrixXd(7, 7); // Augmented Covariance Matrix Declaration
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5,5) = P_;
	P_aug(5, 5) = std_a_*std_a_;
	P_aug(6, 6) = std_yawdd_*std_yawdd_;

	MatrixXd L = P_aug.llt().matrixL();

	X_sig_aug.fill(0.0);
	X_sig_aug.col(0) = X_aug; // Preparation of X_sigma_augmented Matrix
	for (int i = 0; i< 7; i++)
	{
		X_sig_aug.col(i + 1) = X_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		X_sig_aug.col(i + 1 + n_aug_) = X_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}


	/*Augmented sigma matrix prediction*/

	
	Xsig_pred_.fill(0.0);
	for (int i = 0; i< (2 * 7 + 1); i++)
	{
		//extract values for better readability
		double p_x = X_sig_aug(0, i);
		double p_y = X_sig_aug(1, i);
		double v = X_sig_aug(2, i);
		double yaw = X_sig_aug(3, i);
		double yawd = X_sig_aug(4, i);
		double nu_a = X_sig_aug(5, i);
		double nu_yawdd = X_sig_aug(6, i);

		//predicted state values
		double px_p, py_p;
		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}



		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		/*while (yaw_p> M_PI) yaw_p -= 2.*M_PI;
		while (yaw_p<-M_PI) yaw_p += 2.*M_PI;*/


		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	
	}
	weights_ = VectorXd(15);
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (int i = 1; i < 15; i++)
	{
		weights_(i) = 1 / (2 * (lambda_ + n_aug_));
	}

	/* State Mean Prediction */
	x_.fill(0.0);
	for (int i = 0; i<15; i++)
	{
		
		x_ += Xsig_pred_.col(i)*weights_(i);
	}



	P_.fill(0.0);
	/* Predict state covariance matrix */
	for (int i = 0; i<15; i++)
	{
		VectorXd xdiff=Xsig_pred_.col(i) - x_;
		MatrixXd calculated=weights_(i)*xdiff*xdiff.transpose();
		P_ += calculated;
	}


}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {


	VectorXd z = VectorXd(2);
	z(0) = meas_package.raw_measurements_[0];
	z(1) = meas_package.raw_measurements_[1];
	MatrixXd Zsig = MatrixXd(2, 15);
	Zsig.fill(0.0);
	VectorXd z_pred = VectorXd(2);
	MatrixXd S = MatrixXd(2, 2);
	Zsig=Xsig_pred_.block(0, 0, 2, 15);

	z_pred.fill(0.0);
	for (int i = 0; i<15; i++)
	{
		z_pred = z_pred + weights_(i)*Zsig.col(i);
	}

	S.fill(0.0);
	for (int i = 0; i<15; i++)
	{
		S = S + weights_(i)*(Zsig.col(i) - z_pred)*(Zsig.col(i) - z_pred).transpose();
	}
	MatrixXd R = MatrixXd(2, 2);
	R << std_laspx_*std_laspx_, 0,
		0, std_laspy_*std_laspy_;

	S = S + R;
	MatrixXd Tc = MatrixXd(5, 2);
	Tc.fill(0.0);
	for (int i = 0; i<15; i++)
	{
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
		Tc = Tc + weights_(i)*x_diff*(Zsig.col(i) - z_pred).transpose();
	}

	MatrixXd K = MatrixXd(5, 2);
	//calculate Kalman gain K;
	K = Tc * S.inverse();
	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;

	while (x_(3)> M_PI) x_(3) -= 2.*M_PI;
	while (x_(3)<-M_PI) x_(3) += 2.*M_PI;

	P_ = P_ - K*S*K.transpose();

	float NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

	
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
	VectorXd z = VectorXd(3);
	z(0) = meas_package.raw_measurements_[0];
	z(1) = meas_package.raw_measurements_[1];
	z(2) = meas_package.raw_measurements_[2];
	while (z(1)> M_PI) z(1) -= 2.*M_PI;
	while (z(1)<-M_PI) z(1) += 2.*M_PI;
	MatrixXd Zsig = MatrixXd(3, 15);
	VectorXd z_pred = VectorXd(3);
	MatrixXd S = MatrixXd(3, 3);
	Zsig.fill(0.0);
	for (int i = 0; i < 2 * 7 + 1; i++) {  //2n+1 simga points

											   // extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);
		
		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;
		const double eps = 0.01; 

		// measurement model
		Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
		Zsig(1, i) = atan2(p_y, p_x);                                 //phi
		Zsig(2, i) = (p_x*v1 + p_y*v2) / std::max(eps,sqrt(p_x*p_x + p_y*p_y));   //r_dot
	}

	z_pred.fill(0.0);
	//calculate mean predicted measurement
	for (int i = 0; i<15; i++)
	{
		z_pred = z_pred + weights_(i)*Zsig.col(i);
		/*while (z_pred(1)> M_PI) z_pred(1) -= 2.*M_PI;
		while (z_pred(1)<-M_PI) z_pred(1) += 2.*M_PI;*/
	}

	//calculate innovation covariance matrix S
	S.fill(0.0);
	for (int i = 0; i < 2 * 7 + 1; i++) {  //2n+1 simga points
											   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}
	MatrixXd R = MatrixXd(3, 3);
	R << std_radr_*std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0, std_radrd_*std_radrd_;
	S = S + R;
	MatrixXd Tc = MatrixXd(5, 3);
	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < 2 * 7 + 1; i++) {  //2n+1 simga points

											   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	MatrixXd K = MatrixXd(3, 3);
	//calculate Kalman gain K;
	//Kalman gain K;
	K = Tc * S.inverse();
	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K*S*K.transpose();
	
	while (x_(3)> M_PI) x_(3) -= 2.*M_PI;
	while (x_(3)<-M_PI) x_(3) += 2.*M_PI;

	float NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
	
}
