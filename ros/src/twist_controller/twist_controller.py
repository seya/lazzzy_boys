
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.throttle_controller = kwargs['throttle_controller']
        self.brake_controller = kwargs['brake_controller']
        self.steering_controller = kwargs['steering_controller']
        self.smooth_filter = kwargs['smoothing_filter']
        self.steering_adjustment_controller = kwargs['steering_adjustment_controller']
        self.previous_time = 0
        self.counter = 1

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        sample_time = kwargs['sample_time']
        proposed_linear_velocity = kwargs['proposed_velocity'][0]
        proposed_angular_velocity = kwargs['proposed_velocity'][1]
        current_linear_velocity = kwargs['current_velocity'][0]
        current_angular_velocity = kwargs['current_velocity'][1]

        # Throttle and Brake control
        linear_velocity_error = proposed_linear_velocity - current_linear_velocity

        throttle = self.throttle_controller.step(linear_velocity_error, sample_time)
        brake = self.brake_controller.step(-linear_velocity_error, sample_time)

        if proposed_linear_velocity == 0:
            throttle = 0

        # Steering control
        angular_velocity_error = proposed_angular_velocity - current_angular_velocity

        steering = self.steering_controller.get_steering(proposed_linear_velocity,
                                                         proposed_angular_velocity,
                                                         current_linear_velocity)

        steering = steering + self.steering_adjustment_controller.step(angular_velocity_error,
                                                                       sample_time)
        steering = self.smooth_filter.filt(steering)

        # Return values
        angular_velocity_error = proposed_angular_velocity - current_angular_velocity

        steering = self.steering_controller.get_steering(proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity)
        steering = steering + self.steering_adjustment_controller.step(angular_velocity_error, sample_time)
        steering = self.smooth_filter.filt(steering)
        return throttle, brake, steering

    def reset(self):
        self.throttle_controller.reset()
        self.brake_controller.reset()
        self.steering_adjustment_controller.reset()
