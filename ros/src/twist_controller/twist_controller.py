
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.throttle_controller = kwargs['throttle_controller']
        self.brake_controller = kwargs['brake_controller']
        self.steering_controller = kwargs['steering_controller']

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        sample_time = kwargs['sample_time']
        target_velocity = kwargs['linear_velocity']
        angular_velocity = kwargs['angular_velocity']
        current_velocity = kwargs['current_velocity']

        error = target_velocity - current_velocity

        throttle = self.throttle_controller.step(error, sample_time)
        brake = self.brake_controller.step(-error, sample_time)
        steering = self.steering_controller.get_steering(target_velocity, angular_velocity, current_velocity)
        return throttle, brake, steering

    def reset(self):
        self.throttle_controller.reset()
        self.brake_controller.reset()
