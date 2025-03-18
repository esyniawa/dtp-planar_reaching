import os
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import create_jacobian, create_dh_matrix


class PlanarArms:
    # joint limits
    l_upper_arm_limit, u_upper_arm_limit = np.radians((-5., 175.))  # in degrees [°]
    l_forearm_limit, u_forearm_limit = np.radians((-5., 175.))  # in degrees [°]

    # DH parameter
    scale = 1.0
    shoulder_length = scale * 50.0  # in [mm]
    upper_arm_length = scale * 220.0  # in [mm]
    forearm_length = scale * 160.0  # in [mm]

    # visualisation parameters
    x_limits = (-450, 450)
    y_limits = (-50, 400)

    def __init__(self,
                 init_angles_left: np.ndarray,
                 init_angles_right: np.ndarray,
                 radians: bool = False):

        """Constructor: initialize current joint angles, positions and trajectories"""
        if isinstance(init_angles_left, tuple | list):
            init_angles_left = np.array(init_angles_left)
        if isinstance(init_angles_right, tuple | list):
            init_angles_right = np.array(init_angles_right)

        self.angles_left = self.check_values(init_angles_left, radians)
        self.angles_right = self.check_values(init_angles_right, radians)

        self.trajectory_thetas_left = [self.angles_left]
        self.trajectory_thetas_right = [self.angles_right]

        self.trajectory_gradient_left = [np.array((0., 0.))]
        self.trajectory_gradient_right = [np.array((0., 0.))]

        self.end_effector_left = [PlanarArms.forward_kinematics(arm='left',
                                                                thetas=self.angles_left,
                                                                radians=True)[:, -1]]
        self.end_effector_right = [PlanarArms.forward_kinematics(arm='right',
                                                                 thetas=self.angles_right,
                                                                 radians=True)[:, -1]]

    @staticmethod
    def check_values(angles: np.ndarray, radians: bool):
        
        """ Check whether the sample joints are ok. """
        
        # Check number of joints
        assert angles.size == 2, "Arm must contain two angles: angle shoulder, angle elbow"

        # Check if joint are in rad not in degree
        if not radians:
            angles = np.radians(angles)

        # Check joint limits
        if angles[0] < PlanarArms.l_upper_arm_limit or angles[0] > PlanarArms.u_upper_arm_limit:
            raise AssertionError('Check joint limits for upper arm')
        elif angles[1] < PlanarArms.l_forearm_limit or angles[1] > PlanarArms.u_forearm_limit:
            raise AssertionError('Check joint limits for forearm')

        return angles

    @staticmethod
    def clip_values(angles: np.ndarray, radians: bool):
        assert angles.size == 2, "Arm must contain two angles: angle shoulder, angle elbow"

        if not radians:
            angles = np.radians(angles)

        angles[0] = np.clip(angles[0], a_min=PlanarArms.l_upper_arm_limit, a_max=PlanarArms.u_upper_arm_limit)
        angles[1] = np.clip(angles[1], a_min=PlanarArms.l_forearm_limit, a_max=PlanarArms.u_forearm_limit)

        return angles if radians else np.degrees(angles)

    @staticmethod
    def get_bounds():
        return (np.array((PlanarArms.l_upper_arm_limit, PlanarArms.l_forearm_limit)),
                np.array((PlanarArms.u_upper_arm_limit, PlanarArms.u_forearm_limit)))

    @staticmethod
    def __circular_wrap(x: float, x_min: int | float, x_max: int | float):
        # Calculate the range of the interval
        interval_range = x_max - x_min

        # Calculate the wrapped value of x
        wrapped_x = x_min + ((x - x_min) % interval_range)

        return wrapped_x

    @staticmethod
    def circ_values(thetas: np.ndarray, radians: bool = True):
        """
        This wrapper function is intended to prevent phase jumps in the inverse kinematics due to large errors in the
        gradient calculation. This means that joint angles are only possible within the given limits.

        :param thetas:
        :param radians:
        :return:
        """
        if not radians:
            theta1, theta2 = np.radians(thetas)
        else:
            theta1, theta2 = thetas

        theta1 = PlanarArms.__circular_wrap(x=theta1,
                                            x_min=PlanarArms.l_upper_arm_limit,
                                            x_max=PlanarArms.u_upper_arm_limit)

        theta2 = PlanarArms.__circular_wrap(x=theta2,
                                            x_min=PlanarArms.l_forearm_limit,
                                            x_max=PlanarArms.u_forearm_limit)

        return np.array((theta1, theta2))

    @staticmethod
    def forward_kinematics(arm: str, thetas: np.ndarray, radians: bool = False, check_limits: bool = True):

        # Check whether sampled joint are okay
        if check_limits:
            theta1, theta2 = PlanarArms.check_values(thetas, radians)
        else:
            theta1, theta2 = thetas

        if arm == 'right':
            const = 1
        elif arm == 'left':
            const = - 1
            theta1 = np.pi - theta1
            theta2 = - theta2
        else:
            raise ValueError('Please specify if the arm is right or left!')

        # Create DH matrices
        A0 = create_dh_matrix(a=const * PlanarArms.shoulder_length, d=0,
                              alpha=0, theta=0)

        A1 = create_dh_matrix(a=PlanarArms.upper_arm_length, d=0,
                              alpha=0, theta=theta1)

        A2 = create_dh_matrix(a=PlanarArms.forearm_length, d=0,
                              alpha=0, theta=theta2)

        # Shoulder -> elbow
        A01 = A0 @ A1
        # Elbow -> hand
        A12 = A01 @ A2

        return np.column_stack(([0, 0], A0[:2, 3], A01[:2, 3], A12[:2, 3]))

    @staticmethod
    def inverse_kinematics(arm: str,
                           end_effector: np.ndarray,
                           starting_angles: np.ndarray,
                           learning_rate: float = 0.01,
                           max_iterations: int = 5000,
                           abort_criteria: float = 1,  # in [mm]
                           radians: bool = False):

        if not radians:
            starting_angles = np.radians(starting_angles)

        thetas = starting_angles.copy()
        for i in range(max_iterations):
            # Compute the forward kinematics for the current joint angles
            current_position = PlanarArms.forward_kinematics(arm=arm,
                                                             thetas=thetas,
                                                             radians=True)[:, -1]

            # Calculate the error between the current end effector position and the desired end point
            error = end_effector - current_position

            # abort when error is smaller than the breaking condition
            if np.linalg.norm(error) < abort_criteria:
                break

            # Calculate the Jacobian matrix for the current joint angles
            J = create_jacobian(thetas=thetas, arm=arm,
                                a_sh=PlanarArms.upper_arm_length,
                                a_el=PlanarArms.forearm_length,
                                radians=True)

            delta_thetas = learning_rate * np.linalg.inv(J) @ error
            thetas += delta_thetas
            # prevent phase jumps due to large errors
            thetas = PlanarArms.circ_values(thetas, radians=True)

        return thetas

    @staticmethod
    def random_theta(return_radians=True):
        """
        Returns random joint angles within the limits.
        """
        theta1 = np.random.uniform(PlanarArms.l_upper_arm_limit, PlanarArms.u_upper_arm_limit)
        theta2 = np.random.uniform(PlanarArms.l_forearm_limit, PlanarArms.u_forearm_limit)

        if return_radians:
            return np.array((theta1, theta2))
        else:
            return np.degrees((theta1, theta2))

    @staticmethod
    def random_position(arm: str):
        """
        Returns random position within the joint limits.
        """
        new_theta = PlanarArms.random_theta()
        return PlanarArms.forward_kinematics(arm=arm, thetas=new_theta, radians=True)[:, -1]

    @staticmethod
    def __cos_space(start: float | np.ndarray, stop: float | np.ndarray, num: int):
        """
        For the calculation of gradients and trajectories. Derivation of this function is sin(x),
        so that the maximal change in the middle of the trajectory.
        """

        if isinstance(start, np.ndarray) and isinstance(stop, np.ndarray):
            if not start.size == stop.size:
                raise ValueError('Start and stop vector must have the same dimensions.')

        # calc changes
        offset = stop - start

        # function to modulate the movement.
        if isinstance(start, np.ndarray):
            x_lim = np.repeat(np.pi, repeats=start.size)
        else:
            x_lim = np.pi

        x = - np.cos(np.linspace(0, x_lim, num, endpoint=True)) + 1.0
        x /= np.amax(x)

        # linear space
        y = np.linspace(0, offset, num, endpoint=True)

        return start + x * y

    def reset_all(self):
        """Reset position to default and delete trajectories"""
        self.__init__(init_angles_left=self.trajectory_thetas_left[0],
                      init_angles_right=self.trajectory_thetas_right[0],
                      radians=True)

    def reset_arm_to_angle(self, arm: str, thetas: np.ndarray, radians: bool = True):
        """
        Set the joint angle of one arm to a new joint angle without movement. Thetas = must be in degrees
        """
        thetas = PlanarArms.check_values(thetas, radians=radians)
        if arm == 'right':
            self.__init__(init_angles_right=thetas,
                          init_angles_left=self.trajectory_thetas_left[-1],
                          radians=True)

        elif arm == 'left':
            self.__init__(init_angles_left=thetas,
                          init_angles_right=self.trajectory_thetas_right[-1],
                          radians=True)
        else:
            raise ValueError('Please specify if the arm is right or left!')

    def set_trajectory(self, arm: str, trajectory: list):
        """
        Set trajectory of arm to a set list
        """

        if arm == 'right':
            self.reset_arm_to_angle(arm=arm, thetas=trajectory[0], radians=True)
            self.trajectory_thetas_right = trajectory[0] + trajectory
            self.angles_right = trajectory[-1]

            self.trajectory_thetas_left = [self.angles_left] * len(self.trajectory_thetas_right)

        elif arm == 'left':
            self.reset_arm_to_angle(arm=arm, thetas=trajectory[0], radians=True)
            self.trajectory_thetas_left = trajectory[0] + trajectory
            self.angles_left = trajectory[-1]

            self.trajectory_thetas_right = [self.angles_right] * len(self.trajectory_thetas_left)

        else:
            raise ValueError('Please specify if the arm is right or left!')

    def change_angle(self, arm: str, new_thetas: np.ndarray, num_iterations: int = 100, radians: bool = False,
                     break_at: None | int = None):
        """
        Change the joint angle of one arm to a new joint angle.
        """
        new_thetas = self.check_values(new_thetas, radians=radians)
        if arm == 'right':

            trajectory = self.__cos_space(start=self.angles_right, stop=new_thetas, num=num_iterations)

            for j, delta_theta in enumerate(trajectory):
                self.trajectory_gradient_right.append(delta_theta - self.trajectory_thetas_right[-1])
                self.trajectory_gradient_left.append(np.array((0., 0.)))

                self.trajectory_thetas_right.append(delta_theta)
                self.trajectory_thetas_left.append(self.angles_left)

                self.end_effector_right.append(PlanarArms.forward_kinematics(arm='right',
                                                                             thetas=delta_theta,
                                                                             radians=True)[:, -1])
                self.end_effector_left.append(self.end_effector_left[-1])

                if break_at == j:
                    break

            # set current angle to the new thetas
            self.angles_right = self.trajectory_thetas_right[-1]

        elif arm == 'left':

            trajectory = self.__cos_space(start=self.angles_left, stop=new_thetas, num=num_iterations)[:, -1]

            for j, delta_theta in enumerate(trajectory):
                self.trajectory_gradient_left.append(delta_theta - self.trajectory_thetas_left[-1])
                self.trajectory_gradient_right.append(np.array((0., 0.)))

                self.trajectory_thetas_left.append(delta_theta)
                self.trajectory_thetas_right.append(self.angles_right)

                self.end_effector_left.append(PlanarArms.forward_kinematics(arm='left',
                                                                            thetas=delta_theta,
                                                                            radians=True)[:, -1])
                self.end_effector_right.append(self.end_effector_right[-1])

                if break_at == j:
                    break

            # set current angle to the new thetas
            self.angles_left = self.trajectory_thetas_left[-1]

        else:
            raise ValueError('Please specify if the arm is right or left!')

    def change_position_straight(self, moving_arm: str,
                                 new_position: np.ndarray,
                                 num_iterations: int = 100,
                                 break_at: None | int = None) :

        """
        Change the joint angle of one arm to a new position.
        """

        if moving_arm == 'right':
            current_pos = self.end_effector_right[-1]

            angle, distance = PlanarArms.calc_motor_vector(init_pos=current_pos, end_pos=new_position,
                                                           arm=moving_arm)

            trajectory = self.__cos_space(start=0.0, stop=distance, num=num_iterations)

            for j, delta_distance in enumerate(trajectory):
                new_pos = PlanarArms.calc_position_from_motor_vector(init_pos=current_pos, angle=angle,
                                                                     norm=delta_distance, arm=moving_arm, radians=False)

                new_theta = PlanarArms.inverse_kinematics(arm=moving_arm, end_effector=new_pos,
                                                          starting_angles=self.trajectory_thetas_right[-1],
                                                          radians=True)

                self.trajectory_gradient_right.append(new_theta - self.trajectory_thetas_right[-1])
                self.trajectory_gradient_left.append(np.array((0., 0.)))

                self.trajectory_thetas_right.append(new_theta)
                self.trajectory_thetas_left.append(self.angles_left)

                self.end_effector_right.append(new_pos)
                self.end_effector_left.append(self.end_effector_left[-1])

                if break_at == j:
                    break

            # set current angle to the new thetas
            self.angles_right = self.trajectory_thetas_right[-1]

        elif moving_arm == 'left':
            current_pos = self.end_effector_left[-1]

            angle, distance = PlanarArms.calc_motor_vector(init_pos=current_pos, end_pos=new_position,
                                                           arm=moving_arm)

            trajectory = self.__cos_space(start=0.0, stop=distance, num=num_iterations)

            for j, delta_distance in enumerate(trajectory):
                new_pos = PlanarArms.calc_position_from_motor_vector(init_pos=current_pos, angle=angle,
                                                                     norm=delta_distance, arm=moving_arm, radians=False)

                new_theta = PlanarArms.inverse_kinematics(arm=moving_arm, end_effector=new_pos,
                                                          starting_angles=self.trajectory_thetas_left[-1],
                                                          radians=True)

                self.trajectory_gradient_left.append(new_theta - self.trajectory_thetas_left[-1])
                self.trajectory_gradient_right.append(np.array((0., 0.)))

                self.trajectory_thetas_left.append(new_theta)
                self.trajectory_thetas_right.append(self.angles_right)

                self.end_effector_left.append(new_pos)
                self.end_effector_right.append(self.end_effector_right[-1])

                if break_at == j:
                    break

            # set current angle to the new thetas
            self.angles_left = self.trajectory_thetas_left[-1]

        else:
            raise ValueError('Please specify if the arm is right or left!')

    def move_to_position(self, arm: str, end_effector: np.ndarray, num_iterations: int = 100):
        """
        Move to a certain coordinate within the peripersonal space.
        """
        if arm == 'right':
            new_thetas_to_position = self.inverse_kinematics(arm=arm, end_effector=end_effector,
                                                             starting_angles=self.angles_right, radians=True)

            self.change_angle(arm=arm, new_thetas=new_thetas_to_position, num_iterations=num_iterations, radians=True)

        elif arm == 'left':
            new_thetas_to_position = self.inverse_kinematics(arm=arm, end_effector=end_effector,
                                                             starting_angles=self.angles_left, radians=True)

            self.change_angle(arm=arm, new_thetas=new_thetas_to_position, num_iterations=num_iterations, radians=True)

    def move_to_position_and_return_to_init(self, arm: str, end_effector: np.ndarray, num_iterations: int = 400,
                                            t_wait: int = 5):
        """
        Move to a certain coordinate within the peripersonal space and return to the initial position.
        """

        num_iterations -= t_wait
        movement_time = int(num_iterations/2)

        if arm == 'right':
            new_thetas_to_position = self.inverse_kinematics(arm=arm, end_effector=end_effector,
                                                             starting_angles=self.angles_right, radians=True)

            self.wait(t_wait)
            self.change_angle(arm=arm, new_thetas=new_thetas_to_position, num_iterations=movement_time, radians=True)
            self.change_angle(arm=arm, new_thetas=self.trajectory_thetas_right[0], num_iterations=movement_time, radians=True)

        elif arm == 'left':
            new_thetas_to_position = self.inverse_kinematics(arm=arm, end_effector=end_effector,
                                                             starting_angles=self.angles_left, radians=True)

            self.wait(t_wait)
            self.change_angle(arm=arm, new_thetas=new_thetas_to_position, num_iterations=movement_time, radians=True)
            self.change_angle(arm=arm, new_thetas=self.trajectory_thetas_left[0], num_iterations=movement_time, radians=True)

    def change_to_position_in_trajectory_return_to_init(self, arm: str, end_effectors: list | tuple,
                                                        num_iterations: int = 400, t_wait: int = 5, break_at: int = 100):
        """
        Move to a certain coordinate within the peripersonal space, then change it in the middle of the trajectory to
        another position and return to the initial position.
        """

        num_iterations -= t_wait

        movement_time = int(num_iterations/2)
        movement_time_after_break = int(movement_time - break_at)

        if arm == 'right':
            new_thetas_to_position_1 = self.inverse_kinematics(arm=arm, end_effector=end_effectors[0],
                                                               starting_angles=self.angles_right, radians=True)

            new_thetas_to_position_2 = self.inverse_kinematics(arm=arm, end_effector=end_effectors[1],
                                                               starting_angles=self.angles_right, radians=True)

            self.wait(t_wait)
            # move pos 1
            self.change_angle(arm=arm, new_thetas=new_thetas_to_position_1, num_iterations=movement_time, radians=True,
                              break_at=break_at)
            # change to pos 2
            self.change_angle(arm=arm, new_thetas=new_thetas_to_position_2, num_iterations=movement_time_after_break,
                              radians=True)

            # return to init pos
            self.change_angle(arm=arm, new_thetas=self.trajectory_thetas_right[0], num_iterations=movement_time,
                              radians=True)

        elif arm == 'left':
            new_thetas_to_position_1 = self.inverse_kinematics(arm=arm, end_effector=end_effectors[0],
                                                               starting_angles=self.angles_left, radians=True)

            new_thetas_to_position_2 = self.inverse_kinematics(arm=arm, end_effector=end_effectors[1],
                                                               starting_angles=self.angles_left, radians=True)

            self.wait(t_wait)
            # move pos 1
            self.change_angle(arm=arm, new_thetas=new_thetas_to_position_1, num_iterations=movement_time, radians=True,
                              break_at=break_at)
            # change to pos 2
            self.change_angle(arm=arm, new_thetas=new_thetas_to_position_2, num_iterations=movement_time_after_break,
                              radians=True)
            # return to init pos
            self.change_angle(arm=arm, new_thetas=self.trajectory_thetas_left[0], num_iterations=movement_time,
                              radians=True)

    def touch_arm(self,
                  resting_arm: str,
                  joint_to_touch: int,
                  percentile: float,
                  num_iterations: int = 100):

        """
        Touch one point of the arm with the other arm. The variable joint_to_touch defines the modality:
        0 = shoulder; 1 = upper arm; 2 = forearm
        The variable percentile indicates the position relative to the modality.
        """

        if joint_to_touch not in [0, 1, 2]:
            raise ValueError('joint_to_touch must be either: 0 = shoulder; 1 = upper arm; 2 = forearm ')

        assert percentile >= 0, "percentile must be positive"
        assert percentile <= 1, "percentile must be lower than 1.0"

        if resting_arm == 'right':
            moving_arm = 'left'
            coordinates_rest_arm = self.forward_kinematics(arm=resting_arm, thetas=self.angles_right, radians=True)
        elif resting_arm == 'left':
            moving_arm = 'right'
            coordinates_rest_arm = self.forward_kinematics(arm=resting_arm, thetas=self.angles_left, radians=True)
        else:
            raise ValueError('Please specify if the arm is right or left!')

        # calculate coordinate to touch
        start = coordinates_rest_arm[:, joint_to_touch]
        end = coordinates_rest_arm[:, joint_to_touch + 1]

        touch_point = start + percentile * (end - start)

        # calculate movement trajectory to the touch point
        if moving_arm == 'right':
            self.move_to_position(arm=moving_arm,
                                  end_effector=touch_point,
                                  num_iterations=num_iterations)
        elif moving_arm == 'left':
            self.move_to_position(arm=moving_arm,
                                  end_effector=touch_point,
                                  num_iterations=num_iterations)

    def wait(self, time_steps: int):
        for t in range(time_steps):
            self.trajectory_thetas_right.append(self.angles_right)
            self.trajectory_thetas_left.append(self.angles_left)

            self.trajectory_gradient_right.append(np.array((0., 0.)))
            self.trajectory_gradient_left.append(np.array((0., 0.)))

            self.end_effector_right.append(self.end_effector_right[-1])
            self.end_effector_left.append(self.end_effector_left[-1])

    def training_trial(self,
                       arm: str,
                       position: np.ndarray,
                       t_min: int, t_max: int, t_wait: int = 10,
                       min_distance: float = 50.0,
                       trajectory_save_name: str = None):

        assert position.size == 2, "End effector should be 2-dimensional"

        distance = -1
        while distance <= min_distance:
            start_angles = PlanarArms.random_theta(return_radians=True)
            start_coordinate = PlanarArms.forward_kinematics(arm=arm,
                                                             thetas=start_angles,
                                                             radians=True)[:, -1]

            distance = np.linalg.norm(position - start_coordinate)

        time_interval = int(random.uniform(t_min, t_max))

        self.reset_arm_to_angle(arm=arm, thetas=start_angles, radians=True)
        self.move_to_position(arm=arm, end_effector=position, num_iterations=time_interval)
        self.wait(t_wait)
        if trajectory_save_name is not None:
            self.save_state(trajectory_save_name)

    def training_fixed_position(self,
                                arm: str,
                                position: np.ndarray,
                                training_trials: int,
                                t_min: int, t_max: int, t_wait: int = 10,
                                trajectory_save_name: str = None):

        assert position.size == 2, "End effector should be 2-dimensional"

        for trial in range(training_trials):

            if arm == 'right':
                self.reset_arm_to_angle(arm='right', thetas=self.trajectory_thetas_right[0], radians=True)
            elif arm == 'left':
                self.reset_arm_to_angle(arm='left', thetas=self.trajectory_thetas_left[0], radians=True)

            time_interval = int(random.uniform(t_min, t_max))

            self.move_to_position(arm=arm, end_effector=position, num_iterations=time_interval)
            self.wait(t_wait)
            if trajectory_save_name is not None:
                self.save_state(trajectory_save_name + f'_{trial}')

    def test_fixed_position(self,
                            arm: str,
                            position: np.ndarray,
                            t_min: int, t_max: int, t_wait: int = 10,
                            trajectory_save_name: str = None):

        assert position.size == 2, "End effector should be 2-dimensional"

        if arm == 'right':
            self.reset_arm_to_angle(arm='right', thetas=self.trajectory_thetas_right[0], radians=True)
        elif arm == 'left':
            self.reset_arm_to_angle(arm='left', thetas=self.trajectory_thetas_left[0], radians=True)

        time_interval = int(random.uniform(t_min, t_max))

        self.move_to_position(arm=arm, end_effector=position, num_iterations=time_interval)
        self.wait(t_wait)
        if trajectory_save_name is not None:
            self.save_state(trajectory_save_name)

    def save_state(self, data_name: str = None):
        import datetime

        d = {
            'trajectory_left': self.trajectory_thetas_left,
            'gradients_left': self.trajectory_gradient_left,
            'end_effectors_left': self.end_effector_left,

            'trajectory_right': self.trajectory_thetas_right,
            'gradients_right': self.trajectory_gradient_right,
            'end_effectors_right': self.end_effector_right
        }

        df = pd.DataFrame(d)

        if data_name is not None:
            folder, _ = os.path.split(data_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
        else:
            # get current date
            current_date = datetime.date.today()
            data_name = "PlanarArm_" + current_date.strftime('%Y%m%d')

        df.to_csv(data_name + '.csv', index=False)

    def import_state(self, file: str):
        df = pd.read_csv(file, sep=',')

        # convert type back to np.ndarray because pandas imports them as strings...
        regex_magic = lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=' ', dtype=float)
        for column in df.columns:
            df[column] = df[column].apply(regex_magic)

        # set states
        self.angles_left = df['trajectory_left'].tolist()[-1]
        self.angles_right = df['trajectory_right'].tolist()[-1]

        self.trajectory_thetas_left = df['trajectory_left'].tolist()
        self.trajectory_thetas_right = df['trajectory_right'].tolist()

        self.trajectory_gradient_left = df['gradients_left'].tolist()
        self.trajectory_gradient_right = df['gradients_right'].tolist()

        self.end_effector_left = df['end_effectors_left'].tolist()
        self.end_effector_right = df['end_effectors_right'].tolist()

    # Functions for visualisation
    def plot_current_position(self, plot_name=None, fig_size=(12, 8)):
        """
        Plots the current position of the arms.

        :param plot_name: Define the name of your figure. If none the plot is not saved!
        :param fig_size: Size of the Figure
        """
        coordinates_left = PlanarArms.forward_kinematics(arm='left', thetas=self.angles_left, radians=True)
        coordinates_right = PlanarArms.forward_kinematics(arm='right', thetas=self.angles_right, radians=True)

        fig, ax = plt.subplots(figsize=fig_size)

        ax.plot(coordinates_left[0, :], coordinates_left[1, :], 'b')
        ax.plot(coordinates_right[0, :], coordinates_right[1, :], 'b')

        ax.set_xlabel('x in [mm]')
        ax.set_ylabel('y in [mm]')

        ax.set_xlim(PlanarArms.x_limits)
        ax.set_ylim(PlanarArms.y_limits)

        # save
        if plot_name is not None:
            folder, _ = os.path.split(plot_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(plot_name)

        plt.show()

    def plot_trajectory(self, fig_size=(12, 8),
                        points: list | tuple | None = None,
                        save_name: str = None,
                        frames_per_sec: int = 10,
                        turn_off_axis: bool = False):
        """
        Visualizes the movements performed so far. Use the slider to set the time.

        :param fig_size:
        :param points:
        :param save_name: If not None, the trajectory is saved in a .gif or .mp4
        :param frames_per_sec:
        :param turn_off_axis:
        :return:
        """
        from matplotlib.widgets import Slider
        import matplotlib.animation as animation

        init_t = 0
        num_t = len(self.trajectory_thetas_left)

        coordinates_left = []
        coordinates_right = []

        for i_traj in range(num_t):
            coordinates_left.append(PlanarArms.forward_kinematics(arm='left',
                                                                  thetas=self.trajectory_thetas_left[i_traj],
                                                                  radians=True))

            coordinates_right.append(PlanarArms.forward_kinematics(arm='right',
                                                                   thetas=self.trajectory_thetas_right[i_traj],
                                                                   radians=True))

        fig, ax = plt.subplots(figsize=fig_size)

        if turn_off_axis:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xlabel('x in [mm]')
            ax.set_ylabel('y in [mm]')

        ax.set_xlim(PlanarArms.x_limits)
        ax.set_ylim(PlanarArms.y_limits)

        l, = ax.plot(coordinates_left[init_t][0, :], coordinates_left[init_t][1, :], 'b')
        r, = ax.plot(coordinates_right[init_t][0, :], coordinates_right[init_t][1, :], 'b')

        if points is not None:
            for point in points:
                ax.scatter(point[0], point[1], marker='+')

        val_max = num_t - 1

        if save_name is None:

            ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
            time_slider = Slider(
                ax=ax_slider,
                label='n iteration',
                valmin=0,
                valmax=val_max,
                valinit=0,
            )

            def update(val):
                t = int(time_slider.val)
                l.set_data(coordinates_left[t][0, :], coordinates_left[t][1, :])
                r.set_data(coordinates_right[t][0, :], coordinates_right[t][1, :])
                time_slider.valtext.set_text(t)

            time_slider.on_changed(update)

            plt.show()
        else:
            def animate(t):
                l.set_data(coordinates_left[t][0, :], coordinates_left[t][1, :])
                r.set_data(coordinates_right[t][0, :], coordinates_right[t][1, :])
                return r, l

            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, val_max))

            if save_name[-3:] == 'mp4':
                writer = animation.FFMpegWriter(fps=frames_per_sec)
            else:
                writer = animation.PillowWriter(fps=frames_per_sec)

            ani.save(save_name, writer=writer)
            plt.close(fig)

    @staticmethod
    def calc_motor_vector(init_pos: np.ndarray[float, float], end_pos: np.ndarray[float, float],
                          arm: str, input_theta: bool = False, theta_radians: bool = False):

        if input_theta:
            init_pos = PlanarArms.forward_kinematics(arm=arm, thetas=init_pos, radians=theta_radians)[:, -1]

        diff_vector = end_pos - init_pos
        angle = np.degrees(np.arctan2(diff_vector[1], diff_vector[0])) % 360
        norm = np.linalg.norm(diff_vector)

        return angle, norm, diff_vector

    @staticmethod
    def calc_position_from_motor_vector(init_pos: np.ndarray[float, float],
                                        angle: float,
                                        norm: float,
                                        radians: bool = False):

        x, y = init_pos

        if not radians:
            angle = np.radians(angle)

        new_position = np.array((
            norm * np.cos(angle) + x,
            norm * np.sin(angle) + y
        ))

        return new_position
