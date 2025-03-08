import numpy as np
import torch
from typing import Tuple, Optional
from torch import Tensor
from kinematics.planar_arms import PlanarArms
from collections import deque
import random


def input_transform(thetas: np.ndarray,
                    xy: np.ndarray) -> np.ndarray:
    """
    Transforms angles in radians to normalized values and concatenates with normalized xy coordinates
    For both joints, we normalize using the same limits since they're identical!!! If they're different, we need to
    normalize them separately

    :param thetas: initial joint angles in radians (batch_size, 2)
    :param xy: target cartesian coordinates (batch_size, 2)
    :return: normalized inputs in range [-1, 1] (batch_size, 4)
    """
    # For both joints, we normalize using the same limits since they're identical
    angles_normalized = 2 * (thetas - PlanarArms.l_upper_arm_limit) / (
            PlanarArms.u_upper_arm_limit - PlanarArms.l_upper_arm_limit) - 1

    # Normalize xy coordinates using existing function
    xy_normalized = norm_xy(xy)

    return np.concatenate((angles_normalized, xy_normalized))


def inverse_input_transform(inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms normalized inputs back to original space
    :param inputs: normalized inputs in range [-1, 1] (batch_size, 4)
    :return: tuple of (joint angles in radians, xy coordinates)
    """
    # Split inputs into angles and coordinates
    angles_norm = inputs[:, :2][0]
    xy_norm = inputs[:, 2:][0]

    # Convert normalized angles back to radians
    thetas = (angles_norm + 1) / 2 * (
            PlanarArms.u_upper_arm_limit - PlanarArms.l_upper_arm_limit) + PlanarArms.l_upper_arm_limit

    # Denormalize xy coordinates
    x_mid = (PlanarArms.x_limits[0] + PlanarArms.x_limits[1]) / 2
    y_mid = (PlanarArms.y_limits[0] + PlanarArms.y_limits[1]) / 2
    x_half_range = (PlanarArms.x_limits[1] - PlanarArms.x_limits[0]) / 2
    y_half_range = (PlanarArms.y_limits[1] - PlanarArms.y_limits[0]) / 2

    x = xy_norm[0] * x_half_range + x_mid
    y = xy_norm[1] * y_half_range + y_mid

    return thetas, np.array((x, y))


def target_transform(thetas: np.ndarray) -> np.ndarray:
    """
    Transforms angle changes to normalized range [-1, 1]
    :param thetas: angle changes in radians (batch_size, 2)
    :return: normalized angle changes (batch_size, 2)
    """
    # Maximum angle change is the full range (≈ 3.14 radians or 180 degrees)
    max_angle_change = np.abs(PlanarArms.u_upper_arm_limit - PlanarArms.l_upper_arm_limit)

    # Normalize the changes to [-1, 1]
    return thetas / max_angle_change


def inverse_target_transform(normalized_thetas: np.ndarray) -> np.ndarray:
    """
    Transforms normalized angle changes back to radians
    :param normalized_thetas: normalized angle changes in [-1, 1] range (batch_size, 2)
    :return: angle changes in radians (batch_size, 2)
    """
    # Maximum angle change
    max_angle_change = np.abs(PlanarArms.u_upper_arm_limit - PlanarArms.l_upper_arm_limit)

    # Convert back to radians
    return normalized_thetas * max_angle_change


def norm_xy(xy: np.ndarray,
            x_bounds: Tuple[float, float] = PlanarArms.x_limits,
            y_bounds: Tuple[float, float] = PlanarArms.y_limits, ) -> np.ndarray:
    # Calculate the midpoints of x and y ranges
    x_mid = (x_bounds[0] + x_bounds[1]) / 2
    y_mid = (y_bounds[0] + y_bounds[1]) / 2

    # Calculate the half-ranges
    x_half_range = (x_bounds[1] - x_bounds[0]) / 2
    y_half_range = (y_bounds[1] - y_bounds[0]) / 2

    # Normalize to [-1, 1]
    normalized_x = (xy[0] - x_mid) / x_half_range
    normalized_y = (xy[1] - y_mid) / y_half_range

    return np.array((normalized_x, normalized_y))


def generate_random_movement(arm: str, min_distance: float = 50.):
    # Random joint angles
    init_shoulder_thetas, target_shoulder_thetas = np.random.uniform(low=PlanarArms.l_upper_arm_limit,
                                                                     high=PlanarArms.u_upper_arm_limit,
                                                                     size=2)

    init_elbow_thetas, target_elbow_thetas = np.random.uniform(low=PlanarArms.l_forearm_limit,
                                                               high=PlanarArms.u_forearm_limit,
                                                               size=2)

    init_thetas = np.array((init_shoulder_thetas, init_elbow_thetas))
    target_thetas = np.array((target_shoulder_thetas, target_elbow_thetas))

    # Calculate distance
    init_pos = PlanarArms.forward_kinematics(arm=arm,
                                             thetas=init_thetas,
                                             radians=True)[:, -1]

    target_pos = PlanarArms.forward_kinematics(arm=arm,
                                               thetas=target_thetas,
                                               radians=True)[:, -1]

    distance = np.linalg.norm(target_pos - init_pos)

    # If distance is too small, call function again
    if distance <= min_distance:
        return generate_random_movement(arm=arm, min_distance=min_distance)
    else:
        return init_thetas, target_thetas, init_pos, target_pos


class MovementBuffer:
    def __init__(self,
                 arm: str,
                 buffer_size: int,
                 device: torch.device):

        self.arm = arm
        self.buffer_size = buffer_size
        self.device = device

        self.buffer = deque(maxlen=self.buffer_size)

    def fill_buffer(self, min_distance: float = 50.):
        while len(self.buffer) < self.buffer_size:
            init_thetas, target_thetas, _, target_pos = generate_random_movement(arm=self.arm, min_distance=min_distance)

            # Scale data
            inputs = input_transform(thetas=init_thetas, xy=target_pos)
            targets = target_transform(thetas=target_thetas - init_thetas)

            self.buffer.append((inputs, targets, init_thetas))

    def get_batches(self, batch_size: int) -> Tuple[Tensor, Tensor, np.ndarray]:
        # Get random batch
        inputs, targets, thetas = zip(*random.sample(self.buffer, batch_size))

        # Convert data to tensors and numpy
        inputs = torch.tensor(np.array(inputs), dtype=torch.float, device=self.device)
        targets = torch.tensor(np.array(targets), dtype=torch.float, device=self.device)
        init_thetas = np.array(thetas)

        return inputs, targets, init_thetas

    def __len__(self):
        return len(self.buffer)

    def clear_buffer(self):
        self.buffer = deque(maxlen=self.buffer_size)


def create_batch(arm: str,
                 min_distance: float = 50.,
                 device: torch.device = torch.device('cpu')) -> Tuple[Tensor, Tensor, np.ndarray]:

    """ Generate a single batch of inputs, targets, and initial thetas for evaluation """
    init_thetas, target_thetas, _, target_pos = generate_random_movement(arm=arm, min_distance=min_distance)

    # Scale data
    inputs = input_transform(thetas=init_thetas, xy=target_pos).reshape(1, -1)
    targets = target_transform(thetas=target_thetas - init_thetas).reshape(1, -1)

    return torch.tensor(inputs, dtype=torch.float, device=device), torch.tensor(targets, dtype=torch.float, device=device), init_thetas


if __name__ == "__main__":
    movements = MovementBuffer(arm='right', buffer_size=10000, device=torch.device('cpu'))
    movements.fill_buffer()
    inputs, targets, init_thetas = movements.get_batches(batch_size=8)

    print(inputs, targets, init_thetas)
