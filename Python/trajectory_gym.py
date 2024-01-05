import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


class TrajectoryGym():
    """
    This class is used to record trajectories from a Unity environment.
    It allows the user to record expert trajectories by playing the game
    and saving the trajectories to a file. These trajectories can then be
    used to train a transformer model.
    """

    # create dataset creator, then start the recording loop
    # IMPORTANT: the trajectory gym assumes a working and running environment!
    def __init__(self, env, filepath):
        self.filepath = filepath
        self.ds_creator = TrajectoryDatasetCreator(filepath)
        self.is_recording = True

    # while loop that only ends after the user gives an "end of game" EOG input
    # or the program crashes
    def recording_loop(self):
        # start loop
        while (self.is_recording):
            # wait for user input
            user_action, recording_stop = self.read_input()

            # if the user stops the recording with the EOG
            # finish the current trajectory and save the created dataset
            if recording_stop:
                self.is_recording = False
                self.ds_creator.finishTrajectory()
                self.ds_creator.saveTrajectories()
                break

            # TODO take action and send to environment

            # TODO get observation and send to dataset creator

    def read_input(self) -> ((int, int, int), bool):

        # get input from console
        # TODO define valid input and EOG symbol
        start_message = "todo: define"
        user_input = input(start_message)

        # create input loop
        user_action = None
        is_valid_input = False

        while (not is_valid_input):

            # INPUT SANITATION
            # delete leading and trailing whitespaces

            user_input = user_input.strip()

            # TODO EOG
            if (user_input == 'EOG'):
                # stop recording
                user_action = None
                is_valid_input = True
            # TODO valid input
            
            # invalid input
            else:
                print(f"invalid input: {user_input}")

        if user_action is None:
            return None, True
        else:
            return user_action, False

class TrajectoryDatasetCreator():
    """
    Creates and saves a dataset of trajectories created by the user.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.trajectories = []
        self.currentTrajectory = []

    def addTimestep(self, state, reward, action):
        # add given state, reward and action to the current trajectory
        raise NotImplementedError()

    def finishTrajectory(self):
        self.trajectories += [self.currentTrajectory]

    def saveTrajectories(self):
        # save list of trajectories to a file
        # should be loadable by a torch dataset for the transfomer!
        torch.save(self.trajectories, self.filepath)


if __name__ == '__main__':
    # open connection to unity ml agent
    print("Waiting for Unity environment...")
    env = UnityEnvironment()
    env = UnityToGymWrapper(env, allow_multiple_obs=True)
    print("Unity environment started successfully!")

    # create trajectory gym with this env
    traj_gym = TrajectoryGym(env, "dataset\\first_attempt")
    traj_gym.recording_loop()
    env.close()
