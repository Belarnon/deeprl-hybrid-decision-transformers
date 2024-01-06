import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from typing import List, Tuple

class TrajectoryGym():
    """
    This class is used to record trajectories from a Unity environment.
    It allows the user to record expert trajectories by playing the game
    and saving the trajectories to a file. These trajectories can then be
    used to train a transformer model.
    """

    # create dataset creator, then start the recording loop
    # IMPORTANT: the trajectory gym assumes a working and running environment!
    def __init__(self, gridsize: int, nr_given_blocks: int, env: UnityToGymWrapper, filepath: str):
        self.gridsize = gridsize
        self.nr_given_blocks = nr_given_blocks
        self.env = env
        self.filepath = filepath
        self.ds_creator = TrajectoryDatasetCreator(filepath)

        # call reset on environment to get the first state s_0 for the beginning
        # of the dataset trajectory!
        state = self.env.reset()[0] # state is list with array
        self.ds_creator.addStartState(state)

    # while loop that only ends after the user gives an "end of game" EOG input
    # or the program crashes
    def recording_loop(self):
        # start loop
        while True:
            # wait for user input
            user_action, is_recording = self.read_input()
            #user_action, recording_stop = self.env.action_space.sample(), False

            # if the user stops the recording with the EOG
            # finish the current trajectory and save the created dataset
            if not is_recording: break

            # send action to environment, get observation
            state, reward, is_done, info = self.env.step(user_action)
            state = state[0]
            state_view = state[:100].reshape((10,10))
            
            # send observation to dataset creator to add to current trajectory
            # TODO make decision: do we add "invalid" actions that do not alter
            # the state since the block can not be placed at the chosen coordinates
            # or an empty block has been chosen?? but the reward DOES consider those actions...
            # -> currently we add every action to the trajectory
            self.ds_creator.addStep(state, user_action, reward)

            # check if a terminal state has been hit -> game over
            # then we must invoke the stop of current trajectory recording and call finish trajectory
            if is_done:
                self.ds_creator.finishTrajectory()
                state = self.env.reset()[0] # state is list with array
                self.ds_creator.addStartState(state)

        # EOG has been given
        self.ds_creator.finishTrajectory()
        self.ds_creator.saveTrajectories()


    def read_input(self) -> Tuple[List[int], bool]:

        # create input loop
        start_message = "input (format a x y) : "
        user_action = None
        is_valid_input = False

        while (not is_valid_input):

            # get input from console
            user_input = input(start_message)
            # INPUT SANITATION
            # delete leading and trailing whitespaces
            user_input = user_input.strip()

            # EOG
            if (user_input == 'EOG'):
                # stop recording
                user_action = None
                is_valid_input = True
                break
            # valid input: "a x y"
            # split by ' ' -> check if only 3 element
                
            # if not exactly 3 inputs are given or too much whitespaces
            user_input = user_input.split(' ')
            if len(user_input) != 3:
                print(f"too much/few arguments were given! too much whitespaces are also invalid: {user_input}")
            else:
                # for each element delete leading and trailing whitespaces again
                # also try to convert to int
                try:
                    a = int(user_input[0].strip()) # block index
                    x = int(user_input[1].strip()) # column
                    y = int(user_input[2].strip()) # row
                # if something goes wrong, print not integers and break out
                except:
                    print(f"some inputs were not integers!: {user_input}")
                    continue
                
                # check range
                try:
                    assert(0 <= a and a < self.nr_given_blocks)
                    assert(0 <= x and x < self.gridsize)
                    assert(0 <= y and y < self.gridsize)
                # if an assertion fail, print 'wrong range'
                except:
                    print(f"some inputs were not in the valid range! {self.nr_given_blocks=}, {self.gridsize=}")
                    continue

                # if everything is ok, set user_action to tuple (a,x,y)
                # set valid flag to true
                user_action = [a,x,y]
                is_valid_input = True

        # output: EOG means valid input without user_action and is_recording set to false (-> stop recording)
        # otherwise we return a vild user action with is_recording set to true (-> keep recording)
        if user_action is None:
            return None, False
        else:
            return user_action, True

class TrajectoryDatasetCreator():
    """
    Creates and saves a dataset of trajectories created by the user.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.trajectories = []
        self.currentTrajectory = []

    def addStartState(self, state):
        self.currentTrajectory += [state]

    def addStep(self, state, action, reward):
        # add given state, reward and action to the current trajectory

        # TRAJECTORY: [s_0, a_0, r_0, s_1, a_1, r_1, .. , s_T, a_T, r_T]

        #   s_t+1 : 1d array of length 175
        #       - 0-99 : row-wise flattened grid
        #       - 100-174 : 3x flattened grid for given blocks 
        #   IMPORTANT!! : state is s_t+1 and not s_t!!!!
        #   r_t : reward as float
        #   a_t : [a, x, y]
        #       - a : block index
        #       - x : center column
        #       - y : center row
        self.currentTrajectory += [action, reward, state]

    def finishTrajectory(self):
        # add trajectory to list of trajectories
        # leave out last state! otherwise we would have [.. , s_T, a_T, r_T, s_T+1]
        self.trajectories += [self.currentTrajectory[:-1]]
        # reset current Trajectory
        self.currentTrajectory = []

    def saveTrajectories(self):
        # save list of trajectories to a file
        # should be loadable by a torch dataset for the transfomer!
        torch.save(self.trajectories, self.filepath)
        return


def main():
    # open connection to unity ml agent
    print("Waiting for Unity environment...")
    env = UnityEnvironment()
    env = UnityToGymWrapper(env, allow_multiple_obs=True)
    print("Unity environment started successfully!")

    # create trajectory gym with this env
    # also give gridsize and nur of blocks
    traj_gym = TrajectoryGym(10, 3, env, "Python\\dataset\\first_attempt.pt")
    # start recording loop
    traj_gym.recording_loop()
    # after recording stopped for whatever reason, close environment
    env.close()


if __name__ == '__main__':
    main()
