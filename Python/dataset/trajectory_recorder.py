import torch
import sys

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from typing import List, Tuple

class TrajectoryRecorder():
    """
    DEPRECATED DEPRECATED DEPRECATED
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

    def recording_loop(self, record_mistakes : bool = True):
        """
        this implements a session and a game loop
        - a game goes from starting state until Game Over or End of Game (EOG) is called
        - a session consists of several games

        if record_mistakes is set the mistakes are added to trajectories recorded
        
        IMPORTANT: client is responsible for calling finish_recording to save the data in the
        dataset creator!
        """
        # start the session loop
        while True:
            # call reset on environment to get the first state s_0 for the beginning
            # of the dataset trajectory!
            state = self.env.reset()[0] # state is list with array
            self.ds_creator.addStartState(state)

            # start game loop
            while True:
                # wait for user input
                user_action, recv_EOG, recv_EOS = self.read_input()
                #user_action, recording_stop = self.env.action_space.sample(), False

                # if the user stops the recording with the EOG or EOS
                # finish the current trajectory, deal with it in the session loop
                if recv_EOG or recv_EOS: break

                # otherwise the user_action is valid
                # send action to environment, get observation
                state, reward, is_done, info = self.env.step(user_action)
                state = state[0]
                
                # if the input was a mistake (negative reward and is_done == False)
                # check if we record those
                if reward < 0 and not is_done:
                    if record_mistakes:
                        print("Mistake recorded!")
                        self.ds_creator.addStep(state, user_action, reward)
                    else:
                        print("Mistake!")
                # otherwise just add
                else:
                    self.ds_creator.addStep(state, user_action, reward)

                # check if a terminal state has been hit -> game over
                # step out of the game loop
                if is_done:
                    print("Game Over")
                    break

            # EOG has been given or game over has been reached
            # finish current trajectory
            self.ds_creator.finishTrajectory()
            
            # if EOS has been given, break out of the session loop
            if recv_EOS: break
            # otherwise reset the game and setup new trajectory
            state = self.env.reset()[0] # state is list with array
            self.ds_creator.addStartState(state)

    def finish_recording(self):
        self.ds_creator.saveTrajectories()


    def read_input(self) -> Tuple[List[int], bool, bool]:

        # create input loop
        start_message = "input (format a x y) : "
        user_action = None
        is_valid_input = False
        # flags for EOG and EOS
        recv_EOG, recv_EOS = False, False

        while not is_valid_input:

            # get input from console
            user_input = input(start_message)
            # INPUT SANITATION
            # delete leading and trailing whitespaces
            user_input = user_input.strip()

            # EOG and EOS
            if user_input in ['EOG', 'EOS']:
                # stop current loop as we received a valid input
                # user_action is None and only the flags matter
                user_action = None
                is_valid_input = True
                # set EOS/EOG flags
                if user_input == 'EOG':
                    recv_EOG = True
                else:
                    recv_EOS = True
            # try to parse to valid input
            else:
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

        # simply return user_action, recv_EOG, recv_EOS
        return user_action, recv_EOG, recv_EOS

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


def main(args : List[str]):
    # open connection to unity ml agent
    print("Waiting for Unity environment...")
    env = UnityEnvironment()
    env = UnityToGymWrapper(env, allow_multiple_obs=True)
    print("Unity environment started successfully!")

    filepath = args[1]
    record_mistakes = False
    if len(args) == 3 and args[2] in ["True", "true", "t"]:
        record_mistakes = True

    # create trajectory gym with this env
    # also give gridsize and nur of blocks
    traj_gym = TrajectoryRecorder(10, 3, env, filepath=filepath)
    # start recording loop
    traj_gym.recording_loop(record_mistakes)
    # after recording stopped for whatever reason, close environment
    env.close()
    # finally call finish_recording to save the dataset
    traj_gym.finish_recording()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(sys.argv)
        raise RuntimeError("Missing arguments! At least dataset path has to be given.")
    main(sys.argv)
