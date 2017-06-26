import cv2
import numpy as np
import itertools
import random
import pickle
from random import randint
from enduro.agent import Agent
from enduro.action import Action


class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.epsilon = 0.02
        self.gamma = 0.9
        self.learning_rate = 0.02
        self.horizon = 10

        self.num_features = 12
        self.features = np.zeros(self.num_features)
        self.weights = np.ones(self.num_features)

        self.Q_sa = {}
        permutations = ["".join(seq) for seq in itertools.product("01", repeat=self.num_features)]
        for state in permutations:
            # ACCELERATE LEFT RIGHT BRAKE
            self.Q_sa[state] = [2,0.5,0.5,0]
        #print(self.Q_sa)

        self.last_action = 0
        self.last_reward = 0

        self.road = np.zeros((11,10))
        self.cars = {}
        self.speed = -50
        self.grid = np.zeros((11,10))
        self.prev_road = np.zeros((11,10))
        self.prev_cars = {}
        self.prev_speed = -50
        self.prev_grid = np.zeros((11,10))

        # Log the obtained reward during learning
        self.last_episode = 1
        self.episode_log = np.zeros(6510) - 1.
        self.log = []


    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """

        # Reset the total reward for the episode
        self.total_reward = 0


    def find_state(self, road, cars, speed, grid):
        '''
        Input:
            road: an instance the road of the game
            cars: an instance the cars of the game
            speed: an instance the speed of the game
            grid: an instance the grid of the game

        Return
            state: the string of the state
        '''

        # feature checking

        # locate me and find if i am near the walls where I lose speed
        self.features[:] = 0
        for j in range(0,10):
            if grid[0,j] == 2:
                my_position = j

                if (my_position <= 1):
                    #left wall
                    self.features[0] = 1
                elif (my_position >= 8):
                    #right wall
                    self.features[2] = 1
                elif (my_position > 1 and my_position < 8):
                    # center not wall
                    self.features[1] = 1


        #check in which line there is the first opponent
        existing_opponent_line = -1
        existing_opponent_pos = -1
        for i in range(0, self.horizon):
            for j in range(0,10):
                if (grid[i,j] == 1):
                    existing_opponent_line = i
                    existing_opponent_pos = j
                    if (existing_opponent_line > 0 and existing_opponent_line <= 2):
                        self.features[6] = 1
                    elif (existing_opponent_line > 2 and existing_opponent_line <= 5):
                        self.features[7] = 1
                    elif (existing_opponent_line > 5):
                        self.features[8] = 1
                    # elif (existing_opponent_line > 6 and existing_opponent_line <= 8):
                        # self.features[9] = 1

        #check where is the opponent relatively to me (left, ahead, right)
        opponent_location = -1
        opponent_exists = False

        if (existing_opponent_line != -1 & existing_opponent_pos != -1):
            opponent_exists = True

        if opponent_exists:
            if (my_position - 1 > existing_opponent_pos):
                #left
                self.features[3] = 1
            elif (my_position + 1 < existing_opponent_pos):
                #right
                self.features[5] = 1
            elif (np.abs(existing_opponent_pos-my_position) == 1):
                #ahead
                self.features[4] = 1

        # check speed class
        if (speed > -50 and self.speed  <= -0):
            self.features[9] = 1
        elif (speed  > 0):
            self.features[10] = 1

        # if it has collided
        if (speed < self.prev_speed):
            self.features[11] = 1


        #print self.features
        #print self.weights

        # make string
        state = ""
        for feature in self.features:
            # print(feature)
            state += str(int(feature))

        return state, self.features


    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # ACCELERATE LEFT RIGHT BRAKE
        # key = cv2.waitKey(1)

        # find state
        current_state, _ = self.find_state(self.road, self.cars, self.speed, self.grid)

        # choose the action using epsilon-greedy
        #find greedy option
        greedy_action = np.argmax(self.Q_sa[current_state])
        random_number = random.random()
        # random_action = randint(0,3)

        if (random_number <= (1 - self.epsilon)):
            chosen_action = greedy_action
        else:
            random_action = randint(0,3)
            # print(" "  + str(random_action))
            chosen_action = random_action

        self.last_action = chosen_action
        if chosen_action==0:
            action = Action.ACCELERATE
        elif chosen_action==1:
            action = Action.LEFT
        elif chosen_action==2:
            action = Action.RIGHT
        elif chosen_action==3:
            action = Action.BRAKE
        else:
            print('Something went wrong...')
        # print(self.Q_sa[current_state])
        # print("greedy_action ")
        # print(greedy_action)
        # print("random_action ")
        # print(random_action)
        # print("chosen_action ")
        # print(chosen_action)
        # print("random_number ")
        # print(random_number)

        # Execute the action and get the received reward signal
        self.last_reward = self.move(action)
        self.total_reward += self.last_reward

        return self.last_reward


        #self.last_action = last_action
        #self.last_reward = last_reward

        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BRAKE
        # Do not use plain integers between 0 - 3 as it will not work
        

    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """

        prev_grid =  self.grid
        self.prev_grid = prev_grid
        self.grid = grid

        prev_road =  self.road
        self.prev_road = prev_road
        self.road = road

        prev_cars =  self.cars
        self.prev_cars = prev_cars
        self.cars = cars

        prev_speed =  self.speed
        self.prev_speed = prev_speed
        self.speed = speed

    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """

        # identify states for previous sense info and current sense info
        prev_state, prev_features = self.find_state(self.prev_road, self.prev_cars, self.prev_speed, self.prev_grid)
        state, features = self.find_state(self.road, self.cars, self.speed, self.grid)

        # get best action for this state
        best_action = np.argmax(self.Q_sa[prev_state])
        q_sa_prev =  self.Q_sa[prev_state][self.last_action]
        q_sa =  self.Q_sa[prev_state][best_action]

        # Weights update
        # ACCELERATE, LEFT, RIGHT, BREAK [columns 0,1,2,3]
        self.weights = self.weights + self.learning_rate * (self.last_reward + self.gamma*q_sa - q_sa_prev) * self.features
        # print(self.weights)

        #print(self.Q_sa[prev_state])
        q_sa_prev = np.sum(self.weights * prev_features)
        self.Q_sa[prev_state][self.last_action] = q_sa_prev
        #print(self.Q_sa[prev_state])


    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        if  iteration > 6496:
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
            print(self.weights)

            #print (self.Q_sa)

        # Initialise the log for the next episode
        if episode != self.last_episode:
            iters = np.nonzero(self.episode_log >= 0)
            rewards = self.episode_log[iters]
            self.log.append((np.asarray(iters).flatten(), rewards, np.copy(self.Q_sa)))
            self.last_episode = episode
            self.episode_log = np.zeros(6510) - 1.

        # Log the reward at the current iteration
        self.episode_log[iteration] = self.total_reward

        # You could comment this out in order to speed up iterations
        if not episode % 200:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)


if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=200, draw=True)
    # print 'Total reward: ' + str(a.total_reward)
    pickle.dump(a.log, open("log.p", "wb"))
