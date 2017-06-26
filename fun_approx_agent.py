import cv2
import numpy as np
import pickle
import itertools
import random
from random import randint
from enduro.agent import Agent
from enduro.action import Action


class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        # Learning rate
        self.alpha = 0.01
        # Discounting factor
        self.gamma = 0.9
        # Exploration rate
        self.epsilon = 0.01
        self.horizon = 10
        self.num_features = 23
        self.theta = np.ones(self.num_features) # weights initialised to 1
	self.Qsa = [0,0,0,0]	# Qsa dictionary, will hold 4 actions
        self.prev_reward = 0 # previous reward
        self.road = np.zeros((11,10)) # various initialisations...
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
	self.episode_log_w = np.zeros((501,self.num_features)) - 1. #500 or num_episodes
        self.total_reward = 0
	self.prev_action = 0

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

    def get_phi(self, road, cars, speed, grid):
        '''
        Input:
            road: an instance the road of the game
            cars: an instance the cars of the game
            speed: an instance the speed of the game
            grid: an instance the grid of the game

        Return
            state: the string of the state
	    features: the feature vector
        '''
        features = np.zeros(self.num_features)
	"""
	The features need to cover three requirements: 
		- Collisions avoided
		- Moving faster preferred
		- Staying in the centre preferred
	"""
	center= False
	leftw= False
	rightw= False
	lefto= False
	righto= False
	leftimm= False
	rightimm= False
	fronto= False
	accel= False
	fast= False
	lost= False
	coll= False

	ro=0 #right op
	lo=0
	dwr=0 #dist rigth
	dwl=0
	cr=0 #counts
	cl=0

	me = -1
	for j in range(0,10):
            if grid[0,j] == 2:
                me = j
		if (me <= 1):
			leftw = True
		elif (me >= 8):
			rightw = True
		elif (me > 2 and me < 7):
			center = True	
	dwl=me-1
	dwr=9-me
 	#speed
        if (speed > -50 and self.speed  <= -0):
            accel = True
        elif (speed  > 0):
            fast = True	

	if speed == -50:	# collided before
		coll = True 

	#opponent in front?
	for k in range(10):
		if grid[k,me] == 1 or (me!=0 and grid[k,me-1] == 1) or (me!=9 and grid[k,me+1] == 1):
			fronto = True
		for m in range(9):
			if me!=0 and m<me-1:
				if grid[k,m] == 1:
					cl+=1
					lefto =True
					if k==4 and m == me-1:
						leftimm=True
			if me!=9 and m>me:
				if grid[k,m] == 1:
					cr+=1
					righto =True
					if k==4 and m== me+1:
						rightimm=True

	features = np.zeros((4,self.num_features))
	if center and not fronto: # is in center noone in front
		features[0][0] = 1
		features[1][0] = 0
		features[2][0] = 0
		features[3][0] = 0
	if not center and (dwl>dwr) and not fast: # is not center, closer to the right
		features[0][1] = 0
		features[1][1] = 1
		features[2][1] = 0
		features[3][1] = 0
	if not center and (dwr>=dwl) and not fast: # is not center, closer to the left
		features[0][2] = 0
		features[1][2] = 0
		features[2][2] = 1
		features[3][2] = 0
	if not center and (dwl>dwr) and fast: # is not center, closer to the right
		features[0][17] = 1
		features[1][17] = 0
		features[2][17] = 0
		features[3][17] = 0
	if not center and (dwr>=dwl) and fast: # is not center, closer to the left
		features[0][22] = 1
		features[1][22] = 0
		features[2][22] = 0
		features[3][22] = 0
	if fronto and (leftimm or(cr<cl)): # has front car, more cars left
		features[0][3] = 0
		features[1][3] = 0
		features[2][3] = 1
		features[3][3] = 0
	if fronto and (rightimm or (cl<=cr)): # has front car, more cars right
		features[0][4] = 0
		features[1][4] = 0
		features[2][4] = 1
		features[3][4] = 0
	if not fronto: # no front car 
		features[0][5] = 1
		features[1][5] = 0
		features[2][5] = 0
		features[3][5] = 0
	if coll and leftimm: # has collided, more cars left
		features[0][6] = 0
		features[1][6] = 1
		features[2][6] = 0
		features[3][6] = 0
	if coll and rightimm: # has collided, more cars right
		features[0][7] = 0
		features[1][7] = 0
		features[2][7] = 1
		features[3][7] = 0
	if coll and (dwr>=dwl): # has collided
		features[0][20] = 0
		features[1][20] = 0
		features[2][20] = 1
		features[3][20] = 0
	if coll and (dwl>dwl): # has collided
		features[0][19] = 0
		features[1][19] = 1
		features[2][19] = 0
		features[3][19] = 0
	if coll: # has collided
		features[0][21] = 1
		features[1][21] = 0
		features[2][21] = 0
		features[3][21] = 0
	if rightw and not rightimm: # has wall right, no car left
		features[0][8] = 0
		features[1][8] = 1
		features[2][8] = 0
		features[3][8] = 0
	if leftw and not leftimm: # has wall left, no car right
		features[0][9] = 0
		features[1][9] = 0
		features[2][9] = 1
		features[3][9] = 0
	if rightw and rightimm: # has wall right, car left
		features[0][10] = 1
		features[1][10] = 0
		features[2][10] = 0
		features[3][10] = 0
	if leftw and leftimm: # has wall left, car left
		features[0][11] = 1
		features[1][11] = 0
		features[2][11] = 0
		features[3][11] = 0
	if accel and not fronto: # accelerating, no car front
		features[0][12] = 1
		features[1][12] = 0
		features[2][12] = 0
		features[3][12] = 0
	if accel and fronto and (leftimm or (cr<cl)): # accelerating, car front, more cars left
		features[0][13] = 0
		features[1][13] = 1
		features[2][13] = 0
		features[3][13] = 0
	if accel and fronto and (rightimm or (cl<=cr)): # accelerating, car front, more cars right
		features[0][14] = 0
		features[1][14] = 0
		features[2][14] = 1
		features[3][14] = 0
	if accel and not fronto and not center and (dwr>dwl): # max speed, no front car, left turn coming
		features[0][15] = 0
		features[1][15] = 0
		features[2][15] = 1
		features[3][15] = 0
	if accel and not fronto and not center and (dwl>=dwr): # max speed, no front car, right turn coming
		features[0][16] = 0
		features[1][16] = 1
		features[2][16] = 0
		features[3][16] = 0
	if fast and not fronto and not center and (dwr>dwl): # max speed, no front car, left turn coming
		features[0][15] = 0
		features[1][15] = 0
		features[2][15] = 1
		features[3][15] = 0
	if fast and not fronto and not center and (dwl>=dwr): # max speed, no front car, right turn coming
		features[0][16] = 0
		features[1][16] = 1
		features[2][16] = 0
		features[3][16] = 0
	if fast and center: # max speed
		features[0][18] = 1
		features[1][18] = 0
		features[2][18] = 0
		features[3][18] = 0
	return features

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
	"""
	# You can get the set of possible actions and print them with:
        # print [Action.toString(a) for a in self.getActionsSet()]

	greedy_action = np.argmax(self.Qsa) #

        # choose the action using epsilon-greedy
	random_number = random.random()
        if (random_number <= (1 - self.epsilon)):
            chosen_action = greedy_action
        else:
            random_action = randint(0,3)
            chosen_action = random_action

        if chosen_action==0:
            action = Action.ACCELERATE
        elif chosen_action==1:
            action = Action.LEFT
        elif chosen_action==2:
            action = Action.RIGHT
        elif chosen_action==3:
            action = Action.BRAKE
	#print(action)

        # Execute the action and get the received reward signal
        prev_reward = self.move(action)
        self.total_reward += prev_reward
	self.prev_reward = prev_reward
        self.prev_action = chosen_action

        return self.prev_reward

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
        prev_features = self.get_phi(self.prev_road, self.prev_cars, self.prev_speed, self.prev_grid)
        features = self.get_phi(self.road, self.cars, self.speed, self.grid)
	#print(features)

        #prevQsa = prev_features[self.prev_action]*self.theta 

	q = np.max(self.Qsa)
	#prev_q = prevQsa[self.prev_action]
	prev_q = prev_features[self.prev_action]*self.theta

        self.theta = self.theta + self.alpha * (self.prev_reward + self.gamma*q - prev_q ) * prev_features[self.prev_action]

	self.theta = self.theta/np.linalg.norm(self.theta) # normalize
        self.Qsa = np.dot(features,self.theta) # update with new thetas

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
	if  iteration > 6496:
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
	    #print("Final episode actions...")
            #print(self.Qsa)
	    #print("Final episode theta...")
            #print(self.theta)

	# Initialise the log for the next episode
        if episode != self.last_episode:
            iters = np.nonzero(self.episode_log >= 0)
            rewards = self.episode_log[iters]
            self.log.append((np.asarray(iters).flatten(), rewards, np.copy(self.Qsa)))
            self.episode_log = np.zeros(6510) - 1.
            self.last_episode = episode

        # Log the reward at the current iteration
        self.episode_log[iteration] = self.total_reward
        self.episode_log_w[episode] = self.theta

        # You could comment this out in order to speed up iterations
        #if not episode % 500:
        #cv2.imshow("Enduro", self._image)
        #cv2.waitKey(40)


if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=500, draw=True)
    #print 'Total reward: ' + str(a.total_reward)
    pickle.dump(a.log, open("log.p", "wb"))
    pickle.dump(a.episode_log_w, open("theta.p", "wb"))
