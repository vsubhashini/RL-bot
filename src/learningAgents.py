from game import Directions, Agent, Actions

import random,util,time

class ValueEstimationAgent(Agent):
  """
    Abstract agent which assigns values to (state,action)
    Q-Values for an environment. As well as a value to a 
    state and a policy given respectively by,
    
    V(s) = max_{a in actions} Q(s,a)
    policy(s) = arg_max_{a in actions} Q(s,a)
    
    Both ValueIterationAgent and QLearningAgent inherit 
    from this agent. While a ValueIterationAgent has
    a model of the environment via a MarkovDecisionProcess
    (see mdp.py) that is used to estimate Q-Values before
    ever actually acting, the QLearningAgent estimates 
    Q-Values while acting in the environment. 
  """
  
  def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
    """
    Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    self.alpha = float(alpha)
    self.epsilon = float(epsilon)
    self.gamma = float(gamma)
    self.numTraining = int(numTraining)
    
  ####################################
  #    Override These Functions      #  
  ####################################
  def getQValue(self, state, action):
    """
    Should return Q(state,action)
    """
    util.raiseNotDefined()
    
  def getValue(self, state):
    """
    What is the value of this state under the best action? 
    Concretely, this is given by
    
    V(s) = max_{a in actions} Q(s,a)
    """
    util.raiseNotDefined()  
    
  def getPolicy(self, state):
    """
    What is the best action to take in the state. Note that because
    we might want to explore, this might not coincide with getAction
    Concretely, this is given by
    
    policy(s) = arg_max_{a in actions} Q(s,a)
    
    If many actions achieve the maximal Q-value,
    it doesn't matter which is selected.
    """
    util.raiseNotDefined()  
    
  def getAction(self, state):
    """
    state: can call state.getLegalActions()
    Choose an action and return it.   
    """
    util.raiseNotDefined()    
   
class ReinforcementAgent(ValueEstimationAgent):
  """
    Abstract Reinforcemnt Agent: A ValueEstimationAgent
	  which estimates Q-Values (as well as policies) from experience
	  rather than a model
      
      What you need to know:
		  - The environment will call 
		    observeTransition(state,action,nextState,deltaReward),
		    which will call update(state, action, nextState, deltaReward)
		    which you should override. 
      - Use self.getLegalActions(state) to know which actions
		    are available in a state
  """
  ####################################
  #    Override These Functions      #  
  ####################################
  
  def update(self, state, action, nextState, reward, environment):
    """
	    This class will call this function, which you write, after
	    observing a transition and reward
    """
    util.raiseNotDefined()
        
  ####################################
  #    Read These Functions          #  
  ####################################  
  
  def getLegalActions(self,state):
    """
      Get the actions available for a given
      state. This is what you should use to
      obtain legal actions for a state
    """
    return self.actionFn(state)
  
  def observeTransition(self, state,action,nextState,deltaReward, environment):
    """
    	Called by environment to inform agent that a transition has
    	been observed. This will result in a call to self.update
    	on the same arguments
    	
    	NOTE: Do *not* override or call this function
    """  	
    if isinstance(deltaReward, tuple):
      self.episodeRewards += (deltaReward[0] + deltaReward[1])
    else:
      self.episodeRewards += deltaReward
    self.update(state,action,nextState,deltaReward, environment)
		    
  def startEpisode(self):
    """
      Called by environment when new episode is starting
    """
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0
    
  def stopEpisode(self):
    """
      Called by environment when episode is done
    """ 
    if self.episodesSoFar < self.numTraining:
		  self.accumTrainRewards += self.episodeRewards
    else:
		  self.accumTestRewards += self.episodeRewards
    self.episodesSoFar += 1    
    if self.episodesSoFar >= self.numTraining:
      # Take off the training wheels
      self.epsilon = 0.0    # no exploration
      self.alpha = 0.0      # no learning

  def isInTraining(self): 
      return self.episodesSoFar < self.numTraining
  
  def isInTesting(self):
      return not self.isInTraining()
      
  def __init__(self, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1, qtype='walk'):
    """
    actionFn: Function which takes a state and returns the list of legal actions
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    if actionFn == None:
        actionFn = lambda state: state.getLegalActions()
    self.actionFn = actionFn
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0 
    self.accumTestRewards = 0.0
    self.numTraining = int(numTraining)
    self.epsilon = float(epsilon)
    self.alpha = float(alpha)
    self.gamma = float(gamma)
    self.qtype = str(qtype)
    
  ################################
  # Controls needed for Crawler  #
  ################################
  def setEpsilon(self, epsilon):
    self.epsilon = epsilon
    
  def setLearningRate(self, alpha):
    self.alpha = alpha
    
  def setDiscount(self, discount):
    self.gamma = discount
    
  def doAction(self,state,action):
    """
        Called by inherited class when
        an action is taken in a state
    """
    self.lastState = state
    self.lastAction = action
  
  ###################
  # Pacman Specific #
  ###################
