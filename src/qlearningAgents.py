from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
          
class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.QValues = util.Counter()
    #self.qtype=qtype
    self.environment=None
  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    "*** YOUR CODE HERE ***"
    if self.environment!=None:
      env = self.environment.getCurrentStateEnv(state, self.qtype)
      return self.QValues[(env,action)]
    #else:
    #  print "Environment not set getQValue"
    return self.QValues[(state,action)]
    #util.raiseNotDefined()
  
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    actions = self.getLegalActions(state)
    if len(actions) < 1:
      return 0.0
    if self.environment!=None:
      env = self.environment.getCurrentStateEnv(state, self.qtype)
      return max([self.QValues[(env, action)] for action in actions])
    #else:
    #  print "Environment not set getValue"
    return max([self.QValues[(state, action)] for action in actions])
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    maxqval=float("-inf")
    bestAction=[]
    actions = self.getLegalActions(state)
    if len(actions)==0:
      return None
    for action in actions:
      env=state
      if self.environment!=None:
        env = self.environment.getCurrentStateEnv(state, self.qtype)
      #else:
      #  print "Environment not set getPolicy"
      qValue = self.QValues[(env,action)]
      if qValue > maxqval:
        maxqval=qValue
	bestAction=[action]
      elif qValue==maxqval:
        bestAction.append(action)
    return random.choice(bestAction)
    
  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    if len(legalActions)==0:
      return action
    if util.flipCoin(self.epsilon):
      return random.choice(legalActions)
    else:
      return self.getPolicy(state)
  
  def update(self, state, action, nextState, reward, environment):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    self.environment=environment
    if state=="TERMINAL_STATE":
      return
    env = self.environment.getCurrentStateEnv(state, self.qtype)
    nextenv = self.environment.getCurrentStateEnv(nextState, self.qtype)
    oldQsa = self.QValues[(env, action)]
    nextActions = self.getLegalActions(nextState)
    if len(nextActions)==0:
      maxFutureValue=self.QValues[(nextenv, action)]
    else:
      maxFutureValue=max([self.QValues[(nextenv, nextAction)] for nextAction in nextActions])
    self.QValues[(env, action)] = oldQsa + self.alpha * ( reward + self.gamma * (maxFutureValue) - oldQsa )
    #print "Updated: state: "+str(env)+" action: "+action+" nextState: "+str(nextenv)+" Qvalue: "+str(self.QValues[(env, action)])+"\n"
    

class modularQAgent(ReinforcementAgent):
  """
    A modular Q-Learning Agent
    
    A q learning Agent that learns
    from multiple q learning modules

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    """ Has 2 modules - walk, obst"""
    self.QwalkValues = util.Counter()
    self.QobstValues = util.Counter()
    self.QValues = util.Counter()  #combined
    self.weight=0.75
    self.environment=None
    self.qtype="obstacle" #to get environment
  
  def getQValue(self, state, action):
    """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
    if self.environment!=None:
      env = self.environment.getCurrentStateEnv(state, self.qtype)
      return self.QValues[(env,action)]
    #else:
    #  print "Environment not set getQValue"
    return self.QValues[(state,action)]
  
    
  def getValue(self, state):
    """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    actions = self.getLegalActions(state)
    if len(actions) < 1:
      return 0.0
    if self.environment!=None:
      env = self.environment.getCurrentStateEnv(state, self.qtype)
      return max([self.QValues[(env, action)] for action in actions])
    #else:
    #  print "Environment not set getValue"
    return max([self.QValues[(state, action)] for action in actions])
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    maxqval=float("-inf")
    bestAction=[]
    actions = self.getLegalActions(state)
    if len(actions)==0:
      return None
    qwtype='walk'
    qotype='obstacle' 
    wt=self.weight
    for action in actions:
      envwalk=state
      envobstacle=state
      if self.environment!=None:
        envwalk = self.environment.getCurrentStateEnv(state, qwtype)
        envobstacle = self.environment.getCurrentStateEnv(state, qotype)
      #else:
      #  print "Environment not set getPolicy"
      qwalkValue = self.QwalkValues[(envwalk,action)]
      qobstacleValue = self.QobstValues[(envobstacle,action)]
      qValue = (wt*qwalkValue + (1-wt)*qobstacleValue)
      if qValue > maxqval:
        maxqval=qValue
	bestAction=[action]
      elif qValue==maxqval:
        bestAction.append(action)
    return random.choice(bestAction)
    
  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """  
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    if len(legalActions)==0:
      return action
    if util.flipCoin(self.epsilon):
      return random.choice(legalActions)
    else:
      return self.getPolicy(state)
  
  def update(self, state, action, nextState, reward, environment):
    """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
    self.environment=environment
    if state=="TERMINAL_STATE":
      return
    qwtype='walk'
    qotype='obstacle' 
    rewardwalk, rewardobst = reward
    envwalk = self.environment.getCurrentStateEnv(state, qwtype)
    envobstacle = self.environment.getCurrentStateEnv(state, qotype)
    nextenvwalk = self.environment.getCurrentStateEnv(nextState, qwtype)
    nextenvobstacle = self.environment.getCurrentStateEnv(nextState, qotype)
    oldwalkQsa = self.QwalkValues[(envwalk, action)]
    oldobstQsa = self.QobstValues[(envobstacle, action)]
    nextActions = self.getLegalActions(nextState)
    if len(nextActions)==0:
      maxFutureWalkValue=self.QwalkValues[(nextenvwalk, action)]
      maxFutureObstValue=self.QobstValues[(nextenvobstacle, action)]
    else:
      maxFutureWalkValue=max([self.QValues[(nextenvwalk, nextAction)] for nextAction in nextActions])
      maxFutureObstValue=max([self.QValues[(nextenvobstacle, nextAction)] for nextAction in nextActions])
    self.QwalkValues[(envwalk, action)] = oldwalkQsa + self.alpha * ( rewardwalk + self.gamma * (maxFutureWalkValue) - oldwalkQsa )
    self.QobstValues[(envobstacle, action)] = oldobstQsa + self.alpha * ( rewardobst + self.gamma * (maxFutureObstValue) - oldobstQsa )
    qwalkValue = self.QwalkValues[(envwalk,action)]
    qobstacleValue = self.QobstValues[(envobstacle,action)]
    qValue = (self.weight*qwalkValue + (1-self.weight)*qobstacleValue)
    self.QValues[(envobstacle, action)] = qValue
    #print "Updated: state: "+str(env)+" action: "+action+" nextState: "+str(nextenv)+" Qvalue: "+str(self.QValues[(env, action)])+"\n"
    

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"
  
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action

    
class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition  
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)
    
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass
