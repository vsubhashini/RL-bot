import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    qValues={} #Dictionary of dictionaries {state1: {action1: quality, action2: quality2, ...}, ... }
    values_old={} #Dictionary of dictionaries {state1: {action1: quality, action2: quality2, ...}, ... }
    values_prev=self.values.copy()
    i=0
    while i<self.iterations:
      i+=1
      for state in mdp.getStates():
        actions = mdp.getPossibleActions(state)
        for action in actions:
          quality=0
          for nextStateTransitions in mdp.getTransitionStatesAndProbs(state, action):
            nextState = nextStateTransitions[0]
            nextProb = nextStateTransitions[1]
            quality += nextProb * (mdp.getReward(state, action, nextState) + (self.discount * values_prev[nextState]))
          qValues[(state,action)]=quality
	  #print "state: "+str(state)+" Action: "+action+" Qvalue: "+str(qValues[(state,action)])
	if(state=='TERMINAL_STATE'):
	  self.values[state]=0
	else:
          self.values[state] = max(qValues[(state,action)] for action in actions)
      values_prev=self.values.copy()

    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    """Value(state) is computed during init"""
    #givenState=state
    #givenAction=action
    #qValues={} #Dictionary of dictionaries {state1: {action1: quality, action2: quality2, ...}, ... }
    #values_old={} #Dictionary of dictionaries {state1: {action1: quality, action2: quality2, ...}, ... }
    #values_prev=self.values.copy()
    #i=0
    #while i<5:
    #  i+=1
    #  for state in self.mdp.getStates():
    #    actions = self.mdp.getPossibleActions(state)
    #    for action in actions:
    #      quality=0
    #      for nextStateTransitions in self.mdp.getTransitionStatesAndProbs(state, action):
    #        nextState = nextStateTransitions[0]
    #        nextProb = nextStateTransitions[1]
    #        quality += nextProb * (self.mdp.getReward(state, action, nextState) + (self.discount * values_prev[nextState]))
    #      qValues[(state,action)]=quality
    #      #print "state: "+str(state)+" Action: "+action+" Qvalue: "+str(qValues[(state,action)])
    #    if(state=='TERMINAL_STATE'):
    #      self.values[state]=0
    #    else:
    #      self.values[state] = max(qValues[(state,action)] for action in actions)
    #  values_prev=self.values.copy()
    #return qValues[(givenState, givenAction)]
    qValue=0
    for nextStateTransitions in self.mdp.getTransitionStatesAndProbs(state, action):
      nextState = nextStateTransitions[0]
      nextProb = nextStateTransitions[1]
      qValue += nextProb * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState]))
    return qValue
    #util.raiseNotDefined()

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    maxqval=float("-inf")
    bestAction=None
    for action in self.mdp.getPossibleActions(state):
      qValue = self.getQValue(state, action)
      if qValue > maxqval:
        maxqval = qValue
	bestAction=action
    return bestAction
    #util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
