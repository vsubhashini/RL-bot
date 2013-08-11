import random
import sys
import mdp
import environment
import util
import optparse

class Gridworld(mdp.MarkovDecisionProcess):
  """
    Gridworld
  """
  def __init__(self, grid):
    # layout
    if type(grid) == type([]): grid = makeGrid(grid)
    self.grid = grid
    
    # parameters
    self.livingReward = 0.0
    self.noise = 0.2
    #self.noise = 0.0
        
  def setLivingReward(self, reward):
    """
    The (negative) reward for exiting "normal" states.
    
    Note that in the R+N text, this reward is on entering
    a state and therefore is not clearly part of the state's
    future rewards.
    """
    self.livingReward = reward
        
  def setNoise(self, noise):
    """
    The probability of moving in an unintended direction.
    """
    self.noise = noise
        
                                    
  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.
    
    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    if state == self.grid.terminalState:
      return ()
    x,y = state
    #if type(self.grid[x][y]) == int:
    if x==(self.grid.width-1):
      return ('exit',)
    return ('north','west','south','east')
    
  def getStates(self):
    """
    Return list of all states.
    """
    # The true terminal state.
    states = [self.grid.terminalState]
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if self.grid[x][y] != '#':
          state = (x,y)
          states.append(state)
    return states
        
  def getReward(self, state, action, nextState, qtype):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    
    Walking agent learns from departing state
    Obstacle agent learns from arriving state (nextState)
    """
    #walking qlearning agent - default
    if qtype=="walk":
      if state == self.grid.terminalState:
        return 0.0
      x, y = state

    if qtype=="obstacle" or qtype=="combo":
      if nextState == self.grid.terminalState:
        x, y = state
      else:
        x, y = nextState
    cell = self.grid[x][y]
    if type(cell) == int or type(cell) == float:
      if qtype=='combo':
        return (cell, cell)
      return cell
    if qtype=="obstacle":
      return 0.25    #give living reward for avoiding/getting out of obstacle
    #print "Living reward: "+str(self.livingReward)
    if qtype=="walk":
      if action=='east':
        return 0.5
      else:
        return -0.5
    if qtype=="combo":
      if action=='east':
        #return 0.75
        return (0.5, 0.25)
      else:
        return (-0.5, 0.25)
    return self.livingReward
        
  def getEnv(self, state, qtype):
    """
    Get North, South, East, West cell content.
    Contents can be w (wall), o (obstacle), ' '
    
    Note that the Env is the actual state for obstacle module
    For walk module it just returns grid position
    """
    if qtype=="walk":
      return state
    if state == self.grid.terminalState:
      return None
    x, y = state
    cell = self.grid[x][y]
    north=' '
    south=' '
    east=' '
    west=' '

    if(y==self.grid.height-1):
      northcell=None
    else:
      northcell=self.grid[x][y+1]
    if(northcell==None):
      north='w'    #wall
    elif type(northcell) == int or type(northcell) == float:
      if float(northcell)<0.0:
        north='o'  #obstacle
    
    if(y==0):
      southcell=None
    else:
      southcell=self.grid[x][y-1]
    if(southcell==None):
      south='w'    #wall
    elif type(southcell) == int or type(southcell) == float:
      if float(southcell)<0.0:
        south='o'  #obstacle

    if(x==self.grid.width-1):
      eastcell=None
    else:
      eastcell=self.grid[x+1][y]
    if(eastcell==None):
      east='w'    #wall
    elif type(eastcell) == int or type(eastcell) == float:
      if float(eastcell)<0.0:
        east='o'  #obstacle

    if(x==0):
      westcell=None
    else:
      westcell=self.grid[x-1][y]
    if(westcell==None):
      west='w'    #wall
    elif type(westcell) == int or type(westcell) == float:
      if float(westcell)<0.0:
        west='o'  #obstacle

    env = north, south, east, west
    return env

  def getStartState(self):
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if self.grid[x][y] == 'S':
          return (x, y)
    raise 'Grid has no start state'
    
  def isTerminal(self, state):
    """
    Only the TERMINAL_STATE state is *actually* a terminal state.
    The other "exit" states are technically non-terminals with
    a single action "exit" which leads to the true terminal state.
    This convention is to make the grids line up with the examples
    in the R+N textbook.
    """
    return state == self.grid.terminalState
        
                   
  def getTransitionStatesAndProbs(self, state, action):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.          
    """        
        
    if action not in self.getPossibleActions(state):
      raise "Illegal action!"
      
    if self.isTerminal(state):
      return []
    
    x, y = state
    
    if x==(self.grid.width-1):
      termState = self.grid.terminalState
      return [(termState, 1.0)]
      
    successors = []                
                
    northState = (self.__isAllowed(y+1,x) and (x,y+1)) or state
    westState = (self.__isAllowed(y,x-1) and (x-1,y)) or state
    southState = (self.__isAllowed(y-1,x) and (x,y-1)) or state
    eastState = (self.__isAllowed(y,x+1) and (x+1,y)) or state
    #print "NEWS: "+str(northState)+str(eastState)+str(westState)+str(southState)
                        
    if action == 'north' or action == 'south':
      if action == 'north': 
        successors.append((northState,1-self.noise))
      else:
        successors.append((southState,1-self.noise))
                                
      massLeft = self.noise
      successors.append((westState,massLeft/2.0))    
      successors.append((eastState,massLeft/2.0))
                                
    if action == 'west' or action == 'east':
      if action == 'west':
        successors.append((westState,1-self.noise))
      else:
        successors.append((eastState,1-self.noise))
                
      massLeft = self.noise
      successors.append((northState,massLeft/2.0))
      successors.append((southState,massLeft/2.0)) 
      
    successors = self.__aggregate(successors)
                                                                           
    return successors                                
  
  def __aggregate(self, statesAndProbs):
    counter = util.Counter()
    for state, prob in statesAndProbs:
      counter[state] += prob
    newStatesAndProbs = []
    for state, prob in counter.items():
      newStatesAndProbs.append((state, prob))
    return newStatesAndProbs
        
  def __isAllowed(self, y, x):
    if y < 0 or y >= self.grid.height: return False
    if x < 0 or x >= self.grid.width: return False
    return self.grid[x][y] != '#'

class GridworldEnvironment(environment.Environment):
    
  def __init__(self, gridWorld):
    self.gridWorld = gridWorld
    self.reset()
            
  def getCurrentState(self):
    return self.state
        
  def getCurrentStateEnv(self, state, qtype):
    return self.gridWorld.getEnv(state, qtype)

  def getPossibleActions(self, state):        
    return self.gridWorld.getPossibleActions(state)
        
  def inferAction(self, state, nextState):
    if nextState=="TERMINAL_STATE": #state=="TERMINAL_STATE" or 
      return None
    x1, y1 = state
    x2, y2 = nextState
    action=None
    if(x1==x2 and y2==y1+1):
      action='north'
    elif(x1==x2 and y2==y1-1):
      action='south'
    elif(y1==y2 and x2==x1+1):
      action='east'
    elif(y1==y2 and x2==x1-1):
      action='west'
    return action

  def doAction(self, action, qtype):
    successors = self.gridWorld.getTransitionStatesAndProbs(self.state, action) 
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.' 
      if rand < sum:
	#Original code has error here. i.e the correct action corresponding to the
	#randomly chosen next state is not updated properly
        #Infer correct action
        tookAction = self.inferAction(state, nextState)
	if tookAction!=None:
          action=tookAction
        reward = self.gridWorld.getReward(state, action, nextState, qtype)
        self.state = nextState
        return (nextState, reward, action)
    raise 'Total transition probability less than one; sample failure.'    
        
  def reset(self):
    self.state = self.gridWorld.getStartState()

class Grid:
  """
  A 2-dimensional array of immutables backed by a list of lists.  Data is accessed
  via grid[x][y] where (x,y) are cartesian coordinates with x horizontal,
  y vertical and the origin (0,0) in the bottom left corner.  
  
  The __str__ method constructs an output that is oriented appropriately.
  """
  def __init__(self, width, height, initialValue=' '):
    self.width = width
    self.height = height
    self.data = [[initialValue for y in range(height)] for x in range(width)]
    self.terminalState = 'TERMINAL_STATE'
    
  def __getitem__(self, i):
    return self.data[i]
  
  def __setitem__(self, key, item):
    self.data[key] = item
    
  def __eq__(self, other):
    if other == None: return False
    return self.data == other.data
    
  def __hash__(self):
    return hash(self.data)
  
  def copy(self):
    g = Grid(self.width, self.height)
    g.data = [x[:] for x in self.data]
    return g
  
  def deepCopy(self):
    return self.copy()
  
  def shallowCopy(self):
    g = Grid(self.width, self.height)
    g.data = self.data
    return g
    
  def _getLegacyText(self):
    t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
    t.reverse()
    return t
    
  def __str__(self):
    return str(self._getLegacyText())

def makeGrid(gridString):
  walk=False
  if gridString[0][0]=='W':
    walk=True
  if walk:
    obstacleProb=0.0
  else:
    obstacleProb=0.2
  width, height = 10, 3
  grid = Grid(width, height)
  for h in range(height):
    if walk:
      grid[width-1][h]=10
    elif gridString[0][0]=='C':
      grid[width-1][h]=35
    else:
      grid[width-1][h]=2
  for x in range(0,width-1):
    for y in range(0,height):
      if util.flipCoin(obstacleProb):
        grid[x][y]=-2
      else:
        #grid[x][y]='-1'
        grid[x][y]=' '
  grid[0][0]='S'
  return grid    
             
def getBookGrid():
  grid = [[' ',' ',' ',+1],
          [' ','#',' ',-1],
          ['S',' ',' ',' ']]
  return Gridworld(grid)

def getObstacleGrid():
  grid = [['O',' ',' ',+1],
          [' ','#',' ',-1],
          ['S',' ',' ',' ']]
  return Gridworld(grid)

def getWalkGrid():
  grid = [['W',' ',' ',+1],
          [' ','#',' ',-1],
          ['S',' ',' ',' ']]
  return Gridworld(grid)

def getComboGrid():
  grid = [['C',' ',' ',+1],
          [' ','#',' ',-1],
          ['S',' ',' ',' ']]
  return Gridworld(grid)


def getUserAction(state, actionFunction):
  """
  Get an action from the user (rather than the agent).
  
  Used for debugging and lecture demos.
  """
  import graphicsUtils
  action = None
  while True:
    keys = graphicsUtils.wait_for_keys()
    if 'Up' in keys: action = 'north'
    if 'Down' in keys: action = 'south'
    if 'Left' in keys: action = 'west'
    if 'Right' in keys: action = 'east'
    if 'q' in keys: sys.exit(0)
    if action == None: continue
    break
  actions = actionFunction(state)
  if action not in actions:
    action = actions[0]
  return action

def printString(x): print x

def runEpisode(agent, qtype, environment, discount, decision, display, message, pause, episode):

  ###########################
  # GET THE GRIDWORLD
  ###########################

  returns = 0
  totalDiscount = 1.0
  environment.reset()
  
  #for state in mdp.getStates():
  #  display(state)

  if 'startEpisode' in dir(agent): agent.startEpisode()
  message("BEGINNING EPISODE: "+str(episode)+"\n")
  while True:

    # DISPLAY CURRENT STATE
    state = environment.getCurrentState()
    display(state)
    pause()
    
    # END IF IN A TERMINAL STATE
    actions = environment.getPossibleActions(state)
    if len(actions) == 0:
      message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
      return returns
    
    # GET ACTION (USUALLY FROM AGENT)
    action = decision(state)
    if action == None:
      raise 'Error: Agent returned None action'
    
    # EXECUTE ACTION
    nextState, reward, action2 = environment.doAction(action, qtype)
    message("Started in state: "+str(state)+
            "\nSpecified action: "+str(action)+
            "\nTook action: "+str(action2)+
            "\nEnded in state: "+str(nextState)+
            "\nGot reward: "+str(reward)+"\n")    
    # UPDATE LEARNER
    if 'observeTransition' in dir(agent): 
      agent.observeTransition(state, action2, nextState, reward, environment)
    if isinstance(reward, tuple):
       reward=sum(reward)
    
    returns += reward * totalDiscount
    totalDiscount *= discount

  if 'stopEpisode' in dir(agent):
    agent.stopEpisode()

def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="WalkGrid",
                         help='Grid to use (case sensitive; options are WalkGrid, ComboGrid, ObstacleGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=100,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="q",
                         help='Agent type (options is \'q\', default %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')
    optParser.add_option('-v', '--valueSteps',action='store_true' ,default=False,
                         help='Ignore this option.')

    opts, args = optParser.parse_args()
    
    if opts.manual and opts.agent != 'q':
      print '## Disabling Agents in Manual Mode (-m) ##'
      opts.agent = None

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
    # if opts.quiet:      
      opts.pause = False
      # opts.manual = False
      
    if opts.manual:
      opts.pause = True
      
    return opts

  
if __name__ == '__main__':
  
  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  import gridworld
  mdpFunction = getattr(gridworld, "get"+opts.grid)
  mdp = mdpFunction()
  #mdp.setLivingReward(opts.livingReward)
  #mdp.setNoise(opts.noise)
  #env = gridworld.GridworldEnvironment(mdp)

  
  ############################
  ## GET THE DISPLAY ADAPTER
  ############################

  import textGridworldDisplay
  #display = textGridworldDisplay.TextGridworldDisplay(mdp)
  if not opts.textDisplay:
    import graphicsGridworldDisplay
  #  display = graphicsGridworldDisplay.GraphicsGridworldDisplay(mdp, opts.gridSize, opts.speed)
  #display.start()

  ###########################
  # GET THE AGENT
  ###########################

  import valueIterationAgents, qlearningAgents
  a = None
  if opts.agent == 'value':
    a = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, opts.iters)
  elif opts.agent == 'q':
    #env.getPossibleActions, opts.discount, opts.learningRate, opts.epsilon
    #simulationFn = lambda agent, state: simulation.GridworldSimulation(agent,state,mdp)
    gridWorldEnv = GridworldEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    if(opts.grid=="ObstacleGrid"):
      qtype="obstacle"
    elif(opts.grid=="ComboGrid"):
      qtype="combo"
    else:
      qtype="walk"
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn,
                  'qtype': qtype}
    if(opts.grid=="ComboGrid"):
      a = qlearningAgents.modularQAgent(**qLearnOpts)
    else:
      a = qlearningAgents.QLearningAgent(**qLearnOpts)
  elif opts.agent == 'random':
    # # No reason to use the random agent without episodes
    if opts.episodes == 0:
      opts.episodes = 10
    class RandomAgent:
      def getAction(self, state):
        return random.choice(mdp.getPossibleActions(state))
      def getValue(self, state):
        return 0.0
      def getQValue(self, state, action):
        return 0.0
      def getPolicy(self, state):
        "NOTE: 'random' is a special policy value; don't use it in your code."
        return 'random'
      def update(self, state, action, nextState, reward):
        pass      
    a = RandomAgent()
  else:
    if not opts.manual: raise 'Unknown agent type: '+opts.agent
    
    
  ###########################
  # RUN EPISODES
  ###########################
  # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
  #if not opts.manual and opts.agent == 'value':
  #  if opts.valueSteps:
  #    for i in range(opts.iters):
  #      tempAgent = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, i)
  #      display.displayValues(tempAgent, message = "VALUES AFTER "+str(i)+" ITERATIONS")
  #      display.pause()        
  #  
  #  display.displayValues(a, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
  #  display.pause()
  #  display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.iters)+" ITERATIONS")
  #  display.pause()
  #  
  #

  ## FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
  #displayCallback = lambda x: None
  #if not opts.quiet:
  #  if opts.manual and opts.agent == None: 
  #    displayCallback = lambda state: display.displayNullValues(state)
  #  else:
  #    if opts.agent == 'random': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
  #    if opts.agent == 'value': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
  #    if opts.agent == 'q': displayCallback = lambda state: display.displayQValues(a, state, "CURRENT Q-VALUES")

  #messageCallback = lambda x: printString(x)
  #if opts.quiet:
  #  messageCallback = lambda x: None

  ## FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
  #pauseCallback = lambda : None
  #if opts.pause:
  #  pauseCallback = lambda : display.pause()

  ## FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)  
  #if opts.manual:
  #  decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)
  #else:
  #  decisionCallback = a.getAction  
    
  # RUN EPISODES
  if opts.episodes > 0:
    print
    print "RUNNING", opts.episodes, "EPISODES"
    print
  returns = 0
  for episode in range(1, opts.episodes+1):
    mdpFunction = getattr(gridworld, "get"+opts.grid)
    mdp = mdpFunction()
    mdp.setLivingReward(opts.livingReward)
    mdp.setNoise(opts.noise)
    env = gridworld.GridworldEnvironment(mdp)

    display = textGridworldDisplay.TextGridworldDisplay(mdp)
    if not opts.textDisplay:
      display = graphicsGridworldDisplay.GraphicsGridworldDisplay(mdp, opts.gridSize, opts.speed)
    display.start()
    # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
    if not opts.manual and opts.agent == 'value':
      if opts.valueSteps:
        for i in range(opts.iters):
          tempAgent = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, i)
          display.displayValues(tempAgent, message = "VALUES AFTER "+str(i)+" ITERATIONS")
          display.pause()        
      
      display.displayValues(a, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
      display.pause()
      display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.iters)+" ITERATIONS")
      display.pause()
    # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
    displayCallback = lambda x: None
    if not opts.quiet:
      if opts.manual and opts.agent == None: 
        displayCallback = lambda state: display.displayNullValues(state)
      else:
        if opts.agent == 'random': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
        if opts.agent == 'value': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
        if opts.agent == 'q': displayCallback = lambda state: display.displayQValues(a, state, "CURRENT Q-VALUES")
    messageCallback = lambda x: printString(x)
    if opts.quiet:
      messageCallback = lambda x: None
    # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
    pauseCallback = lambda : None
    if opts.pause:
      pauseCallback = lambda : display.pause()
    # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)  
    if opts.manual:
      decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)
    else:
      decisionCallback = a.getAction  
    returns += runEpisode(a, qtype, env, opts.discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
  if opts.episodes > 0:
    print
    print "AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes)
    print
    print
    
  # DISPLAY POST-LEARNING VALUES / Q-VALUES
  if opts.agent == 'q' and not opts.manual:
    display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.episodes)+" EPISODES")
    display.pause()
    display.displayValues(a, message = "VALUES AFTER "+str(opts.episodes)+" EPISODES")
    display.pause()
    
   
