
A reinforcement learning agent/bot that navigates an obstacle ridden gridworld.

The code is in the src file.

To see the options:

python gridworld.py -h

Usage: gridworld.py [options]

Options:
  -h, --help            show this help message and exit
  -d DISCOUNT, --discount=DISCOUNT
                        Discount on future (default 0.9)
  -r R, --livingReward=R
                        Reward for living for a time step (default 0.0)
  -n P, --noise=P       How often action results in unintended direction
                        (default 0.2)
  -e E, --epsilon=E     Chance of taking a random action in q-learning
                        (default 0.3)
  -l P, --learningRate=P
                        TD learning rate (default 0.5)
  -i K, --iterations=K  Number of rounds of value iteration (default 10)
  -k K, --episodes=K    Number of epsiodes of the MDP to run (default 1)
  -g G, --grid=G        Grid to use (case sensitive; options are WalkGrid,
                        ComboGrid, ObstacleGrid, default WalkGrid)
  -w X, --windowSize=X  Request a window width of X pixels *per grid cell*
                        (default 100)
  -a A, --agent=A       Agent type (options is 'q', default q)
  -t, --text            Use text-only ASCII display
  -p, --pause           Pause GUI after each time step when running the MDP
  -q, --quiet           Skip display of any learning episodes
  -s S, --speed=S       Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0
                        is slower (default 1.0)
  -m, --manual          Manually control agent
  -v, --valueSteps      Ignore this option.


To run Walking agent and view values after 200 iterations:
python gridworld.py -a q -k 200 -g WalkGrid -q


To run Obstacle agent and view values after 300 iterations:
python gridworld.py -a q -k 300 -g ObstacleGrid -q

To run Combo agent and view values after 300 iterations:
python gridworld.py -a q -k 300 -g ComboGrid -q

To run the code in manual mode for 5 iterations:
python gridworld.py -a q -k 5 -m -g ObstacleGrid

