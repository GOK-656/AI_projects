import sys

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            # kth value
            it=self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                else:
                    actions=self.mdp.getPossibleActions(state)
                    q_values=[self.getQValue(state,action) for action in actions]
                    it[state]=max(q_values)
            # update k+1 th
            self.values=it


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_values = [p*(self.mdp.getReward(state, action, next_state)+self.discount*self.values[next_state]) for
                    next_state, p in self.mdp.getTransitionStatesAndProbs(state, action)]
        return sum(q_values)
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions=self.mdp.getPossibleActions(state)
        ret = None
        v = -sys.maxsize

        for action in actions:
            if self.getQValue(state, action) > v:
                v=self.getQValue(state, action)
                ret=action
        return ret
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        num = len(states)
        for i in range(self.iterations):
            # kth value
            state=states[i%num]
            if self.mdp.isTerminal(state):
                continue
            else:
                actions=self.mdp.getPossibleActions(state)
                q_values=[self.getQValue(state,action) for action in actions]
                self.values[state]=max(q_values)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predessors=dict()
        states=self.mdp.getStates()
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for next_state, p in self.mdp.getTransitionStatesAndProbs(state, action):
                    if next_state not in predessors.keys():
                        predessors[next_state]=set()
                    if p>0:
                        predessors[next_state].add(state)

        # initialize the priority queue
        pq=util.PriorityQueue()
        for s in states:
            if self.mdp.isTerminal(s):
                continue
            curr=self.values[s]
            high=max([self.getQValue(s,action) for action in self.mdp.getPossibleActions(s)])
            diff=abs(curr-high)
            pq.push(s, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s=pq.pop()
            if self.mdp.isTerminal(s):
                continue
            self.values[s]=max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            for p in predessors[s]:
                curr = self.values[p]
                high = max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)])
                diff = abs(curr - high)
                if diff>self.theta:
                    pq.update(p, -diff)

