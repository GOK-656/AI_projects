import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        # print(legalMoves)
        # print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(newPos)
        # print(newFood)
        # print(newGhostStates[0])
        # print(newScaredTimes)
        ghostPositions=[ghostState.configuration.pos for ghostState in newGhostStates]
        # print(ghostPositions)
        # print(newPos)
        dist_food = 2/min([manhattanDistance(foodPos,newPos) for foodPos in newFood.asList()]) if newFood.asList() else 0
        ghosts = [manhattanDistance(newPos,ghostPos) for ghostPos in ghostPositions]
        dist_ghost = 1/min(ghosts) if min(ghosts)>0 else 0
        # print(newFood.asList())
        # print(dist_food)
        # return successorGameState.getScore()
        return dist_food-dist_ghost+successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        # self.evaluationFunction = better
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def value(self, gameState, depth, agent):
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        num = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent)
        if not actions:
            return self.evaluationFunction(gameState)
        next_agent = agent+1 if agent!=num-1 else 0
        next_depth = depth if next_agent!=0 else depth-1

        maxi = -sys.maxsize
        mini = sys.maxsize

        for action in actions:
            possible_value = self.value(gameState.generateSuccessor(agent,action),next_depth,next_agent)
            # print(possible_value)
            if possible_value>maxi:
                maxi = possible_value
            if possible_value<mini:
                mini=possible_value

        if agent==0:
            return maxi
        return mini


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        all_values = [self.value(gameState.generateSuccessor(0,action),self.depth,1) for action in actions]
        # print(max(all_values))
        ret = actions[all_values.index(max(all_values))]
        # print(ret)
        return ret
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def value(self, gameState, alpha,beta,agent,depth):
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions=gameState.getLegalActions(agent)
        # successors = [gameState.generateSuccessor(agent, action) for action in actions]
        next_agent=agent+1 if agent!=gameState.getNumAgents()-1 else 0
        depth=depth-1 if next_agent==0 else depth
        v=None
        if agent==0:
            v = -sys.maxsize
            # for successor in successors:
            #     print("1",successor)
            for action in actions:
                v = max(v, self.value(gameState.generateSuccessor(agent,action), alpha, beta, next_agent, depth))
                # print("2",gameState.generateSuccessor(agent,action))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            # return self.max_value(gameState,alpha,beta,agent,depth)
        else:
            v = sys.maxsize
            # for successor in successors:
            #     print("1",successor)
            for action in actions:
                v = min(v, self.value(gameState.generateSuccessor(agent,action), alpha, beta, next_agent, depth))
                # if last_agent==0:
                # print("2", gameState.generateSuccessor(agent, action))
                if v < alpha:
                    return v
                beta = min(beta, v)
            # return self.min_value(gameState, alpha, beta, agent, depth)
        return v

    # def max_value(self,state,alpha,beta,agent,depth):
    #     # if depth==0 or state.isWin() or state.isLose():
    #     #     return self.evaluationFunction(state)
    #     if agent!=0:
    #         print("Wrong call")
    #     v=-sys.maxsize
    #     actions=state.getLegalActions(agent)
    #     next_agent = agent+1
    #     # successors = [state.generateSuccessor(agent,action) for action in actions]
    #     for action in actions:
    #         v=max(v,self.value(state.generateSuccessor(agent,action),alpha,beta,next_agent,depth))
    #         if v>=beta:
    #             return v
    #         alpha=max(alpha,v)
    #     return v
    #
    # def min_value(self,state,alpha,beta,agent,depth):
    #     # if depth==0 or state.isWin() or state.isLose():
    #     #     return self.evaluationFunction(state)
    #     if agent==0:
    #         print("Wrong call")
    #     v=sys.maxsize
    #     actions=state.getLegalActions(agent)
    #     # last_agent = agent-1 if agent !=0 else state.getNumAgents()-1
    #     next_agent = agent+1 if agent !=state.getNumAgents()-1 else 0
    #     # successors = [state.generateSuccessor(agent,action) for action in actions]
    #     for action in actions:
    #         v=min(v,self.value(state.generateSuccessor(agent,action),alpha,beta,next_agent,depth))
    #         # if last_agent==0:
    #         if v<=alpha:
    #             return v
    #         beta = min(beta, v)
    #     return v


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        # all_values = [self.value(gameState.generateSuccessor(0, action), -sys.maxsize,sys.maxsize, 1,self.depth) for action in actions]
        # print(max(all_values))
        value=-sys.maxsize
        alpha=-sys.maxsize
        best=actions[0]
        for action in actions:
            v=self.value(gameState.generateSuccessor(0,action),alpha,sys.maxsize,1,self.depth)
            if v>value:
                value=v
                best=action
            alpha=max(alpha,v)
        # print(ret)
        return best
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def value(self, gameState, depth, agent):
        if depth==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        num = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent)
        if not actions:
            return self.evaluationFunction(gameState)
        next_agent = agent+1 if agent!=num-1 else 0
        next_depth = depth-1 if agent==num-1 else depth

        maxi = -sys.maxsize
        expecti = 0.

        for action in actions:
            possible_value = self.value(gameState.generateSuccessor(agent,action),next_depth,next_agent)
            # print(possible_value)
            if possible_value>maxi:
                maxi = possible_value
            expecti += possible_value

        expecti /= len(actions)
        if agent==0:
            return maxi
        return expecti

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        all_values = [self.value(gameState.generateSuccessor(0, action), self.depth, 1) for action in actions]
        # print(max(all_values))
        ret = actions[all_values.index(max(all_values))]
        # print(ret)
        return ret
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    dist_food = sys.maxsize
    if len(foods):
        for food in foods:
            dist_food=min(dist_food,manhattanDistance(food,pos))

    dist_ghost = sys.maxsize
    dist_scared = sys.maxsize
    for i in range(len(ghostStates)):
        if scaredTimes[i]<=0:
            dist_ghost=min(manhattanDistance(ghostStates[i].getPosition(),pos),dist_ghost)
        else:
            dist_scared=min(manhattanDistance(ghostStates[i].getPosition(),pos),dist_scared)

    ret = currentGameState.getScore()
    ret += 5/dist_food if dist_food!=0 else 0
    ret += -10/dist_ghost if dist_ghost!=0 else 0
    ret += 2/dist_scared if dist_scared!=0 else 0
    return ret
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
