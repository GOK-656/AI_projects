# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack
from util import Queue
from util import PriorityQueue
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
         (action, stepCost, successor), where 'action' is the action
         required to get there, 'stepCost' is the incremental
         cost of expanding to that successor and 'successor' is a
         successor to the current state.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    frontier = Stack()
    initialState = problem.getStartState()
    frontier.push((None, None, initialState))
    actions = []
    explored = dict()
    while not frontier.isEmpty():
        prevState, direction, currState = frontier.pop()

        explored[currState] = (prevState, direction)

        if problem.isGoalState(currState):
            temp = currState
            # print(explored)
            # print(temp)
            while temp in explored.keys():
                actions.append(explored[temp][1])
                # print(explored[temp])
                temp = explored[temp][0]
            # print(temp)
            # print(actions[:-1][::-1])
            return actions[-2::-1]

        successors = problem.getSuccessors(currState)
        for dir, cost, nextState in successors:
            if nextState not in explored.keys():
                frontier.push((currState, dir, nextState))

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = Queue()
    initialState = problem.getStartState()
    frontier.push((None, None, initialState))
    actions = []
    explored = dict()
    while not frontier.isEmpty():
        prevState, direction, currState = frontier.pop()

        explored[currState] = (prevState, direction)

        if problem.isGoalState(currState):
            temp = currState
            # print(explored)
            # print(temp)
            while temp in explored.keys():
                actions.append(explored[temp][1])
                # print(explored[temp])
                temp = explored[temp][0]
            # print(temp)
            # print(actions[:-1][::-1])
            return actions[-2::-1]

        successors = problem.getSuccessors(currState)
        for dir, cost, nextState in successors:
            if nextState not in explored.keys():
                frontier.push((currState, dir, nextState))

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = PriorityQueue()
    initialState = problem.getStartState()
    frontier.push((None, None, initialState), 0)
    actions = []
    explored = dict()
    while not frontier.isEmpty():
        curr_priority = frontier.heap[0][0]
        print(curr_priority)
        prevState, direction, currState = frontier.pop()

        explored[currState] = (prevState, direction)

        if problem.isGoalState(currState):
            temp = currState
            # print(explored)
            # print(temp)
            while temp in explored.keys():
                actions.append(explored[temp][1])
                # print(explored[temp])
                temp = explored[temp][0]
            # print(temp)
            # print(actions[:-1][::-1])
            return actions[-2::-1]

        successors = problem.getSuccessors(currState)
        for dir, cost, nextState in successors:
            if nextState not in explored.keys():
                frontier.update((currState, dir, nextState), curr_priority + cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
