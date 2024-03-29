# Hint: from collections import deque
import sys
from collections import deque
from Interface import *


# = = = = = = = QUESTION 1  = = = = = = = #


def consistent(assignment, csp, var, value):
    """
    Checks if a value assigned to a variable is consistent with all binary constraints in a problem.
    Do not assign value to var.
    Only check if this value would be consistent or not.
    If the other variable for a constraint is not assigned,
    then the new value is consistent with the constraint.

    Args:
        assignment (Assignment): the partial assignment
        csp (ConstraintSatisfactionProblem): the problem definition
        var (string): the variable that would be assigned
        value (value): the value that would be assigned to the variable
    Returns:
        boolean
        True if the value would be consistent with all currently assigned values, False otherwise
    """
    # TODO: Question 1
    binaryCons = csp.binaryConstraints
    for binaryCon in binaryCons:
        if binaryCon.affects(var):
            othervar = binaryCon.otherVariable(var)
            if not binaryCon.isSatisfied(assignment.assignedValues[othervar],value):
                return False
    return True
    raise_undefined_error()


def recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod):
    """
    Recursive backtracking algorithm.
    A new assignment should not be created.
    The assignment passed in should have its domains updated with inferences.
    In the case that a recursive call returns failure or a variable assignment is incorrect,
    the inferences made along the way should be reversed.
    See maintainArcConsistency and forwardChecking for the format of inferences.

    Examples of the functions to be passed in:
    orderValuesMethod: orderValues, leastConstrainingValuesHeuristic
    selectVariableMethod: chooseFirstVariable, minimumRemainingValuesHeuristic
    inferenceMethod: noInferences, maintainArcConsistency, forwardChecking

    Args:
        assignment (Assignment): a partial assignment to expand upon
        csp (ConstraintSatisfactionProblem): the problem definition
        orderValuesMethod (function<assignment, csp, variable> returns list<value>):
            a function to decide the next value to try
        selectVariableMethod (function<assignment, csp> returns variable):
            a function to decide which variable to assign next
        inferenceMethod (function<assignment, csp, variable, value> returns set<variable, value>):
            a function to specify what type of inferences to use
    Returns:
        Assignment
        A completed and consistent assignment. None if no solution exists.
    """
    # TODO: Question 1
    if assignment.isComplete():
        return assignment
    cur_var=selectVariableMethod(assignment,csp)
    # current variable is not None
    if cur_var:
        for value in orderValuesMethod(assignment,csp,cur_var):
            if consistent(assignment,csp,cur_var,value):
                assignment.assignedValues[cur_var]=value
                inference = inferenceMethod(assignment,csp,cur_var,value)
                result = recursiveBacktracking(assignment,csp,orderValuesMethod,selectVariableMethod,inferenceMethod)

                if result is not None:
                    return result
                # no valid result
                else:
                    # recover assignment
                    assignment.assignedValues[cur_var] = None
                    if inference:
                        for i in inference:
                            assignment.varDomains[i[0]].add(i[1])   # recover inference

    return None
    raise_undefined_error()


def eliminateUnaryConstraints(assignment, csp):
    """
    Uses unary constraints to eleminate values from an assignment.

    Args:
        assignment (Assignment): a partial assignment to expand upon
        csp (ConstraintSatisfactionProblem): the problem definition
    Returns:
        Assignment
        An assignment with domains restricted by unary constraints. None if no solution exists.
    """
    domains = assignment.varDomains
    for var in domains:
        for constraint in (c for c in csp.unaryConstraints if c.affects(var)):
            for value in (v for v in list(domains[var]) if not constraint.isSatisfied(v)):
                domains[var].remove(value)
                # Failure due to invalid assignment
                if len(domains[var]) == 0:
                    return None
    return assignment


def chooseFirstVariable(assignment, csp):
    """
    Trivial method for choosing the next variable to assign.
    Uses no heuristics.
    """
    for var in csp.varDomains:
        if not assignment.isAssigned(var):
            return var


# = = = = = = = QUESTION 2  = = = = = = = #


def minimumRemainingValuesHeuristic(assignment, csp):
    """
    Selects the next variable to try to give a value to in an assignment.
    Uses minimum remaining values heuristic to pick a variable. Use degree heuristic for breaking ties.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
    Returns:
        the next variable to assign
    """
    nextVar = None
    domains = assignment.varDomains

    # TODO: Question 2
    min_domain=sys.maxsize
    # minVar=None
    most_degree=-1
    for var, domain in assignment.varDomains.items():
        if not assignment.isAssigned(var):
            if domain and len(domain)<min_domain:
                min_domain=len(domain)
                nextVar=var
                # minVar=var
                most_degree=0
                for constraint in csp.binaryConstraints:
                    if constraint.affects(var) and not assignment.isAssigned(constraint.otherVariable(var)):
                        most_degree+=1

            # tie condition
            elif domain and len(domain)==min_domain:
                degree=0
                for constraint in csp.binaryConstraints:
                    if constraint.affects(var) and not assignment.isAssigned(constraint.otherVariable(var)):
                        degree+=1
                if degree>most_degree:
                    min_domain=len(domain)
                    most_degree=degree
                    nextVar=var
    return nextVar
    raise_undefined_error()


def orderValues(assignment, csp, var):
    """
    Trivial method for ordering values to assign.
    Uses no heuristics.
    """
    return list(assignment.varDomains[var])


# = = = = = = = QUESTION 3  = = = = = = = #


def leastConstrainingValuesHeuristic(assignment, csp, var):
    """
    Creates an ordered list of the remaining values left for a given variable.
    Values should be attempted in the order returned.
    The least constraining value should be at the front of the list.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable to be assigned the values
    Returns:
        list<values>
        a list of the possible values ordered by the least constraining value heuristic
    """
    # TODO: Question 3
    satisfies = {}
    for value1 in assignment.varDomains[var]:
        satisfy1 = 0
        # satisfy2 = 0
        for constraint in csp.binaryConstraints:
            if constraint.affects(var):
                # the neighbor variable
                otherVar = constraint.otherVariable(var)
                # if assigned, pass
                if not assignment.isAssigned(otherVar):
                    for v in assignment.varDomains[otherVar]:
                        if constraint.isSatisfied(v,value1):
                            satisfy1+=1

                        # if contraint.isSatisfied(v,value2): satisfy2+=1
        # print(value1)
        # print(satisfies)
        satisfies[value1]=satisfy1

    all_values = list(assignment.varDomains[var])
    # temp = all_values.copy()
    all_values.sort(key=lambda x: satisfies[x], reverse=True)
    # print(all_values)
    return all_values
    raise_undefined_error()


def noInferences(assignment, csp, var, value):
    """
    Trivial method for making no inferences.
    """
    return set([])


# = = = = = = = QUESTION 4  = = = = = = = #


def forwardChecking(assignment, csp, var, value):
    """
    Implements the forward checking algorithm.
    Each inference should take the form of (variable, value)
    where the value is being removed from the domain of variable.
    This format is important so that the inferences can be reversed
    if they result in a conflicting partial assignment.
    If the algorithm reveals an inconsistency,
    any inferences made should be reversed before ending the function.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable that has just been assigned a value
        value (string): the value that has just been assigned
    Returns:
        set< tuple<variable, value> >
        the inferences made in this call or None if inconsistent assignment
    """
    inferences = set([])

    # TODO: Question 4
    for constraint in csp.binaryConstraints:
        if constraint.affects(var):
            otherVar=constraint.otherVariable(var)
            # flag = False
            # domain = assignment.varDomains[otherVar].copy()
            # for v in domain:
            #     if constraint.isSatisfied(v,value):
            #         flag = True
            #         inferences.add((otherVar,v))
            #         # assignment.varDomains[otherVar].remove(v)
            # # no satisfied neighbor values
            # if not flag:
            #     for i in inferences:
            #         assignment.varDomains[i[0]].add(i[1])
            #     return None
            if value in assignment.varDomains[otherVar]:
                if assignment.varDomains[otherVar]-set(value): # still have values to assign
                    inferences.add((otherVar,value))
                    # first delete same color from neighbors
                    assignment.varDomains[otherVar].discard(value)
                else:
                    # recover
                    for a, b in inferences:
                        assignment.varDomains[a].add(b)
                    return None
    return inferences
    raise_undefined_error()


# = = = = = = = QUESTION 5  = = = = = = = #

def revise(assignment, csp, var1, var2, constraint):
    """
    Helper function to maintainArcConsistency and AC3.
    Remove values from var2 domain if constraint cannot be satisfied.
    Each inference should take the form of (variable, value)
    where the value is being removed from the domain of variable.
    This format is important so that the inferences can be reversed
    if they result in a conflicting partial assignment.
    If the algorithm reveals an inconsistency,
    any inferences made should be reversed before ending the function.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var1 (string): the variable with consistent values
        var2 (string): the variable that should have inconsistent values removed
        constraint (BinaryConstraint): the constraint connecting var1 and var2
    Returns:
        set<tuple<variable, value>>
        the inferences made in this call or None if inconsistent assignment
    """
    inferences = set([])

    # TODO: Question 5
    # flag = False
    # for v in assignment.varDomains[var2]:
    #     if consistent(assignment,csp,var1,v):
    #         # inferences.add((var2,v))
    #         flag = True
    #
    # if not flag:
    #     for a,b in inferences:
    #         assignment.varDomains[a].discard(b)
    # else:
    #     for a,b in inferences:
    #         assignment.varDomains[a].add(b)
    #     return None
    # return inferences
    count=0
    for v2 in assignment.varDomains[var2]:
        remove=True
        for v1 in assignment.varDomains[var1]:
            if constraint.isSatisfied(v1,v2):
                remove=False
                break
        if remove:
            if (var2,v2) not in inferences:
                count+=1
            inferences.add((var2,v2))
    if count == len(assignment.varDomains[var2]):
        # for a,b in inferences:
        #     assignment.varDomains[a].add(b)
        return None
    for a,b in inferences:
        assignment.varDomains[a].discard(b)
    return inferences
    raise_undefined_error()


def maintainArcConsistency(assignment, csp, var, value):
    """
    Implements the maintaining arc consistency algorithm.
    Inferences take the form of (variable, value)
    where the value is being removed from the domain of variable.
    This format is important so that the inferences can be reversed
    if they result in a conflicting partial assignment.
    If the algorithm reveals an inconsistency,
    and inferences made should be reversed before ending the function.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable that has just been assigned a value
        value (string): the value that has just been assigned
    Returns:
        set<<variable, value>>
        the inferences made in this call or None if inconsistent assignment
    """
    inferences = set([])
    domains = assignment.varDomains

    # TODO: Question 5
    #  Hint: implement revise first and use it as a helper function"""

    q = deque()
    for constraint in csp.binaryConstraints:
        if constraint.affects(var) and not assignment.isAssigned(constraint.otherVariable(var)):
            q.append((var, constraint.otherVariable(var), constraint))

    while len(q)!=0:
        var1,var2,constraint = q.popleft()
        inference = revise(assignment,csp,var1,var2,constraint)

        if inference:
            inferences |= inference
            for c in csp.binaryConstraints:
                if c.affects(var2):
                    otherVar=c.otherVariable(var2)
                    if not assignment.isAssigned(otherVar):
                        q.append((var2,otherVar,c))
        elif inference is None:
            for a, b in inferences:
                assignment.varDomains[a].add(b)
            return None
    return inferences
    raise_undefined_error()


# = = = = = = = QUESTION 6  = = = = = = = #

def AC3(assignment, csp):
    """
    AC3 algorithm for constraint propagation.
    Used as a pre-processing step to reduce the problem
    before running recursive backtracking.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
    Returns:
        Assignment
        the updated assignment after inferences are made or None if an inconsistent assignment
    """
    inferences = set([])

    # TODO: Question 6
    #  Hint: implement revise first and use it as a helper function"""
    q = deque()
    for var in csp.varDomains:
        for constraint in csp.binaryConstraints:
            # if constraint.affects(var) and not assignment.isAssigned(constraint.otherVariable(var)):
            if constraint.affects(var):
                q.append((var, constraint.otherVariable(var), constraint))

    while len(q)!=0:
        var1, var2, constraint = q.popleft()
        inference = revise(assignment, csp, var1, var2, constraint)

        if inference:
            inferences |= inference
            for c in csp.binaryConstraints:
                if c.affects(var2):
                    otherVar = c.otherVariable(var2)
                    if not assignment.isAssigned(otherVar):
                        q.append((var2, otherVar, c))
        elif inference is None:
            for a, b in inferences:
                assignment.varDomains[a].add(b)
            return None
    return assignment
    raise_undefined_error()


def solve(csp, orderValuesMethod=leastConstrainingValuesHeuristic,
          selectVariableMethod=minimumRemainingValuesHeuristic,
          inferenceMethod=forwardChecking, useAC3=True):
    """
    Solves a binary constraint satisfaction problem.

    Args:
        csp (ConstraintSatisfactionProblem): a CSP to be solved
        orderValuesMethod (function): a function to decide the next value to try
        selectVariableMethod (function): a function to decide which variable to assign next
        inferenceMethod (function): a function to specify what type of inferences to use
        useAC3 (boolean): specifies whether to use the AC3 pre-processing step or not
    Returns:
        dictionary<string, value>
        A map from variables to their assigned values. None if no solution exists.
    """
    assignment = Assignment(csp)

    assignment = eliminateUnaryConstraints(assignment, csp)
    if assignment is None:
        return assignment

    if useAC3:
        assignment = AC3(assignment, csp)
        if assignment is None:
            return assignment

    assignment = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod)
    if assignment is None:
        return assignment

    return assignment.extractSolution()
