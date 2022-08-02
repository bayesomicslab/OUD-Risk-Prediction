import numpy as np
import pandas as pd
import pickle

'''
Node class to hold information about each centroid that we found from dbscan.
A node has the information on a single building,  the location of that building (latitude and longitude),
the types that building could be, the left node, and the right node.

Functions:
Location - returns the location information of a specific node (latitude and longitude)
Name - returns the name of a specific node i.e. name of a location given by google API
Type - returns the types that a location my based on google api places 
'''
class Node:

    '''
    data - a list consisting of [location name, location types, and location points]
    left - the node to the left in depth + 1
    right - the node to the right in depth + 1
    '''
    def __init__(self, data, left, right):

        self.location = data[2]
        self.name = data[0]
        self.type = data[1]
        self.left = left
        self.right = right

    def Location(self):
        return self.location
    def Name(self):
        return self.name
    def Type(self):
        return self.type

'''
KDTree class is designed to hold our tree containing all of the nodes that represent a significant location determined
by the DBSCAN centroids. Each individual node is its own significant location.
A KDTree is similar to a binary search except it rotates the axis that it splits on on each time which is dependent on 
the depth in the tree you are on and the number of axis you have in the graph.

Building a tree - to build a tree you give it a list of lists. If we have list x then x[0]...x[n] will look as followed
    [location name, location types, and location points]
queryTree - Needs a query point passed to it and the number of axis it will return the closest point to that location.
'''
class KDTree:

    '''
    Builds the KDtree using the node class above.

    Attribute - tree
    Requires the data input to be in the form of lists of lists as described above
    The number of axis K
    '''
    def __init__(self, data, k = 2):

        self.tree = self.__buildTree(data, k, 0)


    '''
    Recursively builds the tree.
    Gets the length of each subtree, sorts it based off a specific axis given the depth, and makes a node recursively
    calls on each side of the split list
    '''
    def __buildTree(self, data, k = 2, depth = 0):

        num_points = len(data)
        if num_points <= 0:
            return None

        data.sort(key=lambda p: p[2][depth % k])

        return Node(data[num_points//2], left = self.__buildTree(data[: num_points // 2], k , depth + 1),
                    right = self.__buildTree(data[num_points // 2 + 1 :], k , depth + 1))

    '''
    Euclidean distance
    '''
    def __calc(self,q,p):
        x = p.location[0] - q[0]
        y = p.location[1] - q[1]
        return np.sqrt(x ** 2 + y ** 2)

    '''
    Calculate the distance of 2 points compared to the query point being observed.
    Reutrns the better point and the distance of it to the query point
    '''
    def __dist(self,query, p1, p2 = None):

        if p1 is None:
            return (p2,self.__calc(query,p2))
        if p2 is None:
            return (p1,self.__calc(query,p1))

        dist1 = self.__calc(query,p1)
        dist2 = self.__calc(query,p2)

        return (p2,dist2) if dist1 > dist2 else (p1,dist1)

    '''
    Recursive function to identify the closest point to the query point.
    
    sub_tree is the current point on, query is the point to look at it, k is the axis
    '''
    def __query_helper(self, sub_tree = None, query = None, k = 2, depth = 0):
        if sub_tree is None:
            return None

        # Binary search. If the subtree point axis is greater then that axis for query we set left to left right to right
        if sub_tree.location[depth % k] > query[depth % k]:
            left = sub_tree.left
            right = sub_tree.right
        else:
            left = sub_tree.right
            right = sub_tree.left

        # Get the distance the query to the current point and a call to the left side
        ret = self.__dist(query, self.__query_helper(left,query, k, depth + 1), sub_tree)

        # identify if the left node is better we keep that one else we search the right side
        return ret[0] if abs(query[depth % k] - sub_tree.location[depth % k]) > self.__dist(query, ret[0])[1] else self.__dist(query, self.__query_helper( right,query, k, depth + 1), ret[0])[0]

    '''
    function to query the tree for a specific point and how close it is to a specific location
    '''
    def queryTree(self, query = None, k = 2):
        if query is None:
            return None

        return self.__query_helper(self.tree,query, k)

if __name__ = '__main__':

    data = 'scraper/data/newData.csv'
    location_data = pd.read_csv(data,sep='\t')

    # Extract individual elements from each column
    name = list(location_data.Name)
    type = list(location_data.newType)
    lat = list(location_data.Latitude)
    long = list(location_data.Longitude)

    # Clean up longitude locations
    long_floats = []
    for item in long:
        if isinstance(item, str):
            if '!3m' in item:
                otherData = item.split('!3m')
                long_data = float(otherData[0])
                long_floats += [long_data]
            else:
                long_floats += [float(item)]
        else:
            long_floats += [float(item)]
    locations = [(float(lat[i]), long_floats[i]) for i in range(len(lat))]

    # eliminate duplicate locations
    data = {}
    for i in range(len(name)):

        if (name[i],locations[i]) not in data:
            data[(name[i],locations[i])] = type[i]

    # set up input
    data_points = []
    for item in data:
        data_points += [[item[0],data[item],item[1]]]

    # building a tree
    location_tree = KDTree(data_points)


    # Showing how to query the tree -- Example brown alumni center
    xxx = location_tree.queryTree([41.8251263,-71.402987])
    xxx.Name()
    xxx.Location()
    xxx.Type()

    # saving tree and testing its ability to open
    with open('locationsTree.pickle', 'wb') as f:
        pickle.dump(location_tree, f)
    f.close()

    ff = open('/Users/devinjmcconnell/Documents/Research/Opioid-LifeRhythm/scraper/locationsTree.pickle', 'rb')
    newTree = pickle.load(ff)
    ff.close()

    testing_loaded = location_tree.queryTree([41.8251263, -71.402987])
    testing_loaded.Name()


