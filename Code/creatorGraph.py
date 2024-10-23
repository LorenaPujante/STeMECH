
import networkx as nx

from querysGraph import *


#########################
# FUNCIONES PRINCIPALES #
#########################

# Crear grafo con las Locations
def getGraphLocations(driver):
    
    # Pedir las Locations a la BD
    q = QueryLocations()
    query = q.getQuery()
    results = driver.executeQuery(query)
    
    # Obtener Nodos y Aristas
    locations, edges = getListNodesEdges_Locations(results)
    beds = getBeds(locations)
    
    # Crear grafo
    G = nx.Graph()
    G.add_nodes_from(locations)
    G.add_edges_from(edges)

    return G, beds

def getBeds(locations):

    beds = []
    for l in locations:
        if l[1]['uri'].startswith('Bed'):
            beds.append(l[0])
    return beds



# Crear grafo con las HUs y Services
def getGraphLogicalLayout(driver):
    
    # Pedir las HUs y Servicios a la BD
    q = QueryLogicalStructure()
    query = q.getQuery()
    results = driver.executeQuery(query)

    # Obtener Nodos y Aristas
    logicalNodes, edges = getListNodesEdges_LogicalLayout(results)
    
    # Crear grafo
    G = nx.Graph()
    G.add_nodes_from(logicalNodes)
    G.add_edges_from(edges)

    return G



########################
# FUNCIONES AUXILIARES #
########################

def getListNodesEdges_Locations(results):
    locationsId = []
    locations = []
    edges = []

    for res in results['results']['bindings']:
        # Location 1
        locURI = res['l']['value']
        locId = extractNode(locURI, locations, locationsId)
        
        # Location 2
        loc1URI = res['l1']['value']
        loc1Id = extractNode(loc1URI, locations, locationsId)

        # Edge
        edgeClass = extractEdgeClass(res['rel']['value'])
        edgeCost = float(res['cost']['value'])
        edge = (locId, loc1Id, {'class': edgeClass, 'cost': edgeCost})
        edges.append(edge)

    return locations, edges


def getListNodesEdges_LogicalLayout(results):
    hus = []
    husId = []
    services = []
    servicesId = []
    edges = []

    for res in results['results']['bindings']:
        # HU
        huURI = res['hu']['value']
        huId = extractNode(huURI, hus, husId)

        # Service
        servURI = res['s']['value']
        servId = extractNode(servURI, services, servicesId)

        # Edge
        edge = (huId, servId)
        edges.append(edge)


    nodes = hus + services
    return nodes, edges



#######
# Aux #
#######
    
def extractNode(uri, listNodes, listIds):
    id = extractNodeId(uri)
    uriShort = extractNodeURIShort(uri)

    if id not in listIds:
        listIds.append(id)
        node = (id, {"uri": uriShort})
        listNodes.append(node)
        
    return id

def extractNodeId(uri):
    uriSplit = uri.split("/")
    id = uriSplit[-1]

    return id

def extractNodeURIShort(uri):
    uriSplit = uri.split("/")
    uriShort = uriSplit[-2]
    uriShort = uriShort.split("#")
    uriShort = uriShort[-1]
    
    id = uriSplit[-1]
    uriShort = "{}/{}".format(uriShort, id)
    
    return uriShort

def extractEdgeClass(uri):
    uriSplit = uri.split("#")
    uriShort = uriSplit[-1]
    
    return uriShort

