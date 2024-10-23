from SPARQLWrapper import SPARQLWrapper, JSON

class DriverGraphDB:

    URL_BASE = "http://localhost:7200/"


    def __init__(self):
        self.repository = ""
    
    def setRepository(self, repository):
        self.repository = repository


    def executeQuery(self, query):
        url = self.URL_BASE + "repositories/" + self.repository
        sparql = SPARQLWrapper(url)

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        return results
