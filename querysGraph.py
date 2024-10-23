
###############################################################
# QUERIES PARA LA CREACION DEL GRAFO/MATRIZ DE LOCALIZACIONES #
###############################################################

class QueryLocations:

    def getQuery(self):
        
        query = ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "SELECT DISTINCT ?l ?rel ?l1 ?cost\n"
        query += "WHERE\n"
        query += "{\n"
        query += "?l a ho:Location.\n"
        query += "?l1 a ho:Location.\n"
        query += "<<?l ?rel ?l1>> ho:cost ?cost\n"
        query += "}\n"
        
        #query += "ORDER BY ?l\n"
        #query += "LIMIT 3000\n"

        return query     


class QueryLogicalStructure:

    def getQuery(self):
        
        query = ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "SELECT DISTINCT ?s ?hu \n"
        query += "WHERE\n"
        query += "{\n"
        query += "?hu a ho:HospitalizationUnit.\n"
        query += "?s a ho:Service.\n"
        query += "?hu ho:hospUnitFromService ?s.\n"
        query += "}\n"

        return query
    


