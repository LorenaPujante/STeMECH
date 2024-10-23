
# QUERIES PARA HACER PRUEBAS, NO SE USAN EN EL PROGRAMA


######################
# QUERIES TEMPORALES #
######################

# Primera y Última fecha y hora de la base de datos
class QueryDates:
    
    def getLastDate(self):
        query = ""
        
        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT (MAX(?ev_end) AS ?maxDate)\n"
        query += "WHERE {\n"
        query += "?ev a ho:Event;\n"
        query += "ho:end ?ev_end.\n"
        query += "}\n"

        return query
    
    def getFirstDate(self):
        query = ""
        
        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT (MIN(?ev_end) AS ?minDate)\n"
        query += "WHERE {\n"
        query += "?ev a ho:Event;\n"
        query += "ho:end ?ev_end.\n"
        query += "}\n"

        return query
    



##########################################################
# QUERIES PARA OBTENER NÚMERO DE PACIENTES CON TESTMICRO #
##########################################################

class QueryNumTestMicro:
    
    def getQuery_7days(self, dateStart, dateEnd):

        query = ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT ?m_id (COUNT(DISTINCT ?p) as ?count)\n"
        query += "WHERE {\n"
        query += "VALUES (?minDate ?maxDate) {{(\"{}\"^^xsd:dateTime \"{}\"^^xsd:dateTime)}}\n".format(dateStart, dateEnd)
        query += "?tm a ho:TestMicro;\n"
        query += "ho:start ?tm_start;\n"
        query += "ho:hasFound ?m.\n"
        query += "?m ho:id ?m_id.\n"
        query += "?tm ho:eventFromEpisode/ho:episodeFromPatient ?p.\n"
        query += "FILTER(?tm_start >= ?minDate  &&  ?tm_start <= ?maxDate)\n"
        query += "}\n"
        query += "GROUP BY ?m_id\n"

        return query


    def getQuery_7days_byFloor(self, dateStart, dateEnd):

        query = ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT ?m_id ?floor_id ?floor_desc (COUNT(DISTINCT ?p) as ?count)\n"
        query += "WHERE {\n"
        query += "VALUES (?minDate ?maxDate) {{(\"{}\"^^xsd:dateTime \"{}\"^^xsd:dateTime)}}\n".format(dateStart, dateEnd)
        query += "?tm a ho:TestMicro;\n"
        query += "ho:start ?tm_start;\n"
        query += "ho:hasFound ?m.\n"
        query += "?m ho:id ?m_id.\n"
        query += "?tm ho:eventFromEpisode ?ep.\n"
        query += "?ep ho:episodeFromPatient ?p.\n"
        query += "?ev ho:eventFromEpisode ?ep;\n"
        query += "ho:hasLocation ?bed;\n"
        query += "ho:start ?ev_start;\n"
        query += "ho:end ?ev_end.\n"
        query += "?bed ho:placedIn+ ?floor.\n"
        query += "?floor a ho:Floor;\n"
        query += "ho:id ?floor_id;\n"
        query += "ho:description ?floor_desc.\n"
        query += "FILTER ((?tm_start >= ?ev_start) && (?tm_start <= ?ev_end))\n"
        query += "FILTER(?tm_start >= ?minDate  &&  ?tm_start <= ?maxDate)\n"
        query += "FILTER((?ev_start >= ?minDate && ?ev_end <= ?maxDate)\n"
        query += "|| (?ev_start <= ?minDate && ?ev_end >= ?minDate)\n"
        query += "|| (?ev_start <= ?maxDate && ?ev_end >= ?maxDate))\n"
        query += "}\n"
        query += "GROUP BY ?m_id ?floor_id ?floor_desc\n"
        query += "ORDER BY ?m_id ?floor_id\n"

        return query
    

    def getQuery_7days_byFloor_lax(self, dateStart, dateEnd):

        query = ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT ?m_id ?floor_id ?floor_desc (COUNT(DISTINCT ?p) as ?count)\n"
        query += "WHERE {\n"
        query += "VALUES (?minDate ?maxDate) {{(\"{}\"^^xsd:dateTime \"{}\"^^xsd:dateTime)}}\n".format(dateStart, dateEnd)
        query += "?tm a ho:TestMicro;\n"
        query += "ho:start ?tm_start;\n"
        query += "ho:hasFound ?m.\n"
        query += "?m ho:id ?m_id.\n"
        query += "?tm ho:eventFromEpisode ?ep.\n"
        query += "?ep ho:episodeFromPatient ?p.\n"
        query += "?ev ho:eventFromEpisode ?ep;\n"
        query += "ho:hasLocation ?bed;\n"
        query += "ho:start ?ev_start;\n"
        query += "ho:end ?ev_end.\n"
        query += "?bed ho:placedIn+ ?floor.\n"
        query += "?floor a ho:Floor;\n"
        query += "ho:id ?floor_id;\n"
        query += "ho:description ?floor_desc.\n"
        query += "FILTER(?tm_start >= ?minDate  &&  ?tm_start <= ?maxDate)\n"
        query += "FILTER((?ev_start >= ?minDate && ?ev_end <= ?maxDate)\n"
        query += "|| (?ev_start <= ?minDate && ?ev_end >= ?minDate)\n"
        query += "|| (?ev_start <= ?maxDate && ?ev_end >= ?maxDate))\n"
        query += "}\n"
        query += "GROUP BY ?m_id ?floor_id ?floor_desc\n"
        query += "ORDER BY ?m_id ?floor_id\n"

        return query