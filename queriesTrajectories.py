
############################################
# QUERIES PARA LA CREACION DE TRAYECTORIAS #
############################################

class Q2:

    def getQuery(self, dateStart, dateEnd, idLoc, idMicroorg):

        query = ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT DISTINCT ?p_id #?m ?p #?m ?tm ?ep ?p ?ev ?bed ?loc\n"
        query += "{\n"
        query += "VALUES (?start ?end) {{(\"{}\"^^xsd:dateTime \"{}\"^^xsd:dateTime)}}\n".format(dateStart, dateEnd)
        query += "VALUES (?loc_id ?m_id) {{({} {})}}\n".format(idLoc, idMicroorg)
        # TestMicro with their Patients and Episodes
        query += "?tm ho:eventFromEpisode ?ep;\n"
        query += "a ho:TestMicro;\n"
        query += "ho:start ?tm_start;\n"
        query += "ho:hasFound ?m.\n"
        query += "?m ho:id ?m_id.\n"
        query += "?ep ho:episodeFromPatient ?p.\n"
        query += "?p ho:id ?p_id.\n"
        # Other Events from the Patients and their Beds
        query += "?ev ho:eventFromEpisode ?ep;\n"
        query += "ho:start ?ev_start;\n"
        query += "ho:end ?ev_end;\n"
        query += "ho:hasLocation ?bed.\n"
        # Check if the Bed is in the Location
        query += "?loc (^ho:placedIn)+ ?bed;\n"
        query += "a ho:Location;\n"
        query += "ho:id ?loc_id.\n"
        # The Patient had a positive TestMicro during the search time
        query += "FILTER((?tm_start >= ?start) && (?tm_start <= ?end))\n"
        # The Patient was in the Location during the search time.
        # However, it is not necessary for the TestMicro to happen during the Event
        query += "FILTER((?ev_start >= ?start && ?ev_end <= ?end)\n"
        query += "|| (?ev_start <= ?start && ?ev_end >= ?start)\n"
        query += "|| (?ev_start <= ?end && ?ev_end >= ?end))\n"
        query += "} ORDER BY ?m ?p\n"

        return query
    

class QueryGetEventsFromPatsQ2:

    # Devuelve todos los Eventos con Cama y HospUnit de los Pacientes pasados como parámetro. Los Eventos deben ocurrir entre el periodo definido
    def getQuery(self, dateStart, dateEnd, idsPat):      #idsPat = (1, 2, 3)
        
        query =  ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT DISTINCT ?p_id ?ev ?ev_start ?ev_end ?bed_id ?hu_id\n"
        query += "{\n"
        query += "VALUES (?start ?end) {{(\"{}\"^^xsd:dateTime \"{}\"^^xsd:dateTime)}}\n".format(dateStart, dateEnd)
        query += "?p ^(ho:episodeFromPatient) ?ep;\n"
        query += "a ho:Patient;\n"
        query += "ho:id ?p_id.\n"
        query += "?ev ho:eventFromEpisode ?ep;\n"
        query += "ho:hasLocation ?bed;\n"
        query += "ho:start ?ev_start;\n"
        query += "ho:end ?ev_end;\n"
        query += "ho:hasHospUnit ?hu.\n"
        query += "?hu ho:id ?hu_id.\n"
        query += "?bed ho:id ?bed_id.\n"
        query += "FILTER(?p_id in {})\n".format(idsPat)
        query += "FILTER((?ev_start >= ?start && ?ev_end <= ?end) || (?ev_start <= ?start && ?ev_end >= ?start) || (?ev_start <= ?end && ?ev_end >= ?end))\n"
        query += "} ORDER BY ?p ?ev_start"

        return query






######################
# QUERIES TEMPORALES #
######################

class QueryGetLastTest:
    
    def getQuery(self, dateStart, dateEnd, idMicroorg, idsPat):

        query = ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT (MAX(?tm_start) AS ?lastTMDate)\n"
        query += "{\n"
        query += "VALUES (?start ?end) {{(\"{}\"^^xsd:dateTime \"{}\"^^xsd:dateTime)}}\n".format(dateStart, dateEnd)
        query += "VALUES ?m_id {{{}}}\n".format(idMicroorg)
        query += "?p a ho:Patient;\n"
        query += "ho:id ?p_id.\n"
        query += "FILTER(?p_id in {})\n".format(idsPat)
        query += "?tm a ho:TestMicro;\n"
        query += "ho:eventFromEpisode/ho:episodeFromPatient ?p;\n"
        query += "ho:start ?tm_start;\n"
        query += "ho:hasFound ?m.\n"
        query += "?m ho:id ?m_id.\n"
        query += "FILTER((?tm_start >= ?start) && (?tm_start <= ?end))\n"
        query += "}\n"

        return query
    

    def getQuery_byPatient(self, dateEnd, idMicroorg, idsPat):
        
        query = ""

        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT ?p_id (MAX(?tm_start) AS ?lastTMDate)\n"
        query += "{\n"
        query += "VALUES ?maxDate {{\"{}\"^^xsd:dateTime}}\n".format(dateEnd)
        query += "VALUES ?m_id {{{}}}\n".format(idMicroorg)
        query += "?p a ho:Patient;\n"
        query += "ho:id ?p_id.\n"
        query += "FILTER(?p_id in {})\n".format(idsPat)
        query += "?tm a ho:TestMicro;\n"
        query += "ho:eventFromEpisode/ho:episodeFromPatient ?p;\n"
        query += "ho:start ?tm_start;\n"
        query += "ho:hasFound ?m.\n"
        query += "?m ho:id ?m_id.\n"
        query += "FILTER(?tm_start <= ?maxDate)\n"
        query += "}\n"
        query += "GROUP BY ?p_id\n"

        return query



class QueryGetFirstEpisode:
    
    def getQuery(self, idsPat):

        query = ""    
        
        query += "PREFIX ho: <http://www.semanticweb.org/spatiotemporalHospitalOntology#>\n"
        query += "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        query += "SELECT (MIN(?ep_start) AS ?firstEpDate)\n"
        query += "{\n"
        query += "?p a ho:Patient;\n"
        query += "ho:id ?p_id.\n"
        query += "FILTER(?p_id in {})\n".format(idsPat)
        query += "?ep a ho:Episode;\n"
        query += "ho:episodeFromPatient ?p;\n"
        query += "ho:start ?ep_start;\n"        # Esto es posible porque en el simulador cada Paciente solo tiene un Episodio (no hay reingresos)
        query += "}\n"

        return query