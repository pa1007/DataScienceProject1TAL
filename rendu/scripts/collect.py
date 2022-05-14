#!/usr/bin/env python
# coding: utf-8

import json
import string

import pandas as pd
import wikipedia
import wptools
from SPARQLWrapper import SPARQLWrapper, JSON
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize

# For each category we only select one type of dbpedia object to be sure what we have is what we need
catType = {"Airports": "?a a dbo:Infrastructure .", "Comics_characters": "?a a dbo:Agent .",
           "Artist": "?a a dbo:Animal .",
           "Astronauts": '?a a dbo:Animal .', "Building": "?a a dbo:Building .", "Astronomical_objects": "",
           "City": "?a a dbo:Place .", "Companies": "?a a dbo:Company .", "Foods": "?a a dbo:Food .",
           "Transport": "?a a dbo:MeanOfTransportation .", "Monuments_and_memorials": "?a a dbo:Place .",
           "Politicians": "?a a dbo:Animal .", "Sports_teams": "?a a dbo:Organisation .",
           "Sportspeople": "?a a dbo:Animal .", "Universities_and_colleges": "?a a dbo:Organisation .",
           "Written_communication": ""}

# All the object we need
stop_words = stopwords.words('english')
porter = PorterStemmer()
sparql_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")
sparql_dbpedia.setReturnFormat(JSON)
sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql_wd.setReturnFormat(JSON)


k = 15 # the number of item per category
n = 2 # The number of line we need per wikipedia page to keep it


# ## Exercise 1 :

data = [] # We init the list of the articles
for cat in catType:

    # Sparql request for getting info we need to recover the pages we need per category
    # We make sure to remove the list of something articles

    SPARQL_GET_LISTS = f'''
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX dbp: <http://dbpedia.org/property/>
    PREFIX dbc: <http://dbpedia.org/resource/Category:>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dct: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?page ?p ?code WHERE {{
    ?a dct:subject/skos:broader dbc:{cat} .
    {catType[cat]}
    ?a rdfs:label ?label .
    ?a foaf:name ?page .
    ?p foaf:primaryTopic ?a .
    ?a dbo:wikiPageID ?code .
    FILTER (lang(?label) = 'en')
    FILTER(!STRSTARTS(?label, "List of"))
    }}limit {k}
    '''
    sparql_dbpedia.setQuery(SPARQL_GET_LISTS)
    try:
        ret = sparql_dbpedia.queryAndConvert()
        for r in ret["results"]["bindings"]: # for each result
            url_name = r["code"]['value']  # get the wikipedia page id
            page = wikipedia.page(pageid=url_name) # get the page
            cont = page.content
            number_of_sentences = len(sent_tokenize(cont)) # count the number of sentence we have with nltk
            if number_of_sentences >= n:
                p = wptools.page(pageid=url_name, silent=True)
                p.get_parse() # get infobox
                p.get_wikidata() #get the wikidata info
                data.append( # We store the data
                    {"name": r["page"]['value'], "url_name": url_name, "txt": page.content,
                     "infobox": p.data["infobox"],
                     "wikidata": p.data["wikidata"],
                     "cat": cat, "url": r["p"]['value'], "wikibase": p.data["wikibase"]})
    except Exception as e:
        print()
# We save it to be sure this costly method doesn't need to be rerun
with open("data.json", "w") as f:
    json.dump(data, f)


# ## Exercise 2 :
# 

file = open("data.json", "r") # We load the data from the file
data_to_process = json.load(file)


# In[7]:


# this method is used to pre-process text
# We make it lower,than remove the punctuation, than tokenize, we remove stop words
# We stem every word (get the base word without conjugation
# We tag every word, used later to compare data
def preprocess(txt):
    if type(txt) == str:
        text = txt.lower()
        text_p = "".join([char for char in text if char not in string.punctuation])
        words = word_tokenize(text_p)
        filtered_words = [word for word in words if word not in stop_words]
        stemmed = [porter.stem(word) for word in filtered_words]
        pos = pos_tag(stemmed)
        return pos, stemmed
    else:
        return txt

# This method will get the description in english of every wikidata item we have in the copus
def getDescription(items):
    item = " wd:".join(items) # We generate a list of item in sparql
    SPARQL_DESCR = f'''
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX schema: <http://schema.org/>
    SELECT distinct ?o ?item WHERE {{
    ?item schema:description ?o .
    FILTER ( lang(?o) = "en" )
    VALUES ?item {{ wd:{item} }}
    }}
    '''
    sparql_wd.setQuery(SPARQL_DESCR)
    try:
        res = sparql_wd.queryAndConvert()
        retu = {}
        for x in res["results"]["bindings"]:
            retu.update({x['item']['value'].replace("http://www.wikidata.org/entity/", ""): x['o']['value']}) #get in the dict the item and update the description
        return retu
    except Exception as e:
        print(e)


res = []
to_get = []
for item in data_to_process:
    txt = preprocess(item["txt"]) # Preprocess the text
    to_get.append(item["wikibase"]) # Add item to get the description later
    res.append(
        {"person": item['name'], "text": item['txt'], "processed_text": txt, "desc": "", "processed_desc": "",
         "cat": item['cat'], "base": item["wikibase"]})

descr = getDescription(to_get) # Get the description of every item
for x in res:
    x.update({"desc": descr.get(x['base']), "processed_desc": preprocess(descr.get(x['base']))}) #process the description and save evrything

df = pd.DataFrame(res) # Create the dataframe needed in the project

# We save the dataframe to recover it later
with open("dataframe.json","w") as f :
    f.write(df.to_json())

