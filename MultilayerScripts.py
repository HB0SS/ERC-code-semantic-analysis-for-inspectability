#For now, make a script that's solely for the Dutch Primary corpus, in which I disregard the 1 French work.
#I.e. everything is in latin, which means we need only 1 stopwordlist, and one sort of lemmatizer

#import relevant packages general packages

import os
import string
import numpy
import math
import csv
from collections import Counter

#CLTK is for the processing of Latin text - NLTK for modern text
import cltk
import nltk

#initialize the lemmatizer
#Latin
from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
lemmatizer = BackoffLatinLemmatizer()
#English
from nltk.corpus import wordnet
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizerE = WordNetLemmatizer()
nltk.download('omw-1.4')

#import networkx and drawing functions for the drawing of the networks
import networkx as nx
import matplotlib.pyplot as plt
import networkx.drawing

#Normalization coefficient based on sketchy experiments for the weighting
normWeight = 1 / 0.35

#Parameters that can be changed

#Set which words we want to link all the texts on
investigatedWord = ""
investigatedWordList = [
#"causa", "corpus", "pars", "ratio", "moveo", "aqua", "locus", "tempus", "genus", "natura", 
#"ignis", "terra", 
#"radius", "deus", "species", "forma", "materia", "homo", "anima", "potentia", "electricitas"
#'cause',
#'nature'
"corp", "part", "chos", "natur", "dieu", "homm", "caus", "matier", "terr", "raison", "mouv", "air", "point", "soleil", "forc", "egal", "lign", "eau", "rayon", "vitess"
#"force", "body", "motion", "time", "water", "velocity", "part", "distance", "earth", "reason",
#"cause", "nature", "place", "particle", "matter", "specie", "surface", "light", 
#"line", "radius", "electricity", "sun", "method", "object", "experiment", "fire"
]


#Remove all wordtype tokes that occur less that the minWordFreq
minWordFreq = 2

#Parameters for PMI, how large the window is around each investigated word token and the minimal value for the PMI score
windowsize = 12
minimalPMI = 4

#Parameter for Cosine
minimalCosineSim = 0.21

#Parameter for Jaccard, the minimal value of the overlap between the two lists of collocates (1 ~ 100%)
minimalJaccardOverlap = 0.05

#Parameter for Directed Neighborhood, the minimal value of the overlap between the lists of collocates and the objects listlenght (1 ~ 100%)
minimalDirectedNeighOverlap = 0.2


#Initialize empty containers to be used for the algorithm

#Place where we store all filenames in order of appearance
allFileNames = []
#Place where we keep all the cleaned text-files
corpus = []
#Place where we keep our Latin stopwords
latinStopwords = []

#Dictionary where we keep information on which connections exist, construed as: {(TextName1, TextName2) : True/False}
textConnectionsForSoughtWord = {}

#List where we keep all the stopwords
latinStopwords = []

#PArameters for which functions should be executed
#Will we print the processed texts
PrintProcessedTexts = True
#Do we already have preprocessed texts we can simply read
LoadPreprocessedTexts = False
#What language are we checking?
language = "french"
#Do we want to use the directed algorithm?
directed = False
#Do we want the sparse vector cosines?
sparseVector = True
#Do we want a weighted Graph
weightedGraph = True

#Do we want to remove stopwords
stopwordRemoval = True

#Name of the directory that contains the corpus and will contain the results
direc = "\\texts\\"

pipelineName = ""

def main():
    
    if sparseVector:
        pipelineName = "vector"
    elif not sparseVector:
        pipelineName = "collocates"
    if not LoadPreprocessedTexts:
        #Load the stopwords
        latinStopwords = loadStopwords()

        #Read the corpus into memory: Iterate over all the files and for each file, clean the file up and save (and perhaps print out) the cleaned and lemmatized file
        for filename in os.listdir(os.getcwd() + direc):
            if not filename == "lem" and not filename == "ConnectivityResults":
                with open(os.path.join(os.getcwd() + direc + filename), 'r', encoding='utf8') as f:        
                    text = f.read()
                    text = cleanandSplit(text)
                    #text = cleanSplitAndLemmatizeText(text, latinStopwords)
                    corpus.append(text)
                    #Also, create a ordered list of the encountered text's titles
                    reworkedFilename = filename.split('.')[0]
                    namedWords = (filename.split('.')[0]).split('_')
                    justName = namedWords[0] + namedWords[1]
                    if justName in allFileNames:
                        print("volume case" + justName)
                        justName = ""
                        for word in namedWords:
                            justName += word
                    
                    allFileNames.append(justName)
            if PrintProcessedTexts: 
                with open(os.path.join(os.getcwd() + direc + 'lem\\', reworkedFilename) + "reworked.txt", 'w', encoding='utf8') as f2:
                    f2.write(text)
        #However, if we've already preprocessed the texts, load them in from the right directory directly
    elif LoadPreprocessedTexts:
        for filename in os.listdir(os.getcwd() + direc + "lem\\"):
            with open(os.path.join(os.getcwd() + direc + "lem\\" + filename), 'r', encoding='utf8') as f: 
                
                #####TEMPCODE!!!!!!!!!!!!#########
                text = f.read()
                #text = cleanandSplit(text)

                corpus.append(text)
                namedWords = (filename.split('.')[0].replace("lemmatized", "")).split('_')
                justName = namedWords[0] + namedWords[1]
                if justName in allFileNames:
                    print("volume case" + justName)
                    justName = ""
                    for word in namedWords:
                        justName += word
                
                allFileNames.append(justName)

    for word in investigatedWordList:
        investigatedWord = word
        
        #Initialize the Graph we will be building, add a node for every filename (work)
        if directed:
            Graph = nx.DiGraph()
        else:
            Graph = nx.Graph()
        for fN in allFileNames:
            Graph.add_node(fN)

        if not sparseVector:
            #After this we have stopworded and lemmatized texts ready for the next step of analysis in the list corpus, 
            #which contains very long strings that make up all the differen texts
            #For each of these texts, we start by deriving for a particular word the list of collocates according to some measure (PMI) with particular thresholds
            #Place where we keep lists of collocates associated with the investigated word and a particular text
            allCollocatesAllTexts = []
            for text in corpus:
                allCollocatesAllTexts.append(CalculatePMIReturnCollocates(text, investigatedWord))
            
            generateConnectionsGraph(False, directed, allCollocatesAllTexts, Graph)
        else:
            #Alternatively we create vectors for the words instead of lists of collocates.
            #They get a score for each other wordtype in the corpus. These vectors will then be located in the same vectorspace, and will be comparable
            allSemanticVectorsAllTexts = []

            #Get all wordtypes in an ordered list
            allWordtypesOrdered = getAllWordTypes(corpus)
            
            #Now, for each text, get a PPMI score between the researched word (for example 'deus') and all of the other wordtypes. 
            #This returns a vector for each text, instead of a list of collocates. Save as an ordered list of vectors. 
            for text in corpus:
                allSemanticVectorsAllTexts.append(CalculatePPMIReturnSparseVector(text, investigatedWord, allWordtypesOrdered))
            print("all vectors made moving to generating connections")
            generateConnectionsGraph(True, False, allSemanticVectorsAllTexts, Graph)    

    
    


        #Further initialize the Graph
    
        # if not weightedGraph:
        #     options = {
        #     'pos' : nx.spring_layout(Graph),
        #     'font_size' : 10,
        #     'font_color': 'r',
        #     'with_labels' : 'True',
        #     'node_color': 'black',
        #     'node_size': 100,
        #     'width': 0.1,
        #     'arrowstyle': '->',
        #     'alpha': 1,
        #     }
        # else:
        #     weights = [(Graph[tpl[0]][tpl[1]]['weight'] * normWeight) for tpl in Graph.edges()]
        #     normalized_weights = [(100*weight/sum(weights)) for weight in weights]
        #     options = {
        #     'pos' : nx.spring_layout(Graph),
        #     'font_size' : 10,
        #     'font_color': 'r',
        #     'with_labels' : 'True',
        #     'node_color': 'black',
        #     'node_size': 200,
        #     'width': normalized_weights,
        #     'arrowstyle': '->',
        #     'alpha': 1,
        #     }

        #plt.plot()
        #plt.figure(figsize = (100,100))
        #nx.draw(Graph, **options)
        #plt.show()

        #for index, collocates in enumerate(allCollocatesAllTexts):
            #with open(os.path.join(os.getcwd() + '\\FullCorpus\\CollocationalResults\\') + allFileNames[index] + " minPMI=" + str(minimalPMI) + " word=" + investigatedWord + ".txt", 'w', encoding='utf8') as f2:
                #f2.write('\n'.join(collocates))
            

    #   with open(os.path.join(os.getcwd() + '\\FullCorpus\\ConnectivityResults\\') + "connections" + " minPMI=" + str(minimalPMI) + " minJaccard=" + str(minimalJaccardOverlap) + " word=" + investigatedWord + ".txt", 'w', encoding='utf8') as f2:
        for index, connection in enumerate(textConnectionsForSoughtWord):
            if textConnectionsForSoughtWord[connection]:
                print(connection[0] + " connected to " + connection[1] + " With strength: " + str(textConnectionsForSoughtWord[connection] * normWeight) + "\n")
    #           f2.write(connection[0] + " connected to " + connection[1] + "\n")

        #Turn the text connections into rows per word
        listOfRows = []
        prevWord = ''
        currentrow = []
        for index, connection in enumerate(textConnectionsForSoughtWord):
            if prevWord == '':
                prevWord = connection[0]

            currWord = connection[0]
            

            if(currWord == prevWord):
                currentrow.append(int((textConnectionsForSoughtWord[connection] * normWeight) *1000))     
                prevWord = connection[0]

            if currWord != prevWord:
                #We have found a new word, previous is now a row
                currentrow.insert(0, prevWord)
                while len(currentrow) <= len(allFileNames):
                    currentrow.insert(1, '')
                    
                listOfRows.append(currentrow)
                currentrow = []
                currentrow.append(int((textConnectionsForSoughtWord[connection] * normWeight) *1000)) 
                prevWord = connection[0]
                if index == len(textConnectionsForSoughtWord) - 1:
                    currentrow.insert(0, prevWord)
                    while len(currentrow) <= len(allFileNames):
                        currentrow.insert(1, '')
                    listOfRows.append(currentrow)


        with open(os.path.join(os.getcwd() + direc + 'ConnectivityResults\\') + "aaaa language " + language + " Pipeline " + pipelineName + " " 
        + "minPMI=" + str(minimalPMI) + " windowSize=" + str(windowsize) + " word=" + investigatedWord + " minFreq=" + str(minWordFreq) +  ".csv", 'w', newline='', encoding='utf8') as csvFile:
            connectionWriter = csv.writer(csvFile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
            allFileNames.insert(0, '')
            connectionWriter.writerow(allFileNames)
            for row in listOfRows:
                connectionWriter.writerow(row)
            allFileNames.remove('')




#    print(nx.betweenness_centrality(Graph))



#All functions to be used by the script


def getAllWordTypes(corpusAsTexts):
        CorpusAsString = ""
    
        for text in corpus:
            CorpusAsString += text   

        wordtypes = set(CorpusAsString.split(' '))
        orderedWordtypes = []
        for word in wordtypes:
            orderedWordtypes.append(word)
        return orderedWordtypes    
        
def generateConnectionsGraph(sparseVectorMode: bool, directedMode: bool, PPMIVectorOrCollocatesAllTexts: list, graph: nx.Graph):
    for index1, elements1 in enumerate(PPMIVectorOrCollocatesAllTexts):
        for index2, elements2 in enumerate(PPMIVectorOrCollocatesAllTexts):
            #Disregard the question of self-connectedness
            if index1 is not index2:
                #If we are building a non-directed graph, iterate only once over every pair of texts
                if ("text: " + allFileNames[index2], "text: " + allFileNames[index1]) not in textConnectionsForSoughtWord and not directedMode:
                    #Now, look whether we are doing sparse vector cosine, or Jaccard collocates
                    if not weightedGraph:
                        if not sparseVectorMode: 
                            #Add a connection defined by two textnames, and calculate for those whether their lists of collocates are sufficiently similar according to the J.Index
                            textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )] = calculateJaccard(elements1, elements2)
                        else:
                            #Add a connection defined by two textnames, and calculate for those whether their semantic vectors are similar enough according to Cosine similarity
                            textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )] = calculateCoSineSimilarity(elements1, elements2)
                        #If this operation has returned True, add not only a connection value, but also now add an edge to the graph
                        if textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )]: 
                            graph.add_edge(allFileNames[index1], allFileNames[index2])
                    #If it is a weighted graph we generate an edge for each pair of nodes, with the same measures as weight now, instead of a boolean
                    elif weightedGraph:
                        if not sparseVectorMode: 
                            #Add a connection defined by two textnames, and calculate for those whether their lists of collocates are sufficiently similar according to the J.Index
                            textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )] = calculateJaccard(elements1, elements2, weightedGraph)
                            graph.add_edge(allFileNames[index1], allFileNames[index2], weight = textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )])  
                        else:
                            #Add a connection defined by two textnames, and calculate for those whether their semantic vectors are similar enough according to Cosine similarity
                            textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )] = calculateCoSineSimilarity(elements1, elements2, weightedGraph)
                            graph.add_edge(allFileNames[index1], allFileNames[index2], weight = textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )])
                elif directed:
                    if not weightedGraph:
                        #Add a connection defined by two textnames, and calculate for those whether the first list of collocates is sufficiently similar to the second according to the directed neighborhood
                        textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )] = calculateDirectedNeighborhood(elements1, elements2)
                        #If this operation has returned True, add not only a connection value, but also now add an edge to the graph
                        if textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )]: 
                            graph.add_edge(allFileNames[index1], allFileNames[index2])
                    elif weightedGraph:
                        #Add a connection defined by two textnames, and calculate for those whether the first list of collocates is sufficiently similar to the second according to the directed neighborhood
                        textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )] = calculateDirectedNeighborhood(elements1, elements2, weightedGraph)
                        graph.add_edge(allFileNames[index1], allFileNames[index2], weight = textConnectionsForSoughtWord[("text: " + allFileNames[index1], "text: " + allFileNames[index2] )])


#Function that takes two sets of collocates and 1) checks their Jaccard index and 2) returns whether it meets the threshold set
def calculateJaccard(collocates1, collocates2, weight = False):
    
    #Cast to sets (removing all duplicates and  'unordering' the list)
    collocates1 = set(collocates1)
    collocates2 = set(collocates2)

    #Use the set operations union and intersection which define the Jaccard to calculate Jaccard
    unionSet = collocates1.union(collocates2)   
    intersectionSet = collocates1.intersection(collocates2)
    if len(unionSet) == 0:
        print("no collocates")
        JaccardScore = 0
    else:
        JaccardScore = len(intersectionSet) / len(unionSet)
    if weight:
        return JaccardScore
    else:    
        #check score against minimal threshold
        return JaccardScore >= minimalJaccardOverlap

def calculateCoSineSimilarity(semVector1, semVector2, weight = False):
    Vec1 = numpy.array(semVector1)
    Vec2 = numpy.array(semVector2)


    if numpy.linalg.norm(Vec1) == 0 or numpy.linalg.norm(Vec2) == 0:
        return 0

    coSineSim = (numpy.dot(Vec1, Vec2)) / (numpy.linalg.norm(Vec1) * numpy.linalg.norm(Vec2))
    if weight:
        return coSineSim
    else:
        return coSineSim > minimalCosineSim

#Calculate whether the first argument is similar enough to the second to be connected
#The same needs to be done for the second argument as first argument at some point (directed, so not symmetrical)
def calculateDirectedNeighborhood(collocates1, collocates2, weight = False):
    #Cast to sets (removing all duplicates and  'unordering' the list)
    collocates1 = set(collocates1)
    collocates2 = set(collocates2)
    
    #Use the set operation intersection which define the directed neighborhood
    otherSetSize = len(collocates2)
    intersectionSet = collocates1.intersection(collocates2)
    directedNeighborhoodScore = len(intersectionSet) / otherSetSize
    if otherSetSize > 5:
        if weight:
            return directedNeighborhoodScore
        else:
            return directedNeighborhoodScore >= minimalDirectedNeighOverlap
    else:
        return False
#Function for cleaning up a text and then lemmatizing it
def cleanSplitAndLemmatizeText(text: str, latinStopwords: list) -> str:
    #Remove all capital's, remove all interpunction, remove all numbers, remove all one letter words
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('','', string.digits))
    words = text.split(' ')
    words = set(words)
    if stopwordRemoval:
        for word in words:
            if len(word) == 1:
                text = text.replace(' ' + word + ' ', ' @@@ ')
    #remove all odd ligiture
    text = text.replace('ſ', 's')

    #Split up the text in words and remove all the words that are also in the stopwordlist
    text = text.split(' ')
    
    if language == 'latin':
        if stopwordRemoval:
            textContainer = []
            for word in text:
                if word not in latinStopwords:
                    textContainer.append(word)
                else:
                    textContainer.append('@@@')
            text = textContainer

    textStrings = []
    lemText = []

    if language == 'latin':
        #Use the CLTK lemmatizer (latin only) to lemmatize the text and use the map function to extract the new text
        lemText = lemmatizer.lemmatize(text)
        textStrings = map(lambda x: x[1], lemText)
    elif language == 'english':
        #use english wordnet lemmatizer and 
        for word in text:
            if word is not '': 
                lemText.append(lemmatizerE.lemmatize(word, get_wordnet_pos(word)))
        textStrings = lemText
    else:
        print("no valid language found")
    
    print(textStrings)

    if language == 'latin':
        #Remove the stopwords again (perhaps some words are turned into stopwords)
        if stopwordRemoval:
            textStringsContainer = []
            for textString in textStrings:
                if textStrings not in latinStopwords:
                    textStringsContainer.append(textString)
                else:
                    textStringsContainer.append('@@@')

            textStrings = textStringsContainer

    #Now, use a counter object to check whether the frequency of all words meets the minimal threshold, if not, they are removed
    freqOfAllWords = Counter(textStrings)

    anotherTextContainer = []
    for textString in textStrings:
        if freqOfAllWords[textString] > minWordFreq:
            anotherTextContainer.append(textString)
        else:
            anotherTextContainer.append('@@@')

    textStrings = anotherTextContainer

    #Return the text as a whole and not a list of words
    text = ' '.join(textStrings)
    return text

def cleanandSplit(text:str):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('','', string.digits))
    words = text.split(' ')
    words = set(words)
    if stopwordRemoval:
        for word in words:
            if len(word) == 1:
                text = text.replace(' ' + word + ' ', ' @@@ ')
    #remove all odd ligiture
    text = text.replace('ſ', 's')

    #Split up the text in words and remove all the words that are also in the stopwordlist
    text = text.split(' ')

    #Now, use a counter object to check whether the frequency of all words meets the minimal threshold, if not, they are removed
    freqOfAllWords = Counter(text)

    anotherTextContainer = []
    for textString in text:
        if freqOfAllWords[textString] > minWordFreq:
            anotherTextContainer.append(textString)
        else:
            anotherTextContainer.append('@@@')

    textStrings = anotherTextContainer

    #Return the text as a whole and not a list of words
    text = ' '.join(textStrings)
    return text


#Function for the loading of the stopwordlist (for latin only)
#Right now, the path is hardcoded to the same directory as where the script is, for a file called LatinStopwords.txt
def loadStopwords() -> list:
    with open(os.path.join(os.getcwd(), "LatinStopwords.txt"), 'r', encoding='utf8') as f:
        #Read the list of stopwords and split them by hard enters
        listOfPureWords = []
        #allwords = f.read()
        #splitwords = allwords.split('\n')
        #structure of the file is: |stopword.append("word")| so we split on the |"| sign, and take the second element of the resulting list, which contain the word
        #for compoundWord in splitwords: listOfPureWords.append(compoundWord.split("\"")[1])
        return listOfPureWords


def CalculatePPMIReturnSparseVector(text: str, soughtWord: str, allWordTypesCorpus: list) -> list:
    #Generate an empty dictionary that will first hold the following information {Wordtype: #ofTimesFoundAroundSoughtWord}
    freqDict = {}
    #Generate an empty list where we will save all the PPMI scores, for all words in the corpus
    PPMIVector = []

    #Split the text up in words and create a counter object for the frequency of all wordtypes in the text (by exclusion, all the others in the corpus have freq 0, and PPMI 0)
    text = text.split(' ')
    freqOfAllWordsText = Counter(text)

    #Generate a list of all indices of the wordtype's tokens we are investigating (where in the text is our soughtword)
    indices = [i for i, x in enumerate(text) if x == soughtWord]

    #Now, we iterate over all of the indices, and see what words we encounter next to each of them within the chosen windowsize
    for index in indices:
        for x in range(windowsize *-1, windowsize + 1):
            #Check whether we remain in the bounds of our text
            if(len(text) > index + x and index + x >= 0):
                foundWord = text[index + x]
                #Now, we add the found word to the dictionary, either it is new, in which case we've observed it once now, or it is not, then we add one to the frequency
                if foundWord in freqDict:
                    freqDict[foundWord] += 1
                elif foundWord not in freqDict:
                    freqDict[foundWord] = 1

    #Now we must generate our sparse vector for ALL wordtypes (not just those in the text).
    #To keep the order we iterate over all wordtypes in the corpus via its list.
    #Either the wordtype is contained in the text, in which case we calculate PPMI based on our findings in the text.
    #If not contained in the text, the number of times they are found next to each other is 0, in such a case, we will never get a positive PMI.
    #So, provide 0 as score for our soughtword in this text.
    
    #Now we extract all the values needed for the PMI calculation
    #1: How often does the word we are investigating occur?
    freqSoughtword = freqOfAllWordsText[soughtWord]
    #2: From that, derive how many locations have been investigated by multiplying the number of occurences with the number of investigations that take place per occurence
    noCheckedPlaces = freqSoughtword * windowsize * 2 - 1
    #3: The total number of words in the text
    totalWords = len(text)

    #Itereer over alle woorden en bepaal het aantal keer dat een woord is voorgekomen naast het woord dat we bekijken
    #Als dat nul is, dan geven we PMI 0, anders gaan we PMI berekenen, op basis van de bovenstaande info
    for word in allWordTypesCorpus:
        if word in freqDict: ##if word in text:
           
            #4: For every other wordtype that has at least one occurence, extract how often it has occured within the windows around the investigated word
            #and how often this word occurs in the text at all. This together gives the PMI value for the pair investigated word - some word with at least one cooccurence within the windows around the investigated word
            #if word in freqDict:
            PMI = math.log2((freqDict[word] / noCheckedPlaces) / (freqOfAllWordsText[word] / totalWords))
            if PMI > 0:
                PPMIVector.append(PMI)
            else:
                PPMIVector.append(0)
            ##else:
            ##    PPMIVector.append(0)        
        else:
            PPMIVector.append(0)
    print("Vector made")
    return PPMIVector

#Function that calculates the PMI-values given a specific word under investigation
#It returns a list of collocates that have passed the minimal PMI values for this word
def CalculatePMIReturnCollocates(text: str, soughtWord: str):
    #Generate an empty dictionary that will first hold the following information {Wordtype: #ofTimesFoundAroundSoughtWord}
    freqDict = {}
    #Generate an empty list where we will save all the words that score high enough to meet the PMI threshold
    collocates = []

    #Split the text up in words and create a counter object for the frequency of all wordtypes in the text
    text = text.split(' ')
    freqOfAllWords = Counter(text)
    
    
    #Generate a list of all indices of the wordtype's tokens we are investigating (where in the text is our soughtword)
    indices = [i for i, x in enumerate(text) if x == soughtWord]
    
    #Now, we iterate over all of the indices, and see what words we encounter next to each of them within the chosen windowsize
    for index in indices:
        for x in range(windowsize *-1, windowsize + 1):
            #Check whether we remain in the bounds of our text
            if(len(text) > index + x and index + x >= 0):
                foundWord = text[index + x]
                #Now, we add the found word to the dictionary, either it is new, in which case we've observed it once now, or it is not, then we add one to the frequency
                if foundWord in freqDict:
                    freqDict[foundWord] += 1
                elif foundWord not in freqDict:
                    freqDict[foundWord] = 1
    
    #Now we extract all the values needed for the PMI calculation
    #1: How often does the word we are investigating occur?
    freqSoughtword = freqOfAllWords[soughtWord]
    #2: From that, derive how many locations have been investigated by multiplying the number of occurences with the number of investigations that take place per occurence
    noCheckedPlaces = freqSoughtword * windowsize * 2 - 1
    #3: The total number of words in the text
    totalWords = len(text)
    #4: For every other wordtype that has at least one occurence, extract how often it has occured within the windows around the investigated word
    #and how often this word occurs in the text at all. This together gives the PMI value for the pair investigated word - some word with at least one cooccurence within the windows around the investigated word
    for wordType in freqDict:
        PMI = (freqDict[wordType] / noCheckedPlaces) / (freqOfAllWords[wordType] / totalWords)
        #If the score is high enough, add it to the list of results
        if  PMI >= minimalPMI:
            collocates.append(wordType)
    return collocates        


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

if __name__ == "__main__":
    main()
        
