from dataclasses import replace
import os
import string
import numpy
import math
import csv
from collections import Counter

Contemporary = 2
RejectedStart = 30
RejectedEnd = 30
maximalDistance = 100
direc = "C:\\Users\\hugob\\Desktop\\Chapter on the canon\\LatinReworkedResults\\"
container = []
allResults = []

def getFirstFourElement(inp):
    return inp[0:4]

#Reducing some names with alternate spelling
def duplicatenameCheck(input):
    
    if input == "legra":
        input = "grand"
    if input == "debru":
        input = "bruyn"
    if input == "clerc":
        return "clerc"
    if input == "lecle":
        return "clerc"   
    return input


#Iterate over each set of word based results
for fileNumber, filename in enumerate(os.listdir( direc)):
    input = numpy.loadtxt(open(direc + filename, "rb"), delimiter=",", skiprows=0 ,dtype=str, ndmin=0)
    investigatedWord = filename.split(" ")[6].split("=")[1]

    #Create correct makeup on first word being read
    if fileNumber == 0:
        firstRow = input[0][1+RejectedStart:-RejectedEnd].tolist()
        allResults.append(firstRow)
        allResults[0].insert(0,"X")
        allYears = map(getFirstFourElement, firstRow )
        allResults.append(list(allYears))
        allResults[1][0] = "Year"

    #Create location for all the info of this set of results    
    allResults.append([])
    allResults[-1].append(investigatedWord)

    #Remove incorrect self scores
    for i, row in enumerate(input):
        for j, entry in enumerate(row):
            if entry == "2857":
                input[i,j] = ""

    #Begin and end definition
    counter = 1 + RejectedStart
    max = len(input) - RejectedEnd

    #Iterate over each row
    while counter < max:
        #Get out name of author
        name = input[0,counter]
        shortHand = name[4:9]
        shortHand = shortHand.lower()
        shorthand = duplicatenameCheck(shortHand)

        #Get out year of pub
        contempYear = float(name[0:4])
        

        #Get all prev and fut info
        #Get all info row
        investigatedRow = input[counter]
        investigatedColumn = []
        #Get relevant column info
        for row in input:
            investigatedColumn.append(row[counter])


        totalRow = 0
        valuesRow = 0
        #Check whether a certain value is acceptable into the res
        for index, element in enumerate(investigatedRow):
            if element.isdigit():
                checkedYear = float(input[0, index][0:4])
                checkedShortHand = input[0, index][4:9].lower()
                checkedShortHand = duplicatenameCheck(checkedShortHand)
                #if the value is from far enough in the future (and not contemporaneuos)
                if abs(contempYear - checkedYear) > Contemporary:
                    #if the value is not too far away
                    if abs(contempYear - checkedYear) < maximalDistance:
                        #if it is not a score of two works from the same author
                        if shortHand != checkedShortHand:
                            #if it is not a 0
                            if (float(element)) != 0:
                                totalRow += float(element)
                                valuesRow += 1
                            else:
                                pass#print(name + " is scored a 0 with " + input[0,index])    
                        else:
                            pass#print(name + " is the same author as " + input[0,index] + " row" + " which equates to " + str(index -30) + " and "  + str(counter -30))
                    else:
                        pass#print(str(checkedYear) + " was too far from with " + name)
                else:
                    pass#print(str(checkedYear) + " was contemporary with " + name )
            else:
                pass#print("Not a vlaue")
    
        #Calculate the average similarity (LOOK AT IT)
        if valuesRow > 0:
            rowAverageSim = totalRow / valuesRow
        else:
            rowAverageSim = 0

        #Check the same as for the row
        totalColumn = 0
        valuesColumn = 0
        #Check whether a certain value is acceptable into the res
        for index, element in enumerate(investigatedColumn):            
            if element.isdigit():
                checkedYear = float(input[0, index][0:4])
                checkedShortHand = input[0, index][4:9].lower()
                checkedShortHand = duplicatenameCheck(checkedShortHand)
                #if the value is from far enough in the past (and not contemporaneuos)
                if abs(contempYear - checkedYear) > Contemporary:
                    #if the value is not too far away
                    if abs(contempYear - checkedYear) < maximalDistance:
                        #if it is not a score of two works from the same author
                        if shortHand != checkedShortHand:
                            #if it is not a score of 0
                            if (float(element)) != 0:
                                totalColumn += float(element)
                                valuesColumn += 1
                            else:
                                pass#print(name + " is scored a 0 with " + input[0,index])    
                        else:
                            pass#print(name + " is the same author as " + input[0,index] + " column" + " which equates to " + str(index - 30) + " and "  + str(counter - 30))    
                    else:
                        pass#print(str(checkedYear) + " was too far from with " + name)
                else:
                    pass#print(str(checkedYear) + " was contemporary with " + name )
            else:
                pass#print("Not a value")
    
        #Calculate the average similarity (LOOK AT IT)
        if valuesColumn > 0:
            columnAverageSim = totalColumn / valuesColumn
        else:
            columnAverageSim = 0

        #If either had only 0's or other problems, then either future or past is undefined
        if rowAverageSim == 0 or  columnAverageSim == 0:
            #print(str(int(contempYear)) + "," + name + ", does not have a defined innovativity")
            allResults[-1].append("Undefined")
        #Else, we calculate the innovativity of the work for this word
        else:
            if rowAverageSim/columnAverageSim == 1:
                innovativity = "NOT DEFINED"
            elif rowAverageSim/columnAverageSim < 1:
                innovativity = (-1 * columnAverageSim/rowAverageSim)
            else:
                innovativity = (rowAverageSim/columnAverageSim)
            
            #print(str(int(contempYear)), "," + name + ", has an innovativity of, " + str(innovativity))
            allResults[-1].append(innovativity)

        counter += 1


with open(os.path.join( direc + 'InnovativityRes') +  ".csv", 'w', newline='', encoding='utf8') as csvFile:
    connectionWriter = csv.writer(csvFile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    for row in allResults:
        connectionWriter.writerow(row)














