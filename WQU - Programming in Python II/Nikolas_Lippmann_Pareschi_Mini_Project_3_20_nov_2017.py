# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:27:58 2017

@author: Nikolas
"""

# 1. Check out the nutrient-db python utility from GitHub from
# https://github.com/schirinos/nutrient-db.git

# 2.Run the main program with python nutrientdb.py -e > nutrients.json
# to convert the USDA data to JSON format. For further details, check
# https://github.com/schirinos/nutrient-db.You might have to install the
# python utility for MongoDB interface via pip install pymongo

# 3. Load the JSON dataset into Pandas dataframe
# using the built in python
# c json class. Extract values of the following fields in to the
# dataframe - food names, group, id, and manufacturer

# 4. For the ‘Amino Acids’ nutrient group output a table showing the
# different constituents of the group (Alanine, Glycine, Histidine etc)
# and the foods in which they are present (Gelatins, dry powder,
# beluga, meat...etc)

# 5. For all the different nutrient group
# (beef Products, Pork Products, dairy and egg products etc.)
# calculate the median Zinc content
# (median of the zinc
# content in all the foods that constitute the nutrient group)

# 6. Plot the distribution of median Zinc Content for different nutrient
# groups as a bar chart.


import pandas as pd
import matplotlib.pyplot as plt

# READING THE JSON FILE

df2 = pd.read_json("./nutrients.json", lines=True)

df3 = df2[['group', 'manufacturer', 'name', 'nutrients']]


amino = ["alanine", "arginine", "asparagine", "aspartic acid", "cysteine",
         "glutamine", "glutamic acid", "glycine", "histidine", "isoleucine",
         "leucine", "lysine", "methionine", "phenylalanine", "proline",
         "serine", "threonine", "tryptophan", "tyrosine", "valine"]

##############################################################################
#
#                           AMINO FOOD
#
##############################################################################


def aminofood(df2, amino):

    ntrStr2 = []
    fudidas = []
    allfood = []
    aminofood = []

    for i in range(len(df2)):
        foodStr = df2.iloc[i]['name'][u'long'].encode('utf-8').lower().strip()
        allfood.append(foodStr)
        for ntr in df2.iloc[i]['nutrients']:
            ntrStr = ntr[u'name'].encode('utf-8').lower().strip()
            fudidas.append(ntrStr)
            if ntrStr in amino:
                ntrStr2.append(ntrStr)
                aminofood.append(df2.iloc[i]['name'][u'long'].encode('utf-8').
                                 lower().strip())

    presentamino = set(ntrStr2)
    print'Aminoacids in food \n'
    print presentamino
    xxx2 = set(aminofood)
    print xxx2
    xxx3 = list(xxx2)
    xxx4 = pd.DataFrame({'Foods with aminoacids': xxx3})
    print xxx4

    xxx5 = xxx3[0:20]
    print xxx5

##############################################################################
#
#                               ZINC FOOD
#
##############################################################################


def zincfood(df2):

    zinc = {}
    for i in range(len(df2)):
        for ntr in df2.iloc[i]['nutrients']:
                zincQty = float(ntr[u'value'])
                foodGrp = df2.iloc[i]['group'].encode('utf-8').lower().strip()

                ntrStr = ntr[u'name'].encode('utf-8').lower().strip()

                if ntrStr == "zinc, zn":
                    if foodGrp in zinc:
                        zinc[foodGrp].append(zincQty)
                    else:
                        zinc[foodGrp] = [zincQty]
    for ntr in df2.iloc[i]['nutrients']:
                ntrStr = ntr[u'name'].encode('utf-8').lower().strip()
                if ntrStr == "zinc, zn":
                    if foodGrp in zinc:
                        zinc[foodGrp].append(zincQty)
                    else:
                        zinc[foodGrp] = [zincQty]

    dictionary = {}

    for f in zinc:

        median = int(len(zinc[f]) / 2)
        zinc[f].sort()
        dictionary[f] = zinc[f][median]
        print dictionary[f]

    print dictionary
    plt.bar(range(len(dictionary)), dictionary.values(), align='center')
    plt.xticks(range(len(dictionary)), dictionary.keys(), rotation=90)
    plt.title('Median Zinc Content per USDA food Group')

    plt.show()


def main():

    aminofood(df2, amino)
    zincfood(df2)


if __name__ == '__main__':
    main()
