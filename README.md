# Data_Science_Challenge_2

ENGIE Deep Learning Challenge - Rooftop orientation prediction 

The energy transition can be summarized by the 4 Ds, four areas that redefine the world of energy: Decarbonisation, Digitalisation, Decentralisation and Decrease in demand. 

Decentralisation means that our relationship with energy needs to be reconsidered. Decentralised production methods such as solar installations will enable renewable energy to be generated locally.

As the price of installing solar has gotten less expensive, more homeowners are turning to it as a possible option for decreasing their energy bill.

Before installing solar panels on your roof, you need to know if you're making a good investment, which means knowing how well your new system will perform, and if it's going to save you money on power bills over the long term.

The challenge is to develop an online application that estimates accurately the potential for electricity generation from PV on buildings or houses in Europe by computing roof orientation and surface, weather conditions and client consumption profile.

As a first step, the goal of this data science challenge is to correctly predict the orientation of a roof from a given satellite image

The data:

The data consists of 2 datasets:

   -  a collection of images of 14,104 rooftops, in various shapes and sizes
   -  the result of a manual tagging campaign of the rooftops, whose orientation was determined by visual inspection. Approx. 70% of the rooftops are present in this dataset, and you are asked to infer the orientation of the remaining 30%
   
Tagging campaign:

The results of the tagging campaign are stored in a comma-separated file, `train.csv`, containing 2 columns:

    `id`, which is the numerical ID of the rooftop. It is the same ID as in the images archive
    `orientation`, which is the orientation of the rooftop as determined by visual inspection
        `1` = the roof appears north-south oriented
        `2` = the roof appears west-east oriented
        `3` = the roof appears flat
        `4` = the orientation could not be determined by visual inspection

The column names are recalled as headers in the CSV file.
