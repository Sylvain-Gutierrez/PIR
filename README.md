# PIR
repository for PIR (ISAE-SUPAERO 2nd year project), subjetc : real neural data analysis
PIR students: Sylvain Gutierrez and Louise Placidet

This README.txt will structure and describe the "useful" documents:

- In Folder: #AdabandFlt#
        
    - #AdaBandFlt_Depolarisation_29_04_20# :
               
       This file contains the final updated AdaBandFlt code which aligns the spikes 
       based on the differents options (1er dépassement, central zero, dépassement 
       positif...)
       
       
    - #AdaBandFlt.py#
        
        This file contains the code of the AdaBandFlt_Depolarisation, which can 
        therefore be imported into PCA and KMEANS notebooks and lighten the code
       
- In Folder: #Affichage_et_Debruitage#
        
    - #Bilan_Debruitage_06_04_20# :
               
       This file contains the code to display the original signal as well as the 
       signal after bandpass and after the wiener filter, as well as their respective 
       periodograms.

- In Folder: #Comparaison_Logiciel#
        
    - #Comparaison_detection_sophie# :
               
       This file is able to open .csv files from Sophie's software and displays the 
       centers of the spikes detected by the software.  It then uses our record_spikes 
       function, to use the information of those centers and recored the spike in the 
       given data.  It is then able to plot all of those spikes aligned at the same 
       point.
       
       
- In Folder: #KMEANS#
        
    - #First_KMEANS_05_05_20# :
               
       This file contains the code which does a KMEANS to the data received just after 
       a PCA and plots the result.
       
- In Folder: #PCA#
        
    - #Bilan_PCA_29_04_20# :
               
       This file contains the code which applies a PCA from the library scikit-learn 
       to the data received and plots the result.