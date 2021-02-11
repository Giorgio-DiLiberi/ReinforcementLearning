# "QuadEnv2" 
Questa cartella contiene la terza versione dell'environment quadricottero. 
Le modifiche rispetto alla prima versione sono riportate anche sul quaderno:
-implementazione delle eqauazioni del moto tramite vettori e matrici

-nel file di simulazione è stata migliorata la sezione per il plot 

-La cartella chiamata log contiene i file generati con tensorboard da leggere tramite comando
>>>>> tensorboard --logdir ./<nome cartella logs>/

-il file .zip contiene la policy allenata da caricare con il file simulator.py

-Per usare questa cartella è possibile:
    # allenare una policy sull environment di quadricottero tramite il codice Training.py

    # Testare una policy preallenata su Quadenv2 tramite il codice simulator.py
    
    # Testare delle azioni scelte dall'utente (non selezionate da una rete neurale) per 	   	osservare e validare il comportamento del modello implementato in Quadenv2 tramite il codice actionTest.py
