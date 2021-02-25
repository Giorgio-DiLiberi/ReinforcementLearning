# "QuadEnv2" 
Questa cartella contiene la quarta versione dell'environment quadricottero. 
Le modifiche rispetto alla precedente versione sono riportate anche sul quaderno e consistono nel fatto che in questo environment Il modello del motore è piu accurato come riportato sul notebook

-La cartella chiamata log contiene i file generati con tensorboard da leggere tramite comando
>>>>> $ tensorboard --logdir ./tensorboardLogs/ (Se la cartella con i logs si chiama tensorboardLogs)

-la cartella Policies contiene le policy allenate da caricare con il file simulator.py

-Per usare questa cartella è possibile:
    # allenare una policy sull environment di quadricottero tramite il codice Training.py

    # Testare una policy preallenata su Quadenv2 tramite il codice simulator.py
    
    # Testare delle azioni scelte dall'utente (non selezionate da una rete neurale) per 	   	osservare e validare il comportamento del modello implementato in Quadenv2 tramite il codice actionTest.py
    
-La cartella Simulation results contiene i risultati di simulazioni fatte con actionTest o con il simulatore di policy allenate.
    
-la cartella "EvalClbkLogs" contiene la best policy salvata con call backs di valutazione fatta durante l'allenamento
