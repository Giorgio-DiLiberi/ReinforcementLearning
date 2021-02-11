# "QuadEnv2" 
Questa cartella contiene la seconda versione dell'environment quadricottero. 
Le modifiche rispetto alla prima versione sono riportate anche sul quaderno:
-normalizzazione di action_space e obs_space tra -1 e 1 per ogni valore;
-implementazione del metodo di runge kutta tamite vettori
-trasformazione del self.state in np.array

-nel file di simulazione è stata aggiunta una sezione per il plot con mathplotlib

-La cartella chiamata log contiene i file generati con tensorboard da leggere tramite comando
>>>>> tensorboard --logdir ./<nome cartella logs>/

-il file .zip contiene la policy allenata da caricare con il file simulator.py

-Per usare questa cartella è possibile:
    # allenare una policy sull environment di quadricottero tramite il codice Training.py

    # Testare una policy preallenata su Quadenv2 tramite il codice simulator.py
    
    # Testare delle azioni scelte dall'utente (non selezionate da una rete neurale) per osservare e validare il comportamento del modello implementato in Quadenv2 tramite il codice actionTest.py
