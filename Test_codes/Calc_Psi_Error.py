# codice esplicativo su come viene calcolato l'errore su Psi

PHI = quat2Att()  # La funzione restituisce un array contenente gli angoli di Eulero [Phi, Theta, Psi]

Psi_ref = np.arctan2(V_NED_ref[1], V_NED_ref[0]) # = atan2(V_Est_ref / V_Nord_ref)
Psi_err = Psi_ref - PHI[2]

# Controllo per dare un errore di sengo tale per cui venga sempre effettuata la virata nel verso più conveniente
if Psi_err>=np.pi:
Psi_err = Psi_err - (2 * np.pi)

elif Psi_err<-np.pi:
Psi_err = Psi_err + (2 * np.pi)