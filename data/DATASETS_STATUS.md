# État des datasets (contrôle du 3 novembre 2026, ~02h20)

## ✅ Fitness exercises, good/bad form (Mendeley) — COMPLET
`data/raw/fitness_form/` : **1 042 vidéos, 2,4 Go**, aucun fichier vide, aucun échec.

| Exercice | good | bad |
|---|---|---|
| squat | 75 | 75 |
| push_up | 72 | 72 |
| bicep_curl | 74 (+1 nommé « bicep ») | 75 |
| shoulder (press) | 75 | 75 |
| tricep | 77 | 77 |
| back (row) | 72 | 72 |
| abs (side bend) | 75 | 75 |

Noms : `subject_XXX_<exercice>_<good|bad>_<front|side|diagonal>.mp4` (26 sujets).

## ⏳ Penn Action — À RELANCER
Échec : le serveur de l'UPenn n'envoie pas tout son certificat HTTPS et Python le refuse
(`CERTIFICATE_VERIFY_FAILED`). Le script utilise maintenant `curl.exe` de Windows, qui gère ce cas
(la vérification de sécurité reste active).
→ Double-cliquer **`telecharger_penn_action.bat`** (à la racine du projet).
