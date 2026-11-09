# État des datasets (mis à jour le 9 novembre 2026)

## ✅ Fitness exercises, good/bad form (Mendeley) : installé
`data/raw/fitness_form/` : **1 042 fichiers, 2,4 Go**, 26 sujets, 7 exercices, 3 vues.
Noms : `subject_XXX_<exercice>_<good|bad>_<front|side|diagonal>.mp4`.

- **1 fichier corrompu** : `subject_024_back_good_diagonal.mp4` (404 octets ; le serveur a renvoyé
  « Service Unavailable » au lieu de la vidéo). Exclu. Pour le récupérer : le supprimer et le
  re-télécharger depuis Mendeley.
- Résolutions variées : 717 vidéos en 810×1440, les autres de 348×624 à 1620×2880 (quelques-unes en paysage).

## ✅ Extraction
`data/processed/all.parquet` : 445 208 frames, 1 041 vidéos, pose détectée sur 99,9 % des frames.
Qualité par clip : `data/keypoint_previews/quality_report.csv` (`make previews`).
Vues peu fiables : pompe de face (98 % de clips douteux), curl de face (52 %).

## ⏸ Penn Action : non installé (optionnel)
Échecs de téléchargement (certificat HTTPS du serveur UPenn, puis réseau).
Téléchargement manuel : https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz → `data/raw/penn_action/`.
