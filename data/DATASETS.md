# Datasets open source (téléchargeables sans compte)

Les données vont dans `data/raw/` (ignoré par Git). Le script de téléchargement d'origine a été perdu
(jamais enregistré dans Git) ; état actuel et liens manuels : `DATASETS_STATUS.md`.

| Dataset | Pourquoi pour nous | Contenu | Licence |
|---|---|---|---|
| **Fitness exercises, good/bad form, multi-view** — [Mendeley Data](https://data.mendeley.com/datasets/kgbb3yn47p/2) | Le plus utile : **squat, pompe, curl** + étiquette **bonne / mauvaise forme** → entraîne le modèle de posture et le LSTM | Vidéos smartphone en salle, 26 personnes, 7 exercices, vues face / côté / diagonale | CC BY 4.0 (citer les auteurs) |
| **Penn Action** — [site](https://dreamdragon.github.io/PennAction) | **squats, push_ups** + autres actions (utile pour la classe « idle »), avec **13 points du corps annotés** pour vérifier nos angles | 2 326 séquences en images JPEG + fichiers `.mat` | Usage recherche, citer Zhang et al., ICCV 2013 |

Optionnel (images, pas de vidéos) : [averrous/workout sur Hugging Face](https://huggingface.co/datasets/averrous/workout)
— 22 exercices, ~13 850 images, ~864 Mo, licence non indiquée.

Dans tous les cas, nos **propres enregistrements** (`scripts/record_clip.py`) restent nécessaires pour
tester sur des personnes et des caméras « réelles ».

## Après téléchargement
```
data/raw/fitness_form/   vidéos (nom = sujet_exercice_forme_vue)
data/raw/penn_action/    frames/ labels/ tools/
```
→ `python training/extract_keypoints.py` crée les Parquet dans `data/processed/`.
