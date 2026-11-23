# Commandes du projet pour Windows (équivalent du Makefile, sans make).
# Usage : .\tasks.ps1 <tâche>      ex. .\tasks.ps1 run     .\tasks.ps1 retrain
param([string]$Task = "help")

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Set-Location $PSScriptRoot

$tasks = [ordered]@{
    "install"     = { & $py -m pip install -e ".[dev]" }
    "run"         = { & $py -m streamlit run frontend/streamlit_app.py }
    "test"        = { & $py -m pytest -q }
    "lint"        = { & $py -m ruff check backend frontend ml tests }
    "qevd-build"  = { & $py ml/data/qevd_split_zip.py build data data/raw/qevd_fit300k }
    "qevd-extract"= { & $py ml/data/extract_qevd.py --workers 10 }
    "reps"        = { & $py ml/data/extract_rep_summaries.py }
    "retrain"     = {
        # Nouveaux modèles dans data/models/candidates/ : à comparer avant de remplacer data/models/.
        & $py ml/data/extract_rep_summaries.py
        if ($?) { & $py ml/training/train_classifier.py --out data/models/candidates/exercise_lstm.pt }
        if ($?) { & $py ml/training/train_form_model.py --out data/models/candidates/form_model.joblib }
    }
    "use-candidates" = {
        Copy-Item data/models/candidates/exercise_lstm.pt data/models/exercise_lstm.pt
        Copy-Item data/models/candidates/form_model.joblib data/models/form_model.joblib
        Write-Host "The app now uses the retrained models. Undo: git checkout -- data/models/"
    }
    "live"        = { & $py frontend/desktop/live_rep_counter.py --exercise squat }
}

if ($tasks.Contains($Task)) { & $tasks[$Task] }
else {
    Write-Host "Tasks:"; $tasks.Keys | ForEach-Object { Write-Host "  .\tasks.ps1 $_" }
}
