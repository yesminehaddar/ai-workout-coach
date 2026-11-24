# Commandes du projet pour Windows (équivalent du Makefile, sans make).
# Usage : .\tasks.ps1 <tâche> [arguments]
#   .\tasks.ps1                  liste des tâches
#   .\tasks.ps1 run              application web (chatbot, coach vidéo, rapports)
#   .\tasks.ps1 live pushup      démo webcam OpenCV (squat | pushup | curl, puis index de caméra ou vidéo)
# Les textes affichés restent en ASCII : PowerShell 5.1 lit ce fichier sans BOM comme ANSI.
param(
    [string]$Task = "help",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Set-Location $PSScriptRoot

function Get-TaskArg([int]$i, [string]$default) { if ($Rest.Count -gt $i) { $Rest[$i] } else { $default } }

# Chaque tâche : groupe (pour l'aide), description, commande.
$tasks = [ordered]@{
    # ---------------------------------------------------------------- installation
    "install" = @{ group = "Setup"; desc = "Install the dependencies and the coach package (editable)"
        run = { & $py -m pip install -e ".[dev]" } }
    "env" = @{ group = "Setup"; desc = "Create .env from .env.example (never overwrites), then add your Groq key"
        run = {
            if (Test-Path .env) { Write-Host ".env already exists: left untouched." }
            else {
                Copy-Item .env.example .env
                Write-Host "Created .env. Put your key after GROQ_API_KEY= (https://console.groq.com/keys)."
                Write-Host ".env is ignored by git: never commit it."
            }
        } }

    # ---------------------------------------------------------------- lancer l'application
    "run" = @{ group = "Launch"; desc = "Web app: Gym Bro planner, live coach, reports, history, profile (http://localhost:8501)"
        run = {
            if (-not (Test-Path .env)) { Write-Host "No .env: the Gym Bro chatbot needs a Groq key (.\tasks.ps1 env)." }
            & $py -m streamlit run frontend/streamlit_app.py
        } }
    "live" = @{ group = "Launch"; desc = "OpenCV webcam demo. Args: [squat|pushup|curl] [camera index or video path]"
        run = { & $py frontend/desktop/live_rep_counter.py --exercise (Get-TaskArg 0 "squat") --source (Get-TaskArg 1 "0") } }
    "demo-video" = @{ group = "Launch"; desc = "Render the backup demo video to data/demo/coach_demo.mp4"
        run = { & $py frontend/desktop/make_demo_video.py } }
    "up" = @{ group = "Launch"; desc = "App + MLflow with Docker (http://localhost:8501, http://localhost:5000)"
        run = {
            if (-not (Test-Path deploy/.env)) {
                Write-Host "Missing deploy/.env: copy deploy/.env.example to deploy/.env and add GROQ_API_KEY."
                exit 1
            }
            docker compose -f deploy/docker-compose.yml up --build
        } }
    "down" = @{ group = "Launch"; desc = "Stop the Docker containers"
        run = { docker compose -f deploy/docker-compose.yml down } }

    # ---------------------------------------------------------------- vérifications
    "test" = @{ group = "Checks"; desc = "Unit tests (offline: Groq is simulated)"
        run = { & $py -m pytest -q } }
    "test-groq" = @{ group = "Checks"; desc = "One real Groq call with your key: prints a chest plan"
        run = {
            $env:RUN_GROQ_LIVE = "1"
            try { & $py -m pytest tests/test_planner.py -k live -s -q -p no:warnings }
            finally { Remove-Item Env:RUN_GROQ_LIVE -ErrorAction SilentlyContinue }
        } }
    "lint" = @{ group = "Checks"; desc = "Ruff on backend, frontend, ml and tests"
        run = { & $py -m ruff check backend frontend ml tests } }

    # ---------------------------------------------------------------- données
    "extract" = @{ group = "Data"; desc = "Mendeley videos -> keypoints + features (data/processed/), resumes"
        run = { & $py ml/data/extract_keypoints.py --workers 6 } }
    "previews" = @{ group = "Data"; desc = "Annotated clips + detection quality report (data/keypoint_previews/)"
        run = { & $py ml/data/render_keypoints.py } }
    "qevd-build" = @{ group = "Data"; desc = "Rebuild the QEVD split zip into data/raw/qevd_fit300k"
        run = { & $py ml/data/qevd_split_zip.py build data data/raw/qevd_fit300k } }
    "qevd-extract" = @{ group = "Data"; desc = "QEVD videos -> keypoints + features"
        run = { & $py ml/data/extract_qevd.py --workers 10 } }
    "reps" = @{ group = "Data"; desc = "Cut the clips into reps -> data/processed/reps.parquet"
        run = { & $py ml/data/extract_rep_summaries.py } }

    # ---------------------------------------------------------------- entraînement
    "train" = @{ group = "Training"; desc = "Train the LSTM classifier and the Random Forest (overwrites data/models/)"
        run = {
            & $py ml/training/train_classifier.py
            if ($LASTEXITCODE -eq 0) { & $py ml/training/train_form_model.py }
        } }
    "retrain" = @{ group = "Training"; desc = "Rebuild reps + train both models into data/models/candidates/"
        run = {
            # Nouveaux modèles dans data/models/candidates/ : à comparer avant de remplacer data/models/.
            & $py ml/data/extract_rep_summaries.py
            if ($LASTEXITCODE -eq 0) { & $py ml/training/train_classifier.py --out data/models/candidates/exercise_lstm.pt }
            if ($LASTEXITCODE -eq 0) { & $py ml/training/train_form_model.py --out data/models/candidates/form_model.joblib }
        } }
    "use-candidates" = @{ group = "Training"; desc = "Make the app use the retrained models"
        run = {
            Copy-Item data/models/candidates/exercise_lstm.pt data/models/exercise_lstm.pt
            Copy-Item data/models/candidates/form_model.joblib data/models/form_model.joblib
            Write-Host "The app now uses the retrained models. Undo: git checkout -- data/models/"
        } }

    # ---------------------------------------------------------------- évaluation
    "eval-reps" = @{ group = "Evaluation"; desc = "Rep counter vs offline peak count"
        run = { & $py ml/evaluation/eval_rep_counter.py } }
    "eval-form" = @{ group = "Evaluation"; desc = "Form rules vs the good/bad labels"
        run = { & $py ml/evaluation/eval_form_rules.py } }
    "eval-qevd" = @{ group = "Evaluation"; desc = "LSTM and form models on QEVD"
        run = { & $py ml/evaluation/eval_qevd.py } }
    "eval-qevd-squat" = @{ group = "Evaluation"; desc = "Squat faults on QEVD front-view clips"
        run = { & $py ml/evaluation/eval_qevd_squat.py } }
}

if ($tasks.Contains($Task)) {
    $global:LASTEXITCODE = 0
    & $tasks[$Task].run
    exit $LASTEXITCODE
}

if ($Task -ne "help") { Write-Host "Unknown task: $Task`n" -ForegroundColor Yellow }
Write-Host "Usage: .\tasks.ps1 <task> [args]"
foreach ($group in ($tasks.Values | ForEach-Object { $_.group } | Select-Object -Unique)) {
    Write-Host "`n$group" -ForegroundColor Cyan
    foreach ($name in $tasks.Keys) {
        if ($tasks[$name].group -eq $group) { Write-Host ("  {0,-16} {1}" -f $name, $tasks[$name].desc) }
    }
}
if ($Task -ne "help") { exit 1 }
