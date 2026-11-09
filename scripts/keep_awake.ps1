# Empêche Windows de se mettre en veille tant que cette fenêtre est ouverte
# (l'écran peut s'éteindre). Utile pendant les longs calculs (extraction, entraînement).
# Lancé par garder_pc_eveille.bat à la racine du projet.
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$power = Add-Type -MemberDefinition $sig -Name Power -Namespace KeepAwake -PassThru

# ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
$flags = [uint32]2147483649
Write-Host "PC maintenu eveille. Fermez cette fenetre pour autoriser la veille."
while ($true) {
    [void]$power::SetThreadExecutionState($flags)
    Start-Sleep -Seconds 60
}
